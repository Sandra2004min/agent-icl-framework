"""Plan-aware context construction with lightweight budget control."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .context import ContextData
from .knowledge import Knowledge
from .trajectory import Trajectory


@dataclass
class ContextBudget:
    """Character budgets for context sections."""

    total_chars: int = 4000
    system_chars: int = 800
    knowledge_chars: int = 1200
    memory_chars: int = 1000
    examples_chars: int = 800
    plan_chars: int = 400


@dataclass
class ContextPackage:
    """Structured context package that can be rendered as a prompt."""

    sections: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        chunks = []
        for title, content in self.sections.items():
            if content:
                chunks.append(f"## {title}\n{content}")
        return "\n\n".join(chunks)

    @property
    def total_chars(self) -> int:
        return len(self.to_prompt())


class ContextBuilder:
    """Build compact, plan-aware context from memory, knowledge, and examples."""

    def __init__(self, budget: Optional[ContextBudget] = None):
        self.budget = budget or ContextBudget()

    def build(
        self,
        task: str,
        system_prompt: str = "",
        plan: Optional[List[str]] = None,
        current_subgoal: Optional[str] = None,
        knowledge_items: Optional[List[Knowledge]] = None,
        trajectories: Optional[List[Trajectory]] = None,
        examples: Optional[List[ContextData]] = None,
        budget: Optional[ContextBudget] = None,
    ) -> ContextPackage:
        """Build a structured prompt package under a character budget."""
        active_budget = budget or self.budget
        sections: Dict[str, str] = {}
        metadata: Dict[str, Any] = {
            "budget": active_budget.__dict__.copy(),
            "current_subgoal": current_subgoal,
        }

        sections["Task"] = _clip(task, max(0, active_budget.total_chars // 5))
        sections["System"] = _clip(system_prompt, active_budget.system_chars)

        plan_text = self._format_plan(plan or [], current_subgoal)
        sections["Plan"] = _clip(plan_text, active_budget.plan_chars)

        knowledge_text, knowledge_ids = self._format_knowledge(
            knowledge_items or [],
            active_budget.knowledge_chars,
        )
        sections["Relevant Knowledge"] = knowledge_text
        metadata["knowledge_ids"] = knowledge_ids

        memory_text, memory_meta = self._format_working_memory(
            trajectories or [],
            current_subgoal,
            active_budget.memory_chars,
        )
        sections["Working Memory"] = memory_text
        metadata.update(memory_meta)

        examples_text, example_ids = self._format_examples(
            examples or [],
            active_budget.examples_chars,
        )
        sections["Examples"] = examples_text
        metadata["example_ids"] = example_ids

        package = ContextPackage(sections=sections, metadata=metadata)
        self._enforce_total_budget(package, active_budget.total_chars)
        package.metadata["total_chars"] = package.total_chars
        return package

    def _format_plan(self, plan: List[str], current_subgoal: Optional[str]) -> str:
        if not plan:
            return ""

        lines = []
        for idx, step in enumerate(plan, 1):
            marker = "*" if current_subgoal and current_subgoal in step else "-"
            lines.append(f"{marker} {idx}. {step}")
        return "\n".join(lines)

    def _format_knowledge(
        self,
        knowledge_items: List[Knowledge],
        budget: int,
    ) -> (str, List[str]):
        ranked = sorted(
            knowledge_items,
            key=lambda item: (
                item.metadata.get("last_retrieval_score", 0.0),
                item.confidence,
                item.importance,
                item.success_rate,
            ),
            reverse=True,
        )
        lines = []
        used_ids = []
        for item in ranked:
            line = f"- [{item.knowledge_type.value}, confidence={item.confidence:.2f}] {item.content}"
            if _would_fit(lines, line, budget):
                lines.append(line)
                used_ids.append(item.knowledge_id)
            elif not lines and budget > 0:
                lines.append(_clip(line, budget))
                used_ids.append(item.knowledge_id)
                break
        return "\n".join(lines), used_ids

    def _format_working_memory(
        self,
        trajectories: List[Trajectory],
        current_subgoal: Optional[str],
        budget: int,
    ) -> (str, Dict[str, Any]):
        observations = []
        summaries = []

        for trajectory in trajectories:
            for subgoal_id, summary in trajectory.subgoal_summaries.items():
                summaries.append({
                    "trajectory_id": trajectory.trajectory_id,
                    "subgoal_id": subgoal_id,
                    "summary": summary,
                    "priority": 1.0 if subgoal_id == current_subgoal else 0.4,
                })

            for observation in trajectory.active_observations:
                subgoal_id = observation.get("subgoal_id")
                observations.append({
                    "trajectory_id": trajectory.trajectory_id,
                    "subgoal_id": subgoal_id,
                    "content": observation.get("content", ""),
                    "importance": observation.get("importance", 0.5),
                    "timestamp": observation.get("timestamp", ""),
                    "priority": (1.0 if subgoal_id == current_subgoal else 0.0) + observation.get("importance", 0.5),
                })

        observations.sort(key=lambda item: (item["priority"], item["timestamp"]), reverse=True)
        summaries.sort(key=lambda item: item["priority"], reverse=True)

        lines = []
        used_observations = []
        for observation in observations:
            line = f"- Observation[{observation['subgoal_id']}]: {observation['content']}"
            if _would_fit(lines, line, budget):
                lines.append(line)
                used_observations.append(observation)

        for summary in summaries:
            line = f"- Summary[{summary['subgoal_id']}]: {summary['summary']}"
            if _would_fit(lines, line, budget):
                lines.append(line)

        return "\n".join(lines), {
            "num_memory_observations": len(used_observations),
            "used_observation_subgoals": [item["subgoal_id"] for item in used_observations],
        }

    def _format_examples(
        self,
        examples: List[ContextData],
        budget: int,
    ) -> (str, List[str]):
        ranked = sorted(examples, key=lambda ctx: (ctx.score, not ctx.is_failure), reverse=True)
        lines = []
        used_ids = []

        for ctx in ranked:
            label = "failure" if ctx.is_failure else "success"
            line = (
                f"- [{label}, score={ctx.score:.2f}] "
                f"Input: {ctx.input_data} Output: {ctx.output_data}"
            )
            if ctx.feedback:
                line += f" Feedback: {ctx.feedback}"
            if _would_fit(lines, line, budget):
                lines.append(line)
                used_ids.append(ctx.trajectory_id)
            elif not lines and budget > 0:
                lines.append(_clip(line, budget))
                used_ids.append(ctx.trajectory_id)
                break

        return "\n".join(lines), used_ids

    def _enforce_total_budget(self, package: ContextPackage, total_chars: int):
        if package.total_chars <= total_chars:
            return

        priority = [
            "Task",
            "System",
            "Plan",
            "Relevant Knowledge",
            "Working Memory",
            "Examples",
        ]
        remaining = total_chars
        new_sections: Dict[str, str] = {}
        for title in priority:
            content = package.sections.get(title, "")
            if not content:
                new_sections[title] = ""
                continue
            section_overhead = len(f"## {title}\n\n")
            allowance = max(0, remaining - section_overhead)
            clipped = _clip(content, allowance)
            new_sections[title] = clipped
            remaining -= len(clipped) + section_overhead
            if remaining <= 0:
                break
        package.sections = new_sections


def _would_fit(lines: List[str], new_line: str, budget: int) -> bool:
    if budget <= 0:
        return False
    current = len("\n".join(lines))
    extra = len(new_line) + (1 if lines else 0)
    return current + extra <= budget


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."
