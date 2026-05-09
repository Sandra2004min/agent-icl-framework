"""Reflective learning strategy with hypothesis validation."""

from typing import Callable, List, Dict, Any, Optional, Union
import json
import re

from .base import LearningStrategy
from ..core.context import ContextData
from ..core.hypothesis import ReflectionHypothesis
from ..core.knowledge import KnowledgeExtractor


class ReflectiveLearningStrategy(LearningStrategy):
    """Improve agent instructions by reflecting on failed trajectories.

    The strategy now separates reflection into three steps:
    1. generate a structured hypothesis,
    2. validate it,
    3. write accepted knowledge to memory and apply the prompt update.
    """

    def __init__(
        self,
        reflection_lm: Any = None,
        max_failures: int = 10,
        reflection_prompt_template: Optional[str] = None,
        hypothesis_validator: Optional[Callable[..., Union[bool, Dict[str, Any]]]] = None,
        validation_min_delta: float = 0.0,
    ):
        """
        Args:
            reflection_lm: Optional LLM used for reflection.
            max_failures: Maximum number of failed cases to analyze.
            reflection_prompt_template: Custom reflection prompt template.
            hypothesis_validator: Optional callable that validates a proposed
                hypothesis before it is written to the knowledge base.
            validation_min_delta: Minimum score delta required when the
                validator returns score_before and score_after.
        """
        super().__init__(name="ReflectiveLearning")
        self.reflection_lm = reflection_lm
        self.max_failures = max_failures
        self.reflection_prompt_template = reflection_prompt_template or self._default_prompt_template()
        self.hypothesis_validator = hypothesis_validator
        self.validation_min_delta = validation_min_delta
        self.last_hypothesis: Optional[ReflectionHypothesis] = None

    def learn(
        self,
        current_config: Dict[str, Any],
        contexts: List[ContextData],
        failed_contexts: List[ContextData],
        knowledge_extractor: KnowledgeExtractor,
    ) -> Dict[str, Any]:
        """Learn from failures and return an improved config when validated."""
        selected_failures = failed_contexts[:self.max_failures]

        if not selected_failures:
            return current_config.copy()

        reflective_dataset = self._build_reflective_dataset(selected_failures)
        current_instruction = current_config.get("system_prompt", "")
        reflection_prompt = self._generate_reflection_prompt(
            current_instruction,
            reflective_dataset,
        )

        hypothesis = self._generate_reflection_hypothesis(
            reflection_prompt=reflection_prompt,
            current_instruction=current_instruction,
            failed_contexts=selected_failures,
        )
        self.last_hypothesis = hypothesis

        new_config = current_config.copy()
        new_config["system_prompt"] = hypothesis.improved_instruction

        accepted = self._validate_hypothesis(
            hypothesis=hypothesis,
            current_config=current_config,
            proposed_config=new_config,
            contexts=contexts,
            failed_contexts=selected_failures,
        )

        if not accepted:
            return current_config.copy()

        knowledge = knowledge_extractor.extract_from_reflection(
            reflective_data={
                "failures": [ctx.to_dict() for ctx in selected_failures],
                "hypothesis": hypothesis.to_dict(),
                "accepted": hypothesis.accepted,
                "validation_reason": hypothesis.validation_reason,
            },
            improved_instruction=hypothesis.improved_instruction,
        )
        knowledge.metadata["reflection_hypothesis"] = hypothesis.to_dict()
        knowledge.metadata["validated"] = True
        if hypothesis.validation_score_after is not None:
            knowledge.source_score = hypothesis.validation_score_after

        return new_config

    def _build_reflective_dataset(
        self,
        failed_contexts: List[ContextData],
    ) -> List[Dict[str, Any]]:
        """Build a structured reflection dataset from failed contexts."""
        dataset = []

        for ctx in failed_contexts:
            dataset.append({
                "Inputs": ctx.input_data,
                "Generated Outputs": ctx.output_data,
                "Feedback": ctx.feedback,
                "Score": ctx.score,
                "Error Patterns": ctx.error_patterns if ctx.error_patterns else ["No specific errors"],
                "Trajectory ID": ctx.trajectory_id,
            })

        return dataset

    def _generate_reflection_prompt(
        self,
        current_instruction: str,
        reflective_dataset: List[Dict[str, Any]],
    ) -> str:
        """Generate the prompt used for reflection."""
        formatted_examples = self._format_examples(reflective_dataset)
        return self.reflection_prompt_template.format(
            current_instruction=current_instruction,
            failure_examples=formatted_examples,
            num_failures=len(reflective_dataset),
        )

    def _generate_reflection_hypothesis(
        self,
        reflection_prompt: str,
        current_instruction: str,
        failed_contexts: List[ContextData],
    ) -> ReflectionHypothesis:
        """Generate a validation-ready hypothesis."""
        response = self._reflect_with_llm(reflection_prompt)
        parsed = self._parse_hypothesis_response(response)
        evidence_ids = [ctx.trajectory_id for ctx in failed_contexts]

        improved_instruction = parsed.get("improved_instruction") or response
        proposed_rule = parsed.get("proposed_rule") or self._infer_rule_from_instruction(
            current_instruction,
            improved_instruction,
        )

        return ReflectionHypothesis.create(
            hypothesis=parsed.get("hypothesis") or "The current instruction is missing guidance for the observed failure cases.",
            applicable_when=parsed.get("applicable_when") or "Use when future inputs resemble the failed cases or feedback patterns.",
            proposed_rule=proposed_rule,
            improved_instruction=improved_instruction,
            evidence_ids=evidence_ids,
            risk=parsed.get("risk") or "May overfit if the observed failures are noisy or not representative.",
            metadata={
                "raw_response": response,
                "num_failures": len(failed_contexts),
            },
        )

    def _reflect_with_llm(self, prompt: str) -> str:
        """Call the reflection LLM or use a deterministic fallback."""
        if self.reflection_lm is None:
            return self._simple_improvement(prompt)

        try:
            response = self.reflection_lm(prompt)
            return self._extract_instruction(response)
        except Exception as e:
            print(f"LLM reflection failed: {e}")
            return prompt

    def _validate_hypothesis(
        self,
        hypothesis: ReflectionHypothesis,
        current_config: Dict[str, Any],
        proposed_config: Dict[str, Any],
        contexts: List[ContextData],
        failed_contexts: List[ContextData],
    ) -> bool:
        """Validate before writing reflection knowledge."""
        if self.hypothesis_validator is None:
            return self._heuristic_validate_hypothesis(hypothesis)

        result = self.hypothesis_validator(
            current_config=current_config,
            proposed_config=proposed_config,
            hypothesis=hypothesis,
            contexts=contexts,
            failed_contexts=failed_contexts,
        )

        if isinstance(result, dict):
            score_before = result.get("score_before")
            score_after = result.get("score_after")
            accepted = bool(result.get("accepted", False))
            if score_before is not None and score_after is not None:
                accepted = accepted and (score_after - score_before >= self.validation_min_delta)
            hypothesis.mark_validation(
                accepted=accepted,
                reason=result.get("reason", "external_validator"),
                score_before=score_before,
                score_after=score_after,
            )
            return accepted

        accepted = bool(result)
        hypothesis.mark_validation(
            accepted=accepted,
            reason="external_validator_bool",
        )
        return accepted

    def _heuristic_validate_hypothesis(self, hypothesis: ReflectionHypothesis) -> bool:
        """Fallback validation for backward-compatible behavior."""
        text = f"{hypothesis.hypothesis} {hypothesis.proposed_rule} {hypothesis.improved_instruction}".lower()
        generic_phrases = [
            "be careful",
            "try harder",
            "do better",
            "think step by step",
        ]
        has_content = (
            len(hypothesis.improved_instruction.strip()) >= 20
            and len(hypothesis.proposed_rule.strip()) >= 10
        )
        is_generic = any(phrase in text for phrase in generic_phrases) and len(hypothesis.proposed_rule.split()) < 8
        accepted = has_content and not is_generic
        hypothesis.mark_validation(
            accepted=accepted,
            reason="heuristic_accept" if accepted else "heuristic_reject",
        )
        return accepted

    def _parse_hypothesis_response(self, response: str) -> Dict[str, Any]:
        """Parse optional JSON reflection output into hypothesis fields."""
        text = response.strip()
        if not text:
            return {}

        code_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        candidates = code_blocks + [text]

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            candidates.append(json_match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                return parsed

        return {}

    def _infer_rule_from_instruction(
        self,
        current_instruction: str,
        improved_instruction: str,
    ) -> str:
        """Infer a compact rule when only an improved instruction is returned."""
        if improved_instruction.startswith(current_instruction):
            suffix = improved_instruction[len(current_instruction):].strip()
            if suffix:
                return suffix[:500]
        return improved_instruction[:500]

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format examples as Markdown text."""
        formatted = []

        for i, example in enumerate(examples, 1):
            ex_str = f"## Example {i}\n\n"

            for key, value in example.items():
                ex_str += f"### {key}\n"

                if isinstance(value, dict):
                    for k, v in value.items():
                        ex_str += f"- **{k}**: {v}\n"
                elif isinstance(value, list):
                    for item in value:
                        ex_str += f"- {item}\n"
                else:
                    ex_str += f"{value}\n"

                ex_str += "\n"

            formatted.append(ex_str)

        return "\n".join(formatted)

    def _extract_instruction(self, llm_response: str) -> str:
        """Extract a code-block response when present."""
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)\n```", llm_response, re.DOTALL)

        if code_blocks:
            return code_blocks[0].strip()

        return llm_response.strip()

    def _simple_improvement(self, original: str) -> str:
        """Simple fallback improvement when no LLM is available."""
        improvements = [
            "Be more precise in your responses.",
            "Double-check your calculations.",
            "Consider edge cases.",
        ]

        return original + "\n\nAdditional guidelines:\n" + "\n".join(f"- {imp}" for imp in improvements)

    def _default_prompt_template(self) -> str:
        """Default reflection prompt template."""
        return """You are an expert at improving AI agent instructions based on failure analysis.

I have an AI agent with the following instruction:

```
{current_instruction}
```

The agent was tested on {num_failures} cases and failed. Here are the details:

{failure_examples}

Your task:
1. Analyze the failure patterns
2. Identify what knowledge or guidelines are missing from the current instruction
3. Propose an improved instruction that addresses these failures
4. Express the improvement as a validation-ready hypothesis

Requirements:
- Keep the core purpose of the instruction
- Add specific guidelines to avoid the observed failures
- Be concise but comprehensive
- Format the improved instruction clearly

Return a JSON object with these fields:
- hypothesis
- applicable_when
- proposed_rule
- improved_instruction
- risk

If you cannot produce JSON, provide the improved instruction within a code block (```).
"""


if __name__ == "__main__":
    from ..core.context import ContextData
    from ..core.knowledge import KnowledgeExtractor

    ctx = ContextData(
        trajectory_id="test1",
        input_data={"question": "What is 2+2?"},
        output_data={"answer": "5"},
        score=0.0,
        is_failure=True,
        feedback="Incorrect answer. Expected: 4, Got: 5",
    )

    strategy = ReflectiveLearningStrategy()
    current_config = {"system_prompt": "You are a math assistant."}
    extractor = KnowledgeExtractor()

    improved_config = strategy.learn(
        current_config=current_config,
        contexts=[ctx],
        failed_contexts=[ctx],
        knowledge_extractor=extractor,
    )

    print("Improved config:")
    print(improved_config["system_prompt"])
