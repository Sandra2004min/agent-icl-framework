"""Prompt candidate tracking and Pareto selection."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class PromptCandidate:
    """A prompt/config candidate produced during optimization."""

    candidate_id: str
    agent_config: Dict[str, Any]
    parent_ids: List[str] = field(default_factory=list)
    mutation_reason: str = ""
    score: float = 0.0
    token_length: int = 0
    failure_coverage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def system_prompt(self) -> str:
        return self.agent_config.get("system_prompt", "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "agent_config": self.agent_config,
            "parent_ids": self.parent_ids,
            "mutation_reason": self.mutation_reason,
            "score": self.score,
            "token_length": self.token_length,
            "failure_coverage": self.failure_coverage,
            "metadata": self.metadata,
        }

    @classmethod
    def from_config(
        cls,
        agent_config: Dict[str, Any],
        parent_ids: Optional[List[str]] = None,
        mutation_reason: str = "",
        score: float = 0.0,
        failure_coverage: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PromptCandidate":
        prompt = agent_config.get("system_prompt", "")
        return cls(
            candidate_id=str(uuid.uuid4()),
            agent_config=agent_config.copy(),
            parent_ids=parent_ids or [],
            mutation_reason=mutation_reason,
            score=score,
            token_length=_estimate_token_length(prompt),
            failure_coverage=failure_coverage,
            metadata=metadata or {},
        )


class PromptCandidatePool:
    """Stores prompt candidates and keeps a bounded Pareto front."""

    def __init__(self, max_size: int = 8):
        self.max_size = max_size
        self.candidates: List[PromptCandidate] = []

    def add(self, candidate: PromptCandidate):
        self.candidates.append(candidate)
        self._deduplicate_by_prompt()
        self.prune()

    def get_best(self) -> Optional[PromptCandidate]:
        if not self.candidates:
            return None
        return sorted(
            self.candidates,
            key=lambda c: (c.score, c.failure_coverage, -c.token_length),
            reverse=True,
        )[0]

    def get_pareto_front(self) -> List[PromptCandidate]:
        front = []
        for candidate in self.candidates:
            if not any(_dominates(other, candidate) for other in self.candidates if other is not candidate):
                front.append(candidate)

        return sorted(
            front,
            key=lambda c: (c.score, c.failure_coverage, -c.token_length),
            reverse=True,
        )

    def prune(self):
        if len(self.candidates) <= self.max_size:
            return

        front = self.get_pareto_front()
        kept = front[: self.max_size]

        if len(kept) < self.max_size:
            kept_ids = {candidate.candidate_id for candidate in kept}
            remaining = [
                candidate for candidate in self.candidates
                if candidate.candidate_id not in kept_ids
            ]
            remaining.sort(
                key=lambda c: (c.score, c.failure_coverage, -c.token_length),
                reverse=True,
            )
            kept.extend(remaining[: self.max_size - len(kept)])

        self.candidates = kept

    def to_dict(self) -> Dict[str, Any]:
        front = self.get_pareto_front()
        return {
            "max_size": self.max_size,
            "num_candidates": len(self.candidates),
            "pareto_front_ids": [candidate.candidate_id for candidate in front],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    def _deduplicate_by_prompt(self):
        by_prompt: Dict[str, PromptCandidate] = {}
        for candidate in self.candidates:
            prompt = candidate.system_prompt
            existing = by_prompt.get(prompt)
            if existing is None or _candidate_sort_key(candidate) > _candidate_sort_key(existing):
                by_prompt[prompt] = candidate
        self.candidates = list(by_prompt.values())


def _dominates(left: PromptCandidate, right: PromptCandidate) -> bool:
    """Return True when left is no worse in all objectives and better in one."""
    no_worse = (
        left.score >= right.score
        and left.failure_coverage >= right.failure_coverage
        and left.token_length <= right.token_length
    )
    strictly_better = (
        left.score > right.score
        or left.failure_coverage > right.failure_coverage
        or left.token_length < right.token_length
    )
    return no_worse and strictly_better


def _candidate_sort_key(candidate: PromptCandidate):
    return (candidate.score, candidate.failure_coverage, -candidate.token_length)


def _estimate_token_length(text: str) -> int:
    if not text:
        return 0
    return len(text.split())
