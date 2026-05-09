"""Reflection hypotheses and validation records."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ReflectionHypothesis:
    """A structured reflection that must pass validation before storage."""

    hypothesis_id: str
    hypothesis: str
    applicable_when: str
    proposed_rule: str
    improved_instruction: str
    evidence_ids: List[str] = field(default_factory=list)
    risk: str = ""
    validation_score_before: Optional[float] = None
    validation_score_after: Optional[float] = None
    accepted: bool = False
    validation_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        hypothesis: str,
        applicable_when: str,
        proposed_rule: str,
        improved_instruction: str,
        evidence_ids: Optional[List[str]] = None,
        risk: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ReflectionHypothesis":
        return cls(
            hypothesis_id=str(uuid.uuid4()),
            hypothesis=hypothesis,
            applicable_when=applicable_when,
            proposed_rule=proposed_rule,
            improved_instruction=improved_instruction,
            evidence_ids=evidence_ids or [],
            risk=risk,
            metadata=metadata or {},
        )

    def mark_validation(
        self,
        accepted: bool,
        reason: str,
        score_before: Optional[float] = None,
        score_after: Optional[float] = None,
    ):
        self.accepted = accepted
        self.validation_reason = reason
        self.validation_score_before = score_before
        self.validation_score_after = score_after

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis": self.hypothesis,
            "applicable_when": self.applicable_when,
            "proposed_rule": self.proposed_rule,
            "improved_instruction": self.improved_instruction,
            "evidence_ids": self.evidence_ids,
            "risk": self.risk,
            "validation_score_before": self.validation_score_before,
            "validation_score_after": self.validation_score_after,
            "accepted": self.accepted,
            "validation_reason": self.validation_reason,
            "metadata": self.metadata,
        }
