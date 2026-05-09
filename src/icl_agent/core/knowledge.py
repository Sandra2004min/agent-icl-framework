"""
Knowledge Module

Stores reusable knowledge extracted from trajectories, reflections,
examples, and retrieval results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json
import re


class KnowledgeType(Enum):
    """Supported knowledge categories."""

    REFLECTION = "reflection"
    EXAMPLE = "example"
    RETRIEVAL = "retrieval"
    RULE = "rule"


@dataclass
class Knowledge:
    """A reusable knowledge item learned from agent experience."""

    knowledge_id: str
    knowledge_type: KnowledgeType
    content: str
    source: str
    confidence: float = 1.0
    usage_count: int = 0
    success_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: Optional[datetime] = None
    source_task: str = ""
    source_score: float = 0.0
    failure_signature: str = ""
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "knowledge_id": self.knowledge_id,
            "knowledge_type": self.knowledge_type.value,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "metadata": self.metadata,
            "importance": self.importance,
            "created_at": _serialize_datetime(self.created_at),
            "last_used_at": _serialize_datetime(self.last_used_at),
            "source_task": self.source_task,
            "source_score": self.source_score,
            "failure_signature": self.failure_signature,
            "evidence_ids": self.evidence_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Knowledge":
        """Create a Knowledge object from serialized data.

        Older saved knowledge files may not contain the memory fields added for
        relevance-aware retrieval, so defaults are supplied here.
        """
        item = data.copy()
        item["knowledge_type"] = KnowledgeType(item["knowledge_type"])
        item["created_at"] = _parse_datetime(item.get("created_at")) or datetime.now()
        item["last_used_at"] = _parse_datetime(item.get("last_used_at"))
        item.setdefault("metadata", {})
        item.setdefault("importance", 0.5)
        item.setdefault("source_task", "")
        item.setdefault("source_score", 0.0)
        item.setdefault("failure_signature", "")
        item.setdefault("evidence_ids", [])
        return cls(**item)

    def update_usage(self, success: bool):
        """Update usage statistics after downstream evaluation."""
        self.usage_count += 1
        self.last_used_at = datetime.now()
        self.success_rate = (
            (self.success_rate * (self.usage_count - 1) + (1.0 if success else 0.0))
            / self.usage_count
        )

    def touch(self):
        """Record that this item was retrieved."""
        self.last_used_at = datetime.now()


class KnowledgeBase:
    """In-memory store for learned knowledge."""

    def __init__(self):
        self.knowledge_items: List[Knowledge] = []
        self._index_by_type: Dict[KnowledgeType, List[Knowledge]] = {
            kt: [] for kt in KnowledgeType
        }

    def add(self, knowledge: Knowledge):
        """Add a knowledge item."""
        self.knowledge_items.append(knowledge)
        self._index_by_type[knowledge.knowledge_type].append(knowledge)

    def get_by_type(self, knowledge_type: KnowledgeType) -> List[Knowledge]:
        """Get knowledge items by type."""
        return self._index_by_type[knowledge_type]

    def get_top_k(self, k: int = 5, by: str = "confidence") -> List[Knowledge]:
        """Get top-k knowledge by a single quality signal."""
        if by == "confidence":
            sorted_items = sorted(self.knowledge_items, key=lambda x: x.confidence, reverse=True)
        elif by == "success_rate":
            sorted_items = sorted(self.knowledge_items, key=lambda x: x.success_rate, reverse=True)
        elif by == "usage_count":
            sorted_items = sorted(self.knowledge_items, key=lambda x: x.usage_count, reverse=True)
        elif by == "importance":
            sorted_items = sorted(self.knowledge_items, key=lambda x: x.importance, reverse=True)
        else:
            sorted_items = self.knowledge_items

        return sorted_items[:k]

    def filter_by_confidence(self, min_confidence: float = 0.5) -> List[Knowledge]:
        """Filter knowledge items by confidence."""
        return [k for k in self.knowledge_items if k.confidence >= min_confidence]

    def retrieve(
        self,
        query: Union[str, Dict[str, Any], Any],
        k: int = 5,
        min_confidence: float = 0.0,
        weights: Optional[Dict[str, float]] = None,
        touch: bool = True,
    ) -> List[Knowledge]:
        """Retrieve knowledge relevant to the current query/context.

        The ranking combines textual relevance with quality and memory signals:
        similarity, confidence, success rate, importance, and recency.
        """
        candidates = [
            item for item in self.knowledge_items
            if item.confidence >= min_confidence
        ]
        if not candidates or k <= 0:
            return []

        query_text = _query_to_text(query)
        similarities = _compute_text_similarities(
            query_text,
            [_knowledge_to_text(item) for item in candidates],
        )

        score_weights = {
            "similarity": 0.45,
            "confidence": 0.20,
            "success_rate": 0.15,
            "importance": 0.10,
            "recency": 0.10,
        }
        if weights:
            score_weights.update(weights)

        now = datetime.now()
        scored_items = []
        for item, similarity in zip(candidates, similarities):
            recency = _recency_score(item, now)
            score = (
                score_weights["similarity"] * similarity
                + score_weights["confidence"] * _clamp01(item.confidence)
                + score_weights["success_rate"] * _clamp01(item.success_rate)
                + score_weights["importance"] * _clamp01(item.importance)
                + score_weights["recency"] * recency
            )
            item.metadata["last_retrieval_score"] = round(score, 6)
            item.metadata["last_retrieval_similarity"] = round(similarity, 6)
            scored_items.append((score, similarity, item.confidence, item))

        scored_items.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        results = [item for _, _, _, item in scored_items[:k]]

        if touch:
            for item in results:
                item.touch()

        return results

    def save_to_file(self, filepath: str):
        """Save the knowledge base to disk."""
        data = [k.to_dict() for k in self.knowledge_items]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> "KnowledgeBase":
        """Load a knowledge base from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        kb = cls()
        for item in data:
            kb.add(Knowledge.from_dict(item))

        return kb


class KnowledgeExtractor:
    """Extract and store knowledge from different learning signals."""

    def __init__(self):
        self.knowledge_base = KnowledgeBase()

    def extract_from_reflection(
        self,
        reflective_data: Dict[str, Any],
        improved_instruction: str,
    ) -> Knowledge:
        """Extract knowledge from reflection results."""
        import uuid

        failures = reflective_data.get("failures", [])
        insights = self._analyze_reflection(reflective_data)
        evidence_ids = [
            str(f.get("trajectory_id"))
            for f in failures
            if isinstance(f, dict) and f.get("trajectory_id")
        ]
        failure_signature = "; ".join(insights[:3])

        knowledge = Knowledge(
            knowledge_id=str(uuid.uuid4()),
            knowledge_type=KnowledgeType.REFLECTION,
            content=improved_instruction,
            source="reflection",
            confidence=0.8,
            importance=0.7,
            source_task=reflective_data.get("task", ""),
            failure_signature=failure_signature,
            evidence_ids=evidence_ids,
            metadata={
                "insights": insights,
                "num_failures_analyzed": len(failures),
            },
        )

        self.knowledge_base.add(knowledge)
        return knowledge

    def extract_from_examples(
        self,
        examples: List[Dict[str, Any]],
        context: str = "",
    ) -> Knowledge:
        """Extract example knowledge from successful demonstrations."""
        import uuid

        formatted_examples = self._format_examples(examples)
        scores = [
            float(ex["score"])
            for ex in examples
            if isinstance(ex, dict) and isinstance(ex.get("score"), (int, float))
        ]
        evidence_ids = [
            str(ex.get("trajectory_id"))
            for ex in examples
            if isinstance(ex, dict) and ex.get("trajectory_id")
        ]

        knowledge = Knowledge(
            knowledge_id=str(uuid.uuid4()),
            knowledge_type=KnowledgeType.EXAMPLE,
            content=formatted_examples,
            source="few_shot_examples",
            confidence=0.9,
            importance=0.65,
            source_task=context,
            source_score=sum(scores) / len(scores) if scores else 0.0,
            evidence_ids=evidence_ids,
            metadata={
                "num_examples": len(examples),
                "context": context,
            },
        )

        self.knowledge_base.add(knowledge)
        return knowledge

    def extract_from_retrieval(
        self,
        retrieved_docs: List[str],
        query: str,
    ) -> Knowledge:
        """Extract knowledge from retrieved documents."""
        import uuid

        combined_content = self._combine_retrieval_results(retrieved_docs)

        knowledge = Knowledge(
            knowledge_id=str(uuid.uuid4()),
            knowledge_type=KnowledgeType.RETRIEVAL,
            content=combined_content,
            source="retrieval",
            confidence=0.7,
            importance=0.5,
            source_task=query,
            metadata={
                "num_docs": len(retrieved_docs),
                "query": query,
            },
        )

        self.knowledge_base.add(knowledge)
        return knowledge

    def extract_rules(self, patterns: Dict[str, Any]) -> List[Knowledge]:
        """Extract rule knowledge from discovered patterns."""
        import uuid

        rules = []

        for pattern_name, pattern_data in patterns.items():
            rule_content = self._formulate_rule(pattern_name, pattern_data)

            knowledge = Knowledge(
                knowledge_id=str(uuid.uuid4()),
                knowledge_type=KnowledgeType.RULE,
                content=rule_content,
                source="pattern_analysis",
                confidence=pattern_data.get("confidence", 0.7),
                importance=pattern_data.get("importance", 0.6),
                source_task=pattern_name,
                failure_signature=pattern_data.get("failure_signature", ""),
                metadata={
                    "pattern_name": pattern_name,
                    "pattern_data": pattern_data,
                },
            )

            rules.append(knowledge)
            self.knowledge_base.add(knowledge)

        return rules

    def get_knowledge_base(self) -> KnowledgeBase:
        """Get the underlying knowledge base."""
        return self.knowledge_base

    def _analyze_reflection(self, reflective_data: Dict[str, Any]) -> List[str]:
        """Analyze reflection data and extract key insights."""
        insights = []
        failures = reflective_data.get("failures", [])

        if failures:
            insights.append(f"Analyzed {len(failures)} failure cases")

            error_types = []
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                if failure.get("error_type"):
                    error_types.append(failure["error_type"])
                error_types.extend(failure.get("error_patterns", []))

            from collections import Counter

            common_errors = Counter(error_types).most_common(3)
            if common_errors:
                insights.append(f"Common errors: {', '.join(e[0] for e in common_errors)}")

        return insights

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format examples as text."""
        formatted = []

        for i, example in enumerate(examples, 1):
            ex_str = f"Example {i}:\n"
            for key, value in example.items():
                ex_str += f"  {key}: {value}\n"
            formatted.append(ex_str)

        return "\n".join(formatted)

    def _combine_retrieval_results(self, docs: List[str]) -> str:
        """Combine retrieved documents into a compact knowledge block."""
        return "\n\n---\n\n".join(docs[:5])

    def _formulate_rule(self, pattern_name: str, pattern_data: Dict[str, Any]) -> str:
        """Create a natural-language rule from a discovered pattern."""
        return f"Rule from pattern '{pattern_name}': {pattern_data.get('description', 'No description')}"


def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _clamp01(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _query_to_text(query: Union[str, Dict[str, Any], Any]) -> str:
    if isinstance(query, str):
        return query
    if isinstance(query, dict):
        return _dict_to_text(query)

    parts = []
    for attr in ("input_data", "output_data", "feedback", "reasoning_summary", "error_patterns"):
        if hasattr(query, attr):
            parts.append(str(getattr(query, attr)))
    return " ".join(parts) if parts else str(query)


def _knowledge_to_text(knowledge: Knowledge) -> str:
    parts = [
        knowledge.content,
        knowledge.source,
        knowledge.source_task,
        knowledge.failure_signature,
    ]
    if knowledge.metadata:
        parts.append(_dict_to_text(knowledge.metadata))
    return " ".join(str(part) for part in parts if part)


def _dict_to_text(data: Dict[str, Any]) -> str:
    parts = []
    for key, value in data.items():
        parts.append(str(key))
        if isinstance(value, dict):
            parts.append(_dict_to_text(value))
        elif isinstance(value, list):
            parts.append(" ".join(_dict_to_text(v) if isinstance(v, dict) else str(v) for v in value))
        else:
            parts.append(str(value))
    return " ".join(parts)


def _compute_text_similarities(query_text: str, documents: List[str]) -> List[float]:
    if not documents:
        return []
    if not query_text.strip():
        return [0.0 for _ in documents]

    from collections import Counter
    import math

    tokenized = [_tokenize(query_text)] + [_tokenize(doc) for doc in documents]
    if not tokenized[0]:
        return [0.0 for _ in documents]

    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    total_docs = len(tokenized)
    idf = {
        token: math.log((1 + total_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }

    query_vec = _tfidf_vector(tokenized[0], idf)
    return [
        _vector_cosine(query_vec, _tfidf_vector(tokens, idf))
        for tokens in tokenized[1:]
    ]


def _token_cosine_similarity(text1: str, text2: str) -> float:
    from collections import Counter
    import math

    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)
    if not tokens1 or not tokens2:
        return 0.0

    counts1 = Counter(tokens1)
    counts2 = Counter(tokens2)
    overlap = set(counts1) & set(counts2)
    numerator = sum(counts1[token] * counts2[token] for token in overlap)
    denom1 = math.sqrt(sum(count * count for count in counts1.values()))
    denom2 = math.sqrt(sum(count * count for count in counts2.values()))
    return numerator / (denom1 * denom2) if denom1 and denom2 else 0.0


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    from collections import Counter

    if not tokens:
        return {}

    counts = Counter(tokens)
    total = float(len(tokens))
    return {
        token: (count / total) * idf.get(token, 1.0)
        for token, count in counts.items()
    }


def _vector_cosine(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    import math

    if not vec1 or not vec2:
        return 0.0

    overlap = set(vec1) & set(vec2)
    numerator = sum(vec1[token] * vec2[token] for token in overlap)
    denom1 = math.sqrt(sum(value * value for value in vec1.values()))
    denom2 = math.sqrt(sum(value * value for value in vec2.values()))
    return numerator / (denom1 * denom2) if denom1 and denom2 else 0.0


def _recency_score(knowledge: Knowledge, now: datetime) -> float:
    ref_time = knowledge.last_used_at or knowledge.created_at
    if not isinstance(ref_time, datetime):
        return 0.5
    age_days = max(0.0, (now - ref_time).total_seconds() / 86400.0)
    return 1.0 / (1.0 + age_days / 30.0)
