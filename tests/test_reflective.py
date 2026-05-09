"""Test ReflectiveLearningStrategy hypothesis validation."""

import json
import sys
sys.path.insert(0, '../src')

from icl_agent.core.context import ContextData
from icl_agent.core.knowledge import KnowledgeExtractor
from icl_agent.strategies import ReflectiveLearningStrategy


def _make_failure():
    return ContextData(
        trajectory_id="fail-1",
        input_data={"question": "What is 2+2?"},
        output_data={"answer": "5"},
        score=0.0,
        is_failure=True,
        feedback="Expected 4, got 5",
        error_patterns=["MathError"],
    )


def _json_reflection(_prompt):
    return json.dumps({
        "hypothesis": "The agent misses arithmetic verification.",
        "applicable_when": "Arithmetic questions with exact numeric answers.",
        "proposed_rule": "Double-check arithmetic before finalizing numeric answers.",
        "improved_instruction": "You are a math assistant. Double-check arithmetic before answering.",
        "risk": "May add slight latency on simple questions.",
    })


def test_reflective_accepts_validated_hypothesis():
    def validator(**kwargs):
        return {
            "accepted": True,
            "score_before": 0.0,
            "score_after": 1.0,
            "reason": "validation improved score",
        }

    strategy = ReflectiveLearningStrategy(
        reflection_lm=_json_reflection,
        hypothesis_validator=validator,
        validation_min_delta=0.1,
    )
    ke = KnowledgeExtractor()
    failure = _make_failure()

    new_config = strategy.learn(
        {"system_prompt": "You are a math assistant."},
        contexts=[failure],
        failed_contexts=[failure],
        knowledge_extractor=ke,
    )

    assert "Double-check arithmetic" in new_config["system_prompt"]
    assert strategy.last_hypothesis.accepted is True
    assert strategy.last_hypothesis.validation_score_after == 1.0
    assert len(ke.get_knowledge_base().knowledge_items) == 1
    knowledge = ke.get_knowledge_base().knowledge_items[0]
    assert knowledge.metadata["validated"] is True
    assert knowledge.metadata["reflection_hypothesis"]["accepted"] is True
    assert "fail-1" in knowledge.evidence_ids
    print("PASS: reflective_accepts_validated_hypothesis")


def test_reflective_rejects_unvalidated_hypothesis():
    def validator(**kwargs):
        return {
            "accepted": False,
            "score_before": 0.0,
            "score_after": 0.0,
            "reason": "no improvement",
        }

    strategy = ReflectiveLearningStrategy(
        reflection_lm=_json_reflection,
        hypothesis_validator=validator,
        validation_min_delta=0.1,
    )
    ke = KnowledgeExtractor()
    failure = _make_failure()
    original_config = {"system_prompt": "You are a math assistant."}

    new_config = strategy.learn(
        original_config,
        contexts=[failure],
        failed_contexts=[failure],
        knowledge_extractor=ke,
    )

    assert new_config["system_prompt"] == original_config["system_prompt"]
    assert strategy.last_hypothesis.accepted is False
    assert len(ke.get_knowledge_base().knowledge_items) == 0
    print("PASS: reflective_rejects_unvalidated_hypothesis")


def test_reflective_heuristic_preserves_default_write():
    strategy = ReflectiveLearningStrategy(reflection_lm=_json_reflection)
    ke = KnowledgeExtractor()
    failure = _make_failure()

    strategy.learn(
        {"system_prompt": "You are a math assistant."},
        contexts=[failure],
        failed_contexts=[failure],
        knowledge_extractor=ke,
    )

    assert strategy.last_hypothesis.accepted is True
    assert len(ke.get_knowledge_base().knowledge_items) == 1
    print("PASS: reflective_heuristic_preserves_default_write")


if __name__ == "__main__":
    print("Running Reflective tests...\n")
    test_reflective_accepts_validated_hypothesis()
    test_reflective_rejects_unvalidated_hypothesis()
    test_reflective_heuristic_preserves_default_write()
    print("\nAll Reflective tests passed!")
