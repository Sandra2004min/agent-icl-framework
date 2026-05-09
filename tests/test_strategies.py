"""Test Learning Strategies (FewShot, Retrieval)"""

import sys
sys.path.insert(0, '../src')

from icl_agent.strategies import FewShotLearningStrategy, RetrievalLearningStrategy
from icl_agent.core.context_builder import ContextBudget
from icl_agent.core.context import ContextData
from icl_agent.core.knowledge import Knowledge, KnowledgeExtractor, KnowledgeType


def _make_context(tid, question, answer, score, is_failure=False, feedback=""):
    return ContextData(
        trajectory_id=tid,
        input_data={"question": question},
        output_data={"answer": answer},
        score=score,
        is_failure=is_failure,
        feedback=feedback,
    )


# ─── FewShot Tests ───

def test_fewshot_basic():
    strategy = FewShotLearningStrategy(
        num_shots=2, include_reasoning=False, include_negative=False
    )
    config = {"system_prompt": "You are a helper."}
    ke = KnowledgeExtractor()

    contexts = [
        _make_context("t1", "What is 2+2?", "4", 1.0),
        _make_context("t2", "What is 3+3?", "6", 1.0),
    ]
    failed = []

    new_config = strategy.learn(config, contexts, failed, ke)
    prompt = new_config["system_prompt"]
    assert "Example 1" in prompt
    assert "You are a helper." in prompt
    print("PASS: fewshot_basic")


def test_fewshot_with_negative():
    strategy = FewShotLearningStrategy(
        num_shots=2, include_reasoning=False, include_negative=True, max_negative=1
    )
    config = {"system_prompt": "Answer questions."}
    ke = KnowledgeExtractor()

    contexts = [
        _make_context("t1", "q1", "correct", 1.0),
    ]
    failed = [
        _make_context("t2", "q2", "wrong", 0.0, is_failure=True, feedback="Incorrect"),
    ]

    new_config = strategy.learn(config, contexts, failed, ke)
    prompt = new_config["system_prompt"]
    assert "Mistake 1" in prompt
    assert "INCORRECT" in prompt
    print("PASS: fewshot_with_negative")


def test_fewshot_no_examples():
    strategy = FewShotLearningStrategy(num_shots=3, include_reasoning=False, include_negative=False)
    config = {"system_prompt": "original prompt"}
    ke = KnowledgeExtractor()

    new_config = strategy.learn(config, [], [], ke)
    assert new_config["system_prompt"] == "original prompt"
    print("PASS: fewshot_no_examples")


def test_fewshot_diversity_selection():
    strategy = FewShotLearningStrategy(num_shots=2, include_reasoning=False, include_negative=False)

    contexts = [
        _make_context("t1", "same question", "a1", 1.0),
        _make_context("t2", "same question", "a2", 0.9),
        _make_context("t3", "different question", "a3", 0.8),
    ]

    selected = strategy._select_diverse_examples(contexts, n=2)
    inputs = [str(c.input_data)[:100] for c in selected]
    assert len(set(inputs)) == 2
    print("PASS: fewshot_diversity_selection")


# ─── Retrieval Tests ───

def test_retrieval_conservative_fallback():
    strategy = RetrievalLearningStrategy(
        top_k=3, auto_extract=False, min_confidence=0.7
    )
    config = {"system_prompt": "original prompt"}
    ke = KnowledgeExtractor()

    new_config = strategy.learn(config, [], [], ke)
    assert new_config["system_prompt"] == "original prompt"
    print("PASS: retrieval_conservative_fallback")


def test_retrieval_base_prompt_preserved():
    strategy = RetrievalLearningStrategy(
        top_k=3, auto_extract=False, min_confidence=0.7
    )
    config = {"system_prompt": "my base prompt"}
    ke = KnowledgeExtractor()

    strategy.learn(config, [], [], ke)
    assert strategy._base_prompt == "my base prompt"

    config2 = {"system_prompt": "modified prompt"}
    strategy.learn(config2, [], [], ke)
    assert strategy._base_prompt == "my base prompt"
    print("PASS: retrieval_base_prompt_preserved")


def test_retrieval_noise_threshold():
    strategy = RetrievalLearningStrategy(
        top_k=3, auto_extract=True, noise_threshold=0.7, reflection_lm=None
    )
    config = {"system_prompt": "test"}
    ke = KnowledgeExtractor()

    contexts = [_make_context(f"t{i}", f"q{i}", "wrong", 0.0, is_failure=True) for i in range(8)]
    contexts += [_make_context("t_ok1", "q_ok", "right", 1.0)]
    failed = [c for c in contexts if c.is_failure]

    strategy.learn(config, contexts, failed, ke)
    # fail_ratio = 8/9 > 0.7, should NOT extract knowledge
    kb = ke.get_knowledge_base()
    assert len(kb.knowledge_items) == 0
    print("PASS: retrieval_noise_threshold")


def test_retrieval_uses_relevant_knowledge():
    strategy = RetrievalLearningStrategy(
        top_k=1, auto_extract=False, min_confidence=0.7
    )
    config = {"system_prompt": "Complete the function."}
    ke = KnowledgeExtractor()
    kb = ke.get_knowledge_base()
    kb.add(Knowledge(
        "k_unrelated",
        KnowledgeType.RULE,
        "For history questions, answer with the full official person name.",
        "test",
        confidence=0.95,
    ))
    kb.add(Knowledge(
        "k_relevant",
        KnowledgeType.RULE,
        "For Python list sorting tasks, preserve odd indices and sort only even positions.",
        "test",
        confidence=0.75,
        importance=0.8,
    ))

    contexts = [
        _make_context(
            "t1",
            "Implement sort_even so even-indexed values are sorted and odd-indexed values stay fixed.",
            "wrong code",
            0.0,
            is_failure=True,
            feedback="The odd indices were changed.",
        )
    ]

    new_config = strategy.learn(config, contexts, contexts, ke)
    prompt = new_config["system_prompt"]
    assert "preserve odd indices" in prompt
    assert "history questions" not in prompt
    assert strategy.last_context_package is not None
    assert "Relevant Knowledge" in strategy.last_context_package.sections
    print("PASS: retrieval_uses_relevant_knowledge")


def test_retrieval_context_builder_budget_limits_knowledge():
    strategy = RetrievalLearningStrategy(
        top_k=2,
        auto_extract=False,
        min_confidence=0.7,
        max_guidelines_chars=80,
        context_budget=ContextBudget(
            total_chars=500,
            system_chars=80,
            knowledge_chars=80,
            memory_chars=0,
            examples_chars=80,
            plan_chars=80,
        ),
    )
    config = {"system_prompt": "Base prompt."}
    ke = KnowledgeExtractor()
    kb = ke.get_knowledge_base()
    kb.add(Knowledge(
        "k1",
        KnowledgeType.RULE,
        "For Python sorting tasks, preserve odd indices and sort even positions. " * 10,
        "test",
        confidence=0.9,
    ))

    contexts = [
        _make_context(
            "t1",
            "Sort even-indexed values while preserving odd indices.",
            "wrong",
            0.0,
            is_failure=True,
            feedback="odd indices changed",
        )
    ]

    new_config = strategy.learn(config, contexts, contexts, ke)
    assert strategy.last_context_package is not None
    assert len(strategy.last_context_package.sections["Relevant Knowledge"]) <= 80
    assert len(new_config["system_prompt"]) < 700
    print("PASS: retrieval_context_builder_budget_limits_knowledge")


def test_retrieval_can_use_legacy_guidelines_format():
    strategy = RetrievalLearningStrategy(
        top_k=1,
        auto_extract=False,
        min_confidence=0.7,
        use_context_builder=False,
    )
    config = {"system_prompt": "Base prompt."}
    ke = KnowledgeExtractor()
    ke.get_knowledge_base().add(Knowledge(
        "k1",
        KnowledgeType.RULE,
        "Use modulo arithmetic for day offsets.",
        "test",
        confidence=0.9,
    ))
    contexts = [_make_context("t1", "What day is 8 days after Monday?", "wrong", 0.0, is_failure=True)]

    new_config = strategy.learn(config, contexts, contexts, ke)
    assert "Important guidelines learned from past experience" in new_config["system_prompt"]
    assert strategy.last_context_package is None
    print("PASS: retrieval_can_use_legacy_guidelines_format")


if __name__ == "__main__":
    print("Running Strategy tests...\n")

    print("--- FewShot ---")
    test_fewshot_basic()
    test_fewshot_with_negative()
    test_fewshot_no_examples()
    test_fewshot_diversity_selection()

    print("\n--- Retrieval ---")
    test_retrieval_conservative_fallback()
    test_retrieval_base_prompt_preserved()
    test_retrieval_noise_threshold()
    test_retrieval_uses_relevant_knowledge()
    test_retrieval_context_builder_budget_limits_knowledge()
    test_retrieval_can_use_legacy_guidelines_format()

    print("\nAll Strategy tests passed!")
