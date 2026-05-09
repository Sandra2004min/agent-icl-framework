"""Test Knowledge Module"""

import sys
import os
import json
import tempfile
sys.path.insert(0, '../src')

from icl_agent.core.knowledge import Knowledge, KnowledgeBase, KnowledgeType, KnowledgeExtractor


def test_knowledge_creation():
    k = Knowledge(
        knowledge_id="k1",
        knowledge_type=KnowledgeType.RULE,
        content="Always check input format",
        source="test",
        confidence=0.9,
    )
    assert k.knowledge_id == "k1"
    assert k.knowledge_type == KnowledgeType.RULE
    assert k.confidence == 0.9
    assert k.usage_count == 0
    assert k.importance == 0.5
    assert k.evidence_ids == []
    print("PASS: knowledge_creation")


def test_knowledge_to_dict():
    k = Knowledge(
        knowledge_id="k2",
        knowledge_type=KnowledgeType.EXAMPLE,
        content="example content",
        source="test",
    )
    d = k.to_dict()
    assert d["knowledge_type"] == "example"
    assert d["content"] == "example content"
    assert "created_at" in d
    assert d["importance"] == 0.5
    print("PASS: knowledge_to_dict")


def test_knowledge_update_usage():
    k = Knowledge(
        knowledge_id="k3",
        knowledge_type=KnowledgeType.REFLECTION,
        content="test",
        source="test",
    )
    k.update_usage(success=True)
    assert k.usage_count == 1
    assert k.success_rate == 1.0

    k.update_usage(success=False)
    assert k.usage_count == 2
    assert k.success_rate == 0.5

    k.update_usage(success=True)
    assert k.usage_count == 3
    assert abs(k.success_rate - 2.0/3.0) < 1e-6
    print("PASS: knowledge_update_usage")


def test_knowledge_base_add_and_query():
    kb = KnowledgeBase()
    k1 = Knowledge("k1", KnowledgeType.RULE, "rule1", "test", confidence=0.9)
    k2 = Knowledge("k2", KnowledgeType.EXAMPLE, "ex1", "test", confidence=0.5)
    k3 = Knowledge("k3", KnowledgeType.RULE, "rule2", "test", confidence=0.7)

    kb.add(k1)
    kb.add(k2)
    kb.add(k3)

    assert len(kb.knowledge_items) == 3
    assert len(kb.get_by_type(KnowledgeType.RULE)) == 2
    assert len(kb.get_by_type(KnowledgeType.EXAMPLE)) == 1
    assert len(kb.get_by_type(KnowledgeType.RETRIEVAL)) == 0
    print("PASS: knowledge_base_add_and_query")


def test_knowledge_base_top_k():
    kb = KnowledgeBase()
    kb.add(Knowledge("k1", KnowledgeType.RULE, "low", "test", confidence=0.3))
    kb.add(Knowledge("k2", KnowledgeType.RULE, "high", "test", confidence=0.9))
    kb.add(Knowledge("k3", KnowledgeType.RULE, "mid", "test", confidence=0.6))

    top2 = kb.get_top_k(k=2, by="confidence")
    assert len(top2) == 2
    assert top2[0].content == "high"
    assert top2[1].content == "mid"
    print("PASS: knowledge_base_top_k")


def test_knowledge_base_filter():
    kb = KnowledgeBase()
    kb.add(Knowledge("k1", KnowledgeType.RULE, "low", "test", confidence=0.3))
    kb.add(Knowledge("k2", KnowledgeType.RULE, "high", "test", confidence=0.9))
    kb.add(Knowledge("k3", KnowledgeType.RULE, "mid", "test", confidence=0.6))

    filtered = kb.filter_by_confidence(0.5)
    assert len(filtered) == 2
    assert all(k.confidence >= 0.5 for k in filtered)
    print("PASS: knowledge_base_filter")


def test_knowledge_base_save_load():
    kb = KnowledgeBase()
    kb.add(Knowledge("k1", KnowledgeType.RULE, "rule content", "test", confidence=0.8))
    kb.add(Knowledge("k2", KnowledgeType.EXAMPLE, "example content", "test", confidence=0.6))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmp_path = f.name

    try:
        kb.save_to_file(tmp_path)
        loaded_kb = KnowledgeBase.load_from_file(tmp_path)
        assert len(loaded_kb.knowledge_items) == 2
        assert loaded_kb.knowledge_items[0].content == "rule content"
        assert loaded_kb.knowledge_items[1].knowledge_type == KnowledgeType.EXAMPLE
        assert loaded_kb.knowledge_items[0].created_at is not None
        print("PASS: knowledge_base_save_load")
    finally:
        os.unlink(tmp_path)


def test_knowledge_base_loads_legacy_json():
    legacy_data = [{
        "knowledge_id": "legacy",
        "knowledge_type": "rule",
        "content": "legacy rule",
        "source": "test",
        "confidence": 0.8,
        "usage_count": 0,
        "success_rate": 0.0,
        "metadata": {},
    }]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmp_path = f.name
        json.dump(legacy_data, f)

    try:
        loaded_kb = KnowledgeBase.load_from_file(tmp_path)
        assert len(loaded_kb.knowledge_items) == 1
        assert loaded_kb.knowledge_items[0].importance == 0.5
        assert loaded_kb.knowledge_items[0].evidence_ids == []
        print("PASS: knowledge_base_loads_legacy_json")
    finally:
        os.unlink(tmp_path)


def test_knowledge_base_retrieve_prefers_relevance():
    kb = KnowledgeBase()
    kb.add(Knowledge(
        "k_unrelated",
        KnowledgeType.RULE,
        "For historical biography questions, provide complete names and dates.",
        "test",
        confidence=0.95,
        importance=0.5,
    ))
    kb.add(Knowledge(
        "k_relevant",
        KnowledgeType.RULE,
        "For Python list sorting tasks, preserve odd indices and sort only even positions.",
        "test",
        confidence=0.75,
        importance=0.8,
    ))

    results = kb.retrieve(
        "Implement a Python function that sorts values at even indices while preserving odd indices.",
        k=1,
        min_confidence=0.7,
    )

    assert len(results) == 1
    assert results[0].knowledge_id == "k_relevant"
    assert results[0].last_used_at is not None
    assert "last_retrieval_score" in results[0].metadata
    print("PASS: knowledge_base_retrieve_prefers_relevance")


def test_knowledge_base_retrieve_filters_low_confidence():
    kb = KnowledgeBase()
    kb.add(Knowledge(
        "k_low",
        KnowledgeType.RULE,
        "For Python list sorting tasks, preserve odd indices and sort only even positions.",
        "test",
        confidence=0.4,
    ))
    kb.add(Knowledge(
        "k_ok",
        KnowledgeType.RULE,
        "For arithmetic questions, double-check calculations.",
        "test",
        confidence=0.8,
    ))

    results = kb.retrieve("Python list sorting even indices", k=5, min_confidence=0.7)
    ids = [item.knowledge_id for item in results]
    assert "k_low" not in ids
    assert "k_ok" in ids
    print("PASS: knowledge_base_retrieve_filters_low_confidence")


def test_knowledge_extractor():
    ke = KnowledgeExtractor()
    kb = ke.get_knowledge_base()
    assert len(kb.knowledge_items) == 0

    ke.extract_from_examples(
        examples=[{"input_data": {"question": "q1"}, "output_data": {"answer": "a1"}, "score": 1.0}],
        context="test_context"
    )
    assert len(kb.knowledge_items) >= 1
    print("PASS: knowledge_extractor")


if __name__ == "__main__":
    print("Running Knowledge module tests...\n")
    test_knowledge_creation()
    test_knowledge_to_dict()
    test_knowledge_update_usage()
    test_knowledge_base_add_and_query()
    test_knowledge_base_top_k()
    test_knowledge_base_filter()
    test_knowledge_base_save_load()
    test_knowledge_base_loads_legacy_json()
    test_knowledge_base_retrieve_prefers_relevance()
    test_knowledge_base_retrieve_filters_low_confidence()
    test_knowledge_extractor()
    print("\nAll Knowledge tests passed!")
