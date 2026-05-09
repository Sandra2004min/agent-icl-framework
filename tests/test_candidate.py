"""Test prompt candidate pool utilities."""

import sys
sys.path.insert(0, '../src')

from icl_agent.core.candidate import PromptCandidate, PromptCandidatePool


def _candidate(cid, prompt, score, coverage, length):
    return PromptCandidate(
        candidate_id=cid,
        agent_config={"system_prompt": prompt},
        score=score,
        failure_coverage=coverage,
        token_length=length,
    )


def test_candidate_from_config():
    candidate = PromptCandidate.from_config(
        {"system_prompt": "short useful prompt"},
        mutation_reason="test",
        score=0.8,
    )

    assert candidate.system_prompt == "short useful prompt"
    assert candidate.token_length == 3
    assert candidate.mutation_reason == "test"
    print("PASS: candidate_from_config")


def test_candidate_pool_pareto_front_keeps_tradeoffs():
    pool = PromptCandidatePool(max_size=4)
    pool.add(_candidate("a", "high score prompt", 0.80, 0.80, 100))
    pool.add(_candidate("b", "compact broad prompt", 0.75, 0.95, 80))
    pool.add(_candidate("c", "dominated prompt", 0.70, 0.50, 120))

    front_ids = [candidate.candidate_id for candidate in pool.get_pareto_front()]
    assert "a" in front_ids
    assert "b" in front_ids
    assert "c" not in front_ids
    print("PASS: candidate_pool_pareto_front_keeps_tradeoffs")


def test_candidate_pool_deduplicates_by_prompt():
    pool = PromptCandidatePool(max_size=4)
    pool.add(_candidate("old", "same prompt", 0.50, 0.20, 10))
    pool.add(_candidate("new", "same prompt", 0.90, 0.20, 10))

    assert len(pool.candidates) == 1
    assert pool.candidates[0].candidate_id == "new"
    print("PASS: candidate_pool_deduplicates_by_prompt")


if __name__ == "__main__":
    print("Running Candidate tests...\n")
    test_candidate_from_config()
    test_candidate_pool_pareto_front_keeps_tradeoffs()
    test_candidate_pool_deduplicates_by_prompt()
    print("\nAll Candidate tests passed!")
