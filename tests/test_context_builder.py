"""Test plan-aware context builder."""

import sys
sys.path.insert(0, '../src')

from icl_agent.core import ContextBudget, ContextBuilder, ContextData, Knowledge, Trajectory
from icl_agent.core.knowledge import KnowledgeType


def test_context_builder_respects_budget():
    builder = ContextBuilder(ContextBudget(total_chars=500, system_chars=120, knowledge_chars=120, memory_chars=120, examples_chars=80, plan_chars=80))
    package = builder.build(
        task="Answer the current math question exactly.",
        system_prompt="You are a careful assistant. " * 20,
        plan=["parse", "solve", "verify"],
        knowledge_items=[
            Knowledge("k1", KnowledgeType.RULE, "Double-check arithmetic before final answers. " * 10, "test", confidence=0.9)
        ],
    )

    assert package.total_chars <= 500
    assert "Task" in package.sections
    assert len(package.sections["System"]) <= 120
    print("PASS: context_builder_respects_budget")


def test_context_builder_prioritizes_current_subgoal_memory():
    traj = Trajectory("t1")
    traj.start_subgoal("parse", "Parse the task")
    traj.add_observation("Old parsing note that should be lower priority", importance=0.9)
    traj.complete_subgoal(summary="Parsing completed")
    traj.start_subgoal("verify", "Verify answer")
    traj.add_observation("Current verification requires checking edge case n=0", importance=0.6)

    builder = ContextBuilder(ContextBudget(total_chars=1000, memory_chars=120))
    package = builder.build(
        task="Verify the generated answer.",
        current_subgoal="verify",
        trajectories=[traj],
    )

    memory = package.sections["Working Memory"]
    assert "Current verification" in memory
    assert package.metadata["used_observation_subgoals"][0] == "verify"
    print("PASS: context_builder_prioritizes_current_subgoal_memory")


def test_context_builder_ranks_knowledge_by_retrieval_score():
    k1 = Knowledge("low", KnowledgeType.RULE, "Low relevance rule", "test", confidence=0.99)
    k2 = Knowledge("high", KnowledgeType.RULE, "High relevance rule", "test", confidence=0.70)
    k1.metadata["last_retrieval_score"] = 0.1
    k2.metadata["last_retrieval_score"] = 0.9

    builder = ContextBuilder(ContextBudget(total_chars=1000, knowledge_chars=200))
    package = builder.build(
        task="Use relevant knowledge.",
        knowledge_items=[k1, k2],
    )

    assert package.metadata["knowledge_ids"][0] == "high"
    assert package.sections["Relevant Knowledge"].find("High relevance") < package.sections["Relevant Knowledge"].find("Low relevance")
    print("PASS: context_builder_ranks_knowledge_by_retrieval_score")


if __name__ == "__main__":
    print("Running ContextBuilder tests...\n")
    test_context_builder_respects_budget()
    test_context_builder_prioritizes_current_subgoal_memory()
    test_context_builder_ranks_knowledge_by_retrieval_score()
    print("\nAll ContextBuilder tests passed!")
