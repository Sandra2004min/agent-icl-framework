"""Test Context Module"""

import sys
sys.path.insert(0, '../src')

from icl_agent.core.context import ContextData, ContextAnalyzer
from icl_agent.core.trajectory import Trajectory, TrajectoryBatch


def test_context_data_creation():
    ctx = ContextData(
        trajectory_id="t1",
        input_data={"question": "What is 2+2?"},
        output_data={"answer": "4"},
        score=1.0,
        is_failure=False,
    )
    assert ctx.trajectory_id == "t1"
    assert ctx.score == 1.0
    assert ctx.is_failure is False
    assert ctx.error_patterns == []
    print("PASS: context_data_creation")


def test_context_data_to_dict():
    ctx = ContextData(
        trajectory_id="t2",
        input_data={"question": "test"},
        output_data={"answer": "ans"},
        score=0.5,
        is_failure=True,
        error_patterns=["format_error"],
        feedback="wrong format",
    )
    d = ctx.to_dict()
    assert d["trajectory_id"] == "t2"
    assert d["is_failure"] is True
    assert "format_error" in d["error_patterns"]
    assert d["feedback"] == "wrong format"
    print("PASS: context_data_to_dict")


def test_context_analyzer_basic():
    analyzer = ContextAnalyzer(failure_threshold=0.5)

    traj1 = Trajectory(trajectory_id="t1")
    traj1.input_data = {"question": "q1"}
    traj1.output_data = {"answer": "a1"}
    traj1.score = 1.0

    traj2 = Trajectory(trajectory_id="t2")
    traj2.input_data = {"question": "q2"}
    traj2.output_data = {"answer": "wrong"}
    traj2.score = 0.0

    contexts = analyzer.analyze_batch([traj1, traj2])
    assert len(contexts) == 2

    failures = [c for c in contexts if c.is_failure]
    successes = [c for c in contexts if not c.is_failure]
    assert len(failures) == 1
    assert len(successes) == 1
    assert failures[0].score == 0.0
    print("PASS: context_analyzer_basic")


if __name__ == "__main__":
    print("Running Context module tests...\n")
    test_context_data_creation()
    test_context_data_to_dict()
    test_context_analyzer_basic()
    print("\nAll Context tests passed!")
