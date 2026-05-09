"""Test Trajectory Module"""

import sys
sys.path.insert(0, '../src')

from icl_agent.core import Trajectory, TrajectoryCapture


def test_trajectory_creation():
    traj = Trajectory(trajectory_id="test1")
    assert traj.trajectory_id == "test1"
    assert len(traj.reasoning_steps) == 0
    print("PASS: trajectory_creation")


def test_trajectory_capture():
    with TrajectoryCapture() as tc:
        tc.log_input({"question": "test"})
        tc.log_output({"answer": "test answer"})
        tc.set_score(1.0)

    traj = tc.get_trajectory()
    assert traj.input_data["question"] == "test"
    assert traj.score == 1.0
    print("PASS: trajectory_capture")


def test_trajectory_error_handling():
    with TrajectoryCapture() as tc:
        tc.log_input({"data": "test"})
        tc.log_error("TestError", "Test error message")

    traj = tc.get_trajectory()
    assert len(traj.errors) == 1
    assert traj.errors[0]["error_type"] == "TestError"
    print("PASS: trajectory_error_handling")


def test_trajectory_subgoal_working_memory():
    traj = Trajectory(trajectory_id="memory1")
    traj.start_subgoal("parse", "Parse the input", plan_step=1)
    traj.add_observation("Input contains comma-separated values", importance=0.9)
    traj.complete_subgoal(summary="Parsed CSV-style input")

    memory = traj.get_working_memory()
    assert memory["current_subgoal"] is None
    assert memory["subgoal_summaries"]["parse"] == "Parsed CSV-style input"
    assert memory["active_observations"][0]["subgoal_id"] == "parse"

    loaded = Trajectory.from_dict(traj.to_dict())
    assert loaded.subgoal_summaries["parse"] == "Parsed CSV-style input"
    assert loaded.active_observations[0]["content"] == "Input contains comma-separated values"
    print("PASS: trajectory_subgoal_working_memory")


def test_trajectory_capture_working_memory():
    with TrajectoryCapture() as tc:
        tc.start_subgoal("solve", "Solve the task")
        tc.log_observation("Need exact numeric answer", importance=0.8)
        tc.complete_subgoal(summary="Solved using exact arithmetic")

    traj = tc.get_trajectory()
    assert traj.subgoal_summaries["solve"] == "Solved using exact arithmetic"
    assert traj.active_observations[0]["importance"] == 0.8
    print("PASS: trajectory_capture_working_memory")


if __name__ == "__main__":
    print("Running Trajectory module tests...\n")
    test_trajectory_creation()
    test_trajectory_capture()
    test_trajectory_error_handling()
    test_trajectory_subgoal_working_memory()
    test_trajectory_capture_working_memory()
    print("\nAll Trajectory tests passed!")
