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


if __name__ == "__main__":
    print("Running Trajectory module tests...\n")
    test_trajectory_creation()
    test_trajectory_capture()
    test_trajectory_error_handling()
    print("\nAll Trajectory tests passed!")
