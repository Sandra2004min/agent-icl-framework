"""
Test Trajectory Module
测试轨迹模块
"""

import sys
sys.path.insert(0, '../src')

from icl_agent.core import Trajectory, TrajectoryCapture


def test_trajectory_creation():
    """测试轨迹创建"""
    traj = Trajectory(trajectory_id="test1")
    assert traj.trajectory_id == "test1"
    assert len(traj.reasoning_steps) == 0
    print("✓ 轨迹创建测试通过")


def test_trajectory_capture():
    """测试轨迹捕获"""
    with TrajectoryCapture() as tc:
        tc.log_input({"question": "test"})
        tc.log_output({"answer": "test answer"})
        tc.set_score(1.0)

    traj = tc.get_trajectory()
    assert traj.input_data["question"] == "test"
    assert traj.score == 1.0
    print("✓ 轨迹捕获测试通过")


def test_trajectory_error_handling():
    """测试错误处理"""
    with TrajectoryCapture() as tc:
        tc.log_input({"data": "test"})
        tc.log_error("TestError", "Test error message")

    traj = tc.get_trajectory()
    assert len(traj.errors) == 1
    assert traj.errors[0]["error_type"] == "TestError"
    print("✓ 错误处理测试通过")


if __name__ == "__main__":
    print("运行Trajectory模块测试...\n")

    test_trajectory_creation()
    test_trajectory_capture()
    test_trajectory_error_handling()

    print("\n所有测试通过！✓")
