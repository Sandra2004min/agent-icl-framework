"""
Trajectory Module - 轨迹捕获模块

负责捕获智能体执行过程中的所有关键信息，包括：
- 输入输出
- 推理步骤
- 工具调用
- 错误信息
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Trajectory:
    """
    单次执行的轨迹记录

    Attributes:
        trajectory_id: 轨迹唯一标识
        timestamp: 执行时间戳
        input_data: 输入数据
        output_data: 输出数据
        reasoning_steps: 推理步骤列表
        tool_calls: 工具调用记录
        errors: 错误信息
        metadata: 其他元数据
        score: 执行得分（如果已评估）
    """

    trajectory_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    subgoals: List[Dict[str, Any]] = field(default_factory=list)
    current_subgoal: Optional[str] = None
    subgoal_summaries: Dict[str, str] = field(default_factory=dict)
    active_observations: List[Dict[str, Any]] = field(default_factory=list)

    def add_reasoning_step(self, step: str, detail: Optional[Dict[str, Any]] = None):
        """添加推理步骤"""
        self.reasoning_steps.append({
            "step": step,
            "detail": detail or {},
            "timestamp": datetime.now().isoformat()
        })

    def add_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool = True
    ):
        """添加工具调用记录"""
        self.tool_calls.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })

    def add_error(self, error_type: str, message: str, detail: Optional[Dict] = None):
        """添加错误信息"""
        self.errors.append({
            "error_type": error_type,
            "message": message,
            "detail": detail or {},
            "timestamp": datetime.now().isoformat()
        })

    def start_subgoal(
        self,
        subgoal_id: str,
        description: str,
        plan_step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Start or activate a subgoal for hierarchical working memory."""
        existing = self._find_subgoal(subgoal_id)
        if existing is None:
            self.subgoals.append({
                "subgoal_id": subgoal_id,
                "description": description,
                "plan_step": plan_step,
                "status": "active",
                "observations": [],
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "completed_at": None,
            })
        else:
            existing["description"] = description
            existing["status"] = "active"
            existing["metadata"] = metadata or existing.get("metadata", {})

        self.current_subgoal = subgoal_id

    def complete_subgoal(
        self,
        subgoal_id: Optional[str] = None,
        summary: str = "",
        status: str = "completed",
    ):
        """Complete a subgoal and preserve a compact summary."""
        target_id = subgoal_id or self.current_subgoal
        if target_id is None:
            return

        subgoal = self._find_subgoal(target_id)
        if subgoal is not None:
            subgoal["status"] = status
            subgoal["summary"] = summary
            subgoal["completed_at"] = datetime.now().isoformat()

        if summary:
            self.subgoal_summaries[target_id] = summary

        if self.current_subgoal == target_id:
            self.current_subgoal = None

    def add_observation(
        self,
        content: str,
        subgoal_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add an observation to active working memory."""
        target_id = subgoal_id or self.current_subgoal
        observation = {
            "subgoal_id": target_id,
            "content": content,
            "importance": importance,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.active_observations.append(observation)

        if target_id is not None:
            subgoal = self._find_subgoal(target_id)
            if subgoal is not None:
                subgoal.setdefault("observations", []).append(observation.copy())

    def get_working_memory(self) -> Dict[str, Any]:
        """Return compact hierarchical working-memory state."""
        return {
            "current_subgoal": self.current_subgoal,
            "subgoals": self.subgoals,
            "subgoal_summaries": self.subgoal_summaries,
            "active_observations": self.active_observations,
        }

    def is_successful(self) -> bool:
        """判断执行是否成功"""
        return len(self.errors) == 0 and self.score is not None and self.score > 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "trajectory_id": self.trajectory_id,
            "timestamp": self.timestamp.isoformat(),
            "input_data": self.input_data,
            "output_data": self.output_data,
            "reasoning_steps": self.reasoning_steps,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "metadata": self.metadata,
            "score": self.score,
            "subgoals": self.subgoals,
            "current_subgoal": self.current_subgoal,
            "subgoal_summaries": self.subgoal_summaries,
            "active_observations": self.active_observations,
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """从字典创建轨迹对象"""
        item = data.copy()
        item["timestamp"] = datetime.fromisoformat(item["timestamp"])
        item.setdefault("subgoals", [])
        item.setdefault("current_subgoal", None)
        item.setdefault("subgoal_summaries", {})
        item.setdefault("active_observations", [])
        return cls(**item)

    def _find_subgoal(self, subgoal_id: str) -> Optional[Dict[str, Any]]:
        for subgoal in self.subgoals:
            if subgoal.get("subgoal_id") == subgoal_id:
                return subgoal
        return None


class TrajectoryCapture:
    """
    轨迹捕获器

    用于在智能体执行过程中自动捕获轨迹信息
    支持上下文管理器模式
    """

    def __init__(self, trajectory_id: Optional[str] = None):
        """
        初始化轨迹捕获器

        Args:
            trajectory_id: 轨迹ID，如果不提供则自动生成
        """
        import uuid
        self.trajectory_id = trajectory_id or str(uuid.uuid4())
        self.trajectory = Trajectory(trajectory_id=self.trajectory_id)
        self._active = False

    def __enter__(self) -> "TrajectoryCapture":
        """进入上下文"""
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        self._active = False
        if exc_type is not None:
            # 捕获异常信息
            self.trajectory.add_error(
                error_type=exc_type.__name__,
                message=str(exc_val),
                detail={"traceback": str(exc_tb)}
            )

    def log_input(self, input_data: Dict[str, Any]):
        """记录输入数据"""
        self.trajectory.input_data = input_data

    def log_output(self, output_data: Dict[str, Any]):
        """记录输出数据"""
        self.trajectory.output_data = output_data

    def log_reasoning(self, step: str, detail: Optional[Dict[str, Any]] = None):
        """记录推理步骤"""
        self.trajectory.add_reasoning_step(step, detail)

    def log_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool = True
    ):
        """记录工具调用"""
        self.trajectory.add_tool_call(tool_name, arguments, result, success)

    def log_error(self, error_type: str, message: str, detail: Optional[Dict] = None):
        """记录错误"""
        self.trajectory.add_error(error_type, message, detail)

    def start_subgoal(
        self,
        subgoal_id: str,
        description: str,
        plan_step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """记录子目标开始"""
        self.trajectory.start_subgoal(subgoal_id, description, plan_step, metadata)

    def complete_subgoal(
        self,
        subgoal_id: Optional[str] = None,
        summary: str = "",
        status: str = "completed",
    ):
        """记录子目标完成"""
        self.trajectory.complete_subgoal(subgoal_id, summary, status)

    def log_observation(
        self,
        content: str,
        subgoal_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """记录工作记忆观察"""
        self.trajectory.add_observation(content, subgoal_id, importance, metadata)

    def set_score(self, score: float):
        """设置得分"""
        self.trajectory.score = score

    def set_metadata(self, key: str, value: Any):
        """设置元数据"""
        self.trajectory.metadata[key] = value

    def get_trajectory(self) -> Trajectory:
        """获取轨迹对象"""
        return self.trajectory

    @staticmethod
    def capture_function(func):
        """
        装饰器：自动捕获函数执行轨迹

        使用方法:
            @TrajectoryCapture.capture_function
            def my_agent_function(input_data):
                # ... agent logic
                return output
        """
        def wrapper(*args, **kwargs):
            with TrajectoryCapture() as tc:
                # 记录输入
                tc.log_input({
                    "args": args,
                    "kwargs": kwargs
                })

                try:
                    # 执行函数
                    result = func(*args, **kwargs)

                    # 记录输出
                    tc.log_output({"result": result})

                    return result
                except Exception as e:
                    tc.log_error(
                        error_type=type(e).__name__,
                        message=str(e)
                    )
                    raise
                finally:
                    # 返回轨迹（可选）
                    pass

        return wrapper


class TrajectoryBatch:
    """
    批量轨迹管理

    用于管理多个轨迹，支持批量操作和统计分析
    """

    def __init__(self):
        self.trajectories: List[Trajectory] = []

    def add(self, trajectory: Trajectory):
        """添加轨迹"""
        self.trajectories.append(trajectory)

    def filter_successful(self) -> List[Trajectory]:
        """筛选成功的轨迹"""
        return [t for t in self.trajectories if t.is_successful()]

    def filter_failed(self) -> List[Trajectory]:
        """筛选失败的轨迹"""
        return [t for t in self.trajectories if not t.is_successful()]

    def get_average_score(self) -> float:
        """计算平均得分"""
        scores = [t.score for t in self.trajectories if t.score is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def get_success_rate(self) -> float:
        """计算成功率"""
        if not self.trajectories:
            return 0.0
        successful = len(self.filter_successful())
        return successful / len(self.trajectories)

    def to_list(self) -> List[Dict[str, Any]]:
        """转换为字典列表"""
        return [t.to_dict() for t in self.trajectories]

    def save_to_file(self, filepath: str):
        """保存到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_list(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> "TrajectoryBatch":
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        batch = cls()
        for item in data:
            batch.add(Trajectory.from_dict(item))

        return batch


# 示例使用
if __name__ == "__main__":
    # 示例1: 使用上下文管理器
    with TrajectoryCapture() as tc:
        tc.log_input({"question": "What is 2+2?"})
        tc.log_reasoning("分析问题", {"type": "arithmetic"})
        tc.log_output({"answer": "4"})
        tc.set_score(1.0)

    trajectory = tc.get_trajectory()
    print("轨迹ID:", trajectory.trajectory_id)
    print("是否成功:", trajectory.is_successful())
    print(trajectory.to_json())

    # 示例2: 使用装饰器
    @TrajectoryCapture.capture_function
    def simple_agent(question):
        # 模拟智能体逻辑
        return f"Answer to: {question}"

    result = simple_agent("What is AI?")
    print(result)
