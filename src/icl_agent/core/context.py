"""
Context Module - 上下文分析模块

负责分析执行轨迹，提取关键的上下文信息：
- 识别失败案例
- 提取错误模式
- 计算上下文相似度
- 生成反馈信息
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import Counter

from .trajectory import Trajectory, TrajectoryBatch


@dataclass
class ContextData:
    """
    上下文数据

    包含从轨迹中提取的关键信息
    """

    # 基本信息
    trajectory_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    score: float

    # 分析结果
    is_failure: bool
    error_patterns: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    feedback: str = ""

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "trajectory_id": self.trajectory_id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "score": self.score,
            "is_failure": self.is_failure,
            "error_patterns": self.error_patterns,
            "reasoning_summary": self.reasoning_summary,
            "feedback": self.feedback,
            "metadata": self.metadata,
        }


class ContextAnalyzer:
    """
    上下文分析器

    从轨迹中提取和分析上下文信息
    """

    def __init__(
        self,
        failure_threshold: float = 0.5,
        similarity_threshold: float = 0.8
    ):
        """
        初始化上下文分析器

        Args:
            failure_threshold: 失败判定阈值（分数低于此值视为失败）
            similarity_threshold: 相似度阈值
        """
        self.failure_threshold = failure_threshold
        self.similarity_threshold = similarity_threshold

    def analyze_trajectory(self, trajectory: Trajectory) -> ContextData:
        """
        分析单个轨迹

        Args:
            trajectory: 待分析的轨迹

        Returns:
            ContextData: 分析后的上下文数据
        """
        # 判断是否失败
        is_failure = self._is_failure(trajectory)

        # 提取错误模式
        error_patterns = self._extract_error_patterns(trajectory)

        # 总结推理过程
        reasoning_summary = self._summarize_reasoning(trajectory)

        # 生成反馈
        feedback = self._generate_feedback(trajectory, is_failure)

        return ContextData(
            trajectory_id=trajectory.trajectory_id,
            input_data=trajectory.input_data,
            output_data=trajectory.output_data,
            score=trajectory.score if trajectory.score is not None else 0.0,
            is_failure=is_failure,
            error_patterns=error_patterns,
            reasoning_summary=reasoning_summary,
            feedback=feedback,
            metadata=trajectory.metadata.copy()
        )

    def analyze_batch(self, trajectories: List[Trajectory]) -> List[ContextData]:
        """
        批量分析轨迹

        Args:
            trajectories: 轨迹列表

        Returns:
            List[ContextData]: 上下文数据列表
        """
        return [self.analyze_trajectory(t) for t in trajectories]

    def identify_failures(
        self,
        trajectories: List[Trajectory],
        scores: Optional[List[float]] = None
    ) -> List[Trajectory]:
        """
        识别失败案例

        Args:
            trajectories: 轨迹列表
            scores: 可选的得分列表（如果轨迹中没有score）

        Returns:
            List[Trajectory]: 失败案例列表
        """
        failures = []

        for i, traj in enumerate(trajectories):
            score = traj.score if traj.score is not None else (scores[i] if scores else 0.0)
            if score < self.failure_threshold or len(traj.errors) > 0:
                failures.append(traj)

        return failures

    def extract_error_patterns(
        self,
        failed_trajectories: List[Trajectory]
    ) -> Dict[str, int]:
        """
        提取错误模式

        Args:
            failed_trajectories: 失败的轨迹列表

        Returns:
            Dict[str, int]: 错误模式及其出现频率
        """
        error_types = []

        for traj in failed_trajectories:
            for error in traj.errors:
                error_types.append(error["error_type"])

        # 统计频率
        return dict(Counter(error_types))

    def compute_context_similarity(
        self,
        context1: ContextData,
        context2: ContextData
    ) -> float:
        """
        计算两个上下文的相似度

        Args:
            context1: 上下文1
            context2: 上下文2

        Returns:
            float: 相似度分数 (0-1)
        """
        # 简单实现：基于输入数据的文本相似度
        # 实际应用中可以使用更复杂的方法（如embedding相似度）

        def dict_to_text(d: Dict) -> str:
            """将字典转换为文本"""
            return " ".join(str(v) for v in d.values())

        text1 = dict_to_text(context1.input_data)
        text2 = dict_to_text(context2.input_data)

        # 简单的jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def group_similar_contexts(
        self,
        contexts: List[ContextData],
        similarity_threshold: Optional[float] = None
    ) -> List[List[ContextData]]:
        """
        将相似的上下文分组

        Args:
            contexts: 上下文列表
            similarity_threshold: 相似度阈值

        Returns:
            List[List[ContextData]]: 分组后的上下文
        """
        threshold = similarity_threshold or self.similarity_threshold

        groups = []
        used = set()

        for i, ctx1 in enumerate(contexts):
            if i in used:
                continue

            group = [ctx1]
            used.add(i)

            for j, ctx2 in enumerate(contexts[i + 1:], start=i + 1):
                if j in used:
                    continue

                similarity = self.compute_context_similarity(ctx1, ctx2)
                if similarity >= threshold:
                    group.append(ctx2)
                    used.add(j)

            groups.append(group)

        return groups

    def summarize_context_group(
        self,
        contexts: List[ContextData]
    ) -> Dict[str, Any]:
        """
        总结一组相似上下文的共同特征

        Args:
            contexts: 上下文列表

        Returns:
            Dict: 总结信息
        """
        if not contexts:
            return {}

        # 收集错误模式
        all_errors = []
        for ctx in contexts:
            all_errors.extend(ctx.error_patterns)

        error_frequency = dict(Counter(all_errors))

        # 计算平均分数
        avg_score = sum(ctx.score for ctx in contexts) / len(contexts)

        # 失败率
        failure_rate = sum(1 for ctx in contexts if ctx.is_failure) / len(contexts)

        return {
            "num_contexts": len(contexts),
            "avg_score": avg_score,
            "failure_rate": failure_rate,
            "common_errors": error_frequency,
            "sample_inputs": [ctx.input_data for ctx in contexts[:3]],
            "sample_outputs": [ctx.output_data for ctx in contexts[:3]],
        }

    # ===== 私有辅助方法 =====

    def _is_failure(self, trajectory: Trajectory) -> bool:
        """判断是否失败"""
        if len(trajectory.errors) > 0:
            return True

        if trajectory.score is not None and trajectory.score < self.failure_threshold:
            return True

        return False

    def _extract_error_patterns(self, trajectory: Trajectory) -> List[str]:
        """提取错误模式"""
        patterns = []

        for error in trajectory.errors:
            patterns.append(error["error_type"])

        # 也可以从推理步骤中提取模式
        # ...

        return patterns

    def _summarize_reasoning(self, trajectory: Trajectory) -> str:
        """总结推理过程"""
        if not trajectory.reasoning_steps:
            return "No reasoning steps recorded"

        steps = [step["step"] for step in trajectory.reasoning_steps]
        return " -> ".join(steps)

    def _generate_feedback(
        self,
        trajectory: Trajectory,
        is_failure: bool
    ) -> str:
        """生成反馈信息"""
        if not is_failure:
            return "Execution successful"

        feedback_parts = []

        # 添加错误信息
        if trajectory.errors:
            error_msgs = [f"{e['error_type']}: {e['message']}" for e in trajectory.errors]
            feedback_parts.append("Errors: " + "; ".join(error_msgs))

        # 添加分数信息
        if trajectory.score is not None:
            feedback_parts.append(f"Score: {trajectory.score:.2f} (threshold: {self.failure_threshold})")

        # 添加推理步骤信息（如果有失败）
        if trajectory.reasoning_steps:
            feedback_parts.append(f"Completed {len(trajectory.reasoning_steps)} reasoning steps")

        return " | ".join(feedback_parts) if feedback_parts else "Unknown failure"


# 示例使用
if __name__ == "__main__":
    from .trajectory import Trajectory

    # 创建示例轨迹
    traj1 = Trajectory(trajectory_id="test1")
    traj1.input_data = {"question": "What is 2+2?"}
    traj1.output_data = {"answer": "5"}
    traj1.score = 0.0
    traj1.add_error("MathError", "Incorrect calculation")

    traj2 = Trajectory(trajectory_id="test2")
    traj2.input_data = {"question": "What is 3+3?"}
    traj2.output_data = {"answer": "6"}
    traj2.score = 1.0

    # 分析轨迹
    analyzer = ContextAnalyzer()

    ctx1 = analyzer.analyze_trajectory(traj1)
    ctx2 = analyzer.analyze_trajectory(traj2)

    print("Context 1 (failure):", ctx1.is_failure, ctx1.feedback)
    print("Context 2 (success):", ctx2.is_failure, ctx2.feedback)

    # 计算相似度
    similarity = analyzer.compute_context_similarity(ctx1, ctx2)
    print(f"Similarity: {similarity:.2f}")

    # 识别失败案例
    failures = analyzer.identify_failures([traj1, traj2])
    print(f"Found {len(failures)} failures")
