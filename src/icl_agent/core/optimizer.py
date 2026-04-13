"""
Optimizer Module - 智能体优化器

协调所有组件进行智能体优化：
- 管理优化循环
- 协调学习策略
- 评估性能
- 记录优化过程
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

from .trajectory import Trajectory, TrajectoryBatch, TrajectoryCapture
from .context import ContextAnalyzer, ContextData
from .knowledge import KnowledgeExtractor, Knowledge, KnowledgeBase


@dataclass
class OptimizationResult:
    """
    优化结果

    包含优化过程的所有信息
    """

    # 基本信息
    optimization_id: str
    start_time: datetime
    end_time: datetime
    total_iterations: int

    # 性能指标
    initial_score: float
    final_score: float
    best_score: float
    improvement: float  # 提升百分比

    # 最优配置
    best_agent_config: Dict[str, Any]
    best_instruction: str

    # 优化历史
    score_history: List[float] = field(default_factory=list)
    iteration_logs: List[Dict[str, Any]] = field(default_factory=list)

    # 知识库
    knowledge_base: Optional[KnowledgeBase] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "optimization_id": self.optimization_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_iterations": self.total_iterations,
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "best_score": self.best_score,
            "improvement": self.improvement,
            "best_agent_config": self.best_agent_config,
            "best_instruction": self.best_instruction,
            "score_history": self.score_history,
            "metadata": self.metadata,
        }

    def save(self, filepath: str):
        """保存结果到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class AgentOptimizer:
    """
    智能体优化器

    核心优化引擎，协调所有组件
    """

    def __init__(
        self,
        initial_agent_config: Dict[str, Any],
        learning_strategy: Any,  # LearningStrategy实例
        adapter: Any,  # Adapter实例
        evaluator: Callable[[Any, List[Dict]], float],
        max_iterations: int = 10,
        min_improvement: float = 0.01,
        failure_threshold: float = 0.5,
        verbose: bool = True
    ):
        """
        初始化优化器

        Args:
            initial_agent_config: 初始智能体配置
            learning_strategy: 学习策略
            adapter: 领域适配器
            evaluator: 评估函数
            max_iterations: 最大迭代次数
            min_improvement: 最小改进阈值
            failure_threshold: 失败判定阈值（分数低于此值视为失败，默认0.5）
            verbose: 是否打印详细信息
        """
        self.initial_agent_config = initial_agent_config
        self.learning_strategy = learning_strategy
        self.adapter = adapter
        self.evaluator = evaluator

        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.verbose = verbose

        # 核心组件
        self.context_analyzer = ContextAnalyzer(failure_threshold=failure_threshold)
        self.knowledge_extractor = KnowledgeExtractor()

        # 优化状态
        self.current_agent_config = initial_agent_config.copy()
        self.best_agent_config = initial_agent_config.copy()
        self.best_score = 0.0

        self.score_history = []
        self.iteration_logs = []

    def optimize(
        self,
        trainset: List[Dict[str, Any]],
        valset: Optional[List[Dict[str, Any]]] = None
    ) -> OptimizationResult:
        """
        执行优化

        Args:
            trainset: 训练数据集
            valset: 验证数据集（可选）

        Returns:
            OptimizationResult: 优化结果
        """
        import uuid

        optimization_id = str(uuid.uuid4())
        start_time = datetime.now()

        if self.verbose:
            print(f"开始优化 (ID: {optimization_id})")
            print(f"训练集大小: {len(trainset)}")
            if valset:
                print(f"验证集大小: {len(valset)}")

        # 评估初始性能
        initial_score = self._evaluate_agent(
            self.current_agent_config,
            valset if valset else trainset
        )
        self.best_score = initial_score
        self.score_history.append(initial_score)

        if self.verbose:
            print(f"初始分数: {initial_score:.4f}\n")

        # 优化循环
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"{'='*50}")
                print(f"迭代 {iteration + 1}/{self.max_iterations}")
                print(f"{'='*50}")

            # 执行一次优化迭代
            iteration_result = self._optimize_iteration(
                trainset=trainset,
                valset=valset,
                iteration=iteration
            )

            # 记录日志
            self.iteration_logs.append(iteration_result)

            # 检查是否满足停止条件
            if self._should_stop(iteration_result):
                if self.verbose:
                    print("\n满足停止条件，提前终止优化")
                break

        end_time = datetime.now()

        # 创建优化结果
        result = OptimizationResult(
            optimization_id=optimization_id,
            start_time=start_time,
            end_time=end_time,
            total_iterations=len(self.iteration_logs),
            initial_score=initial_score,
            final_score=self.score_history[-1],
            best_score=self.best_score,
            improvement=((self.best_score - initial_score) / initial_score * 100) if initial_score > 0 else (self.best_score * 100),
            best_agent_config=self.best_agent_config,
            best_instruction=self.best_agent_config.get("system_prompt", ""),
            score_history=self.score_history,
            iteration_logs=self.iteration_logs,
            knowledge_base=self.knowledge_extractor.get_knowledge_base(),
            metadata={
                "max_iterations": self.max_iterations,
                "min_improvement": self.min_improvement,
                "strategy": type(self.learning_strategy).__name__,
                "adapter": type(self.adapter).__name__,
            }
        )

        if self.verbose:
            print(f"\n{'='*50}")
            print("优化完成！")
            print(f"{'='*50}")
            print(f"初始分数: {initial_score:.4f}")
            print(f"最终分数: {result.final_score:.4f}")
            print(f"最佳分数: {result.best_score:.4f}")
            print(f"性能提升: {result.improvement:.2f}%")
            print(f"总迭代次数: {result.total_iterations}")

        return result

    def _optimize_iteration(
        self,
        trainset: List[Dict[str, Any]],
        valset: Optional[List[Dict[str, Any]]],
        iteration: int
    ) -> Dict[str, Any]:
        """
        执行单次优化迭代

        Returns:
            Dict: 迭代结果
        """
        # 1. 在训练集上执行当前智能体并捕获轨迹
        if self.verbose:
            print(f"步骤1: 执行智能体并捕获轨迹...")

        trajectories = self._execute_and_capture(
            self.current_agent_config,
            trainset
        )

        # 2. 分析轨迹，提取上下文
        if self.verbose:
            print(f"步骤2: 分析轨迹...")

        contexts = self.context_analyzer.analyze_batch(trajectories)
        failed_contexts = [ctx for ctx in contexts if ctx.is_failure]

        if self.verbose:
            print(f"  - 总轨迹数: {len(trajectories)}")
            print(f"  - 失败案例: {len(failed_contexts)}")

        # 3. 应用学习策略，生成改进
        if self.verbose:
            print(f"步骤3: 应用学习策略...")

        improved_config = self.learning_strategy.learn(
            current_config=self.current_agent_config,
            contexts=contexts,
            failed_contexts=failed_contexts,
            knowledge_extractor=self.knowledge_extractor
        )

        # 4. 评估改进后的智能体
        if self.verbose:
            print(f"步骤4: 评估改进...")

        eval_set = valset if valset else trainset
        new_score = self._evaluate_agent(improved_config, eval_set)
        old_score = self.score_history[-1] if self.score_history else 0.0

        improvement = new_score - old_score

        if self.verbose:
            print(f"  - 旧分数: {old_score:.4f}")
            print(f"  - 新分数: {new_score:.4f}")
            print(f"  - 改进: {improvement:+.4f}")

        # 5. 更新最佳配置
        if new_score > self.best_score:
            self.best_score = new_score
            self.best_agent_config = improved_config.copy()
            if self.verbose:
                print(f"  [BEST] 发现新的最佳配置!")

        # 6. 更新当前配置（可选：只在改进时更新）
        if new_score >= old_score:
            self.current_agent_config = improved_config
        else:
            if self.verbose:
                print(f"  [SKIP] 性能下降, 保持原配置")

        self.score_history.append(new_score)

        return {
            "iteration": iteration,
            "old_score": old_score,
            "new_score": new_score,
            "improvement": improvement,
            "num_trajectories": len(trajectories),
            "num_failures": len(failed_contexts),
            "is_best": new_score == self.best_score,
        }

    def _execute_and_capture(
        self,
        agent_config: Dict[str, Any],
        dataset: List[Dict[str, Any]]
    ) -> List[Trajectory]:
        """
        执行智能体并捕获轨迹

        Args:
            agent_config: 智能体配置
            dataset: 数据集

        Returns:
            List[Trajectory]: 轨迹列表
        """
        trajectories = []

        for data in dataset:
            # 使用适配器执行智能体
            with TrajectoryCapture() as tc:
                tc.log_input(data)

                try:
                    # 调用适配器执行
                    output = self.adapter.execute(agent_config, data)
                    tc.log_output(output)

                    # 评估输出
                    score = self.evaluator(output, data)
                    tc.set_score(score)

                except Exception as e:
                    tc.log_error(type(e).__name__, str(e))
                    tc.set_score(0.0)

                trajectories.append(tc.get_trajectory())

        return trajectories

    def _evaluate_agent(
        self,
        agent_config: Dict[str, Any],
        dataset: List[Dict[str, Any]]
    ) -> float:
        """
        评估智能体性能

        Args:
            agent_config: 智能体配置
            dataset: 数据集

        Returns:
            float: 平均分数
        """
        trajectories = self._execute_and_capture(agent_config, dataset)
        scores = [t.score for t in trajectories if t.score is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def _should_stop(self, iteration_result: Dict[str, Any]) -> bool:
        """
        判断是否应该停止优化

        Args:
            iteration_result: 迭代结果

        Returns:
            bool: 是否停止
        """
        # 条件1: 达到完美分数
        if self.best_score >= 0.99:
            return True

        # 条件2: 连续多次无改进（至少2次迭代后才检查）
        if len(self.score_history) >= 3:
            recent = self.score_history[-2:]
            if all(abs(s - self.best_score) < self.min_improvement for s in recent):
                return True

        return False


# 示例使用
if __name__ == "__main__":
    # 这只是框架示例，实际使用需要提供具体的策略和适配器

    print("AgentOptimizer 已加载")
    print("使用示例请参考 examples/ 目录")
