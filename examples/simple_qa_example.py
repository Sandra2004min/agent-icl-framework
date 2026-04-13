"""
Simple QA Example
简单的问答优化示例

展示如何使用ICL-Agent框架优化一个简单的问答智能体
"""

import sys
sys.path.insert(0, '../src')

from icl_agent.core import AgentOptimizer, OptimizationResult
from icl_agent.strategies import ReflectiveLearningStrategy
from icl_agent.adapters import QAAdapter


def main():
    print("="*60)
    print("ICL-Agent 简单问答示例")
    print("="*60)
    print()

    # 1. 准备数据集
    trainset = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is 3+3?", "answer": "6"},
        {"question": "What is 5+5?", "answer": "10"},
        {"question": "What is 10-3?", "answer": "7"},
        {"question": "What is 8-2?", "answer": "6"},
    ]

    valset = [
        {"question": "What is 4+4?", "answer": "8"},
        {"question": "What is 9-4?", "answer": "5"},
    ]

    print(f"训练集大小: {len(trainset)}")
    print(f"验证集大小: {len(valset)}")
    print()

    # 2. 配置初始智能体
    initial_config = {
        "system_prompt": "You are a helpful assistant. Answer the question."
    }

    print("初始提示词:")
    print(initial_config["system_prompt"])
    print()

    # 3. 创建适配器
    adapter = QAAdapter()

    # 4. 创建评估函数
    def evaluator(output, data):
        """简单的评估函数"""
        return adapter.evaluate(output, data)

    # 5. 创建学习策略
    strategy = ReflectiveLearningStrategy()

    # 6. 创建优化器
    optimizer = AgentOptimizer(
        initial_agent_config=initial_config,
        learning_strategy=strategy,
        adapter=adapter,
        evaluator=evaluator,
        max_iterations=3,  # 简单示例，只运行3次迭代
        verbose=True
    )

    # 7. 运行优化
    print("\n开始优化...\n")
    result = optimizer.optimize(
        trainset=trainset,
        valset=valset
    )

    # 8. 显示结果
    print("\n" + "="*60)
    print("优化结果摘要")
    print("="*60)
    print(f"初始分数: {result.initial_score:.4f}")
    print(f"最终分数: {result.final_score:.4f}")
    print(f"最佳分数: {result.best_score:.4f}")
    print(f"性能提升: {result.improvement:.2f}%")
    print(f"总迭代次数: {result.total_iterations}")
    print()

    print("最优提示词:")
    print("-" * 60)
    print(result.best_instruction)
    print("-" * 60)
    print()

    print("分数历史:")
    for i, score in enumerate(result.score_history):
        print(f"  迭代 {i}: {score:.4f}")
    print()

    # 9. 保存结果
    result_file = "optimization_result.json"
    result.save(result_file)
    print(f"结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
