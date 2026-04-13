"""
DeepSeek QA Example
使用 DeepSeek API 的真实问答优化示例

运行方式:
  cd icl-agent
  pip install openai
  python examples/deepseek_qa_example.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 从 .env 文件加载环境变量（不依赖 python-dotenv）
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(_env_path):
    with open(_env_path, encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ.setdefault(_key.strip(), _val.strip())

from icl_agent.core import AgentOptimizer, OptimizationResult
from icl_agent.strategies import ReflectiveLearningStrategy
from icl_agent.adapters import QAAdapter
from icl_agent.utils.llm_client import DeepSeekClient


def main():
    print("=" * 60)
    print("ICL-Agent + DeepSeek 真实问答优化示例")
    print("=" * 60)
    print()

    # ── 1. 创建 DeepSeek 客户端 ──
    # 任务模型: deepseek-chat (用于回答问题), temperature=0 确保输出稳定
    task_lm = DeepSeekClient(model="deepseek-chat", temperature=0.0, max_tokens=256)

    # 反思模型: 同样使用 deepseek-chat (也可换用 deepseek-reasoner 做更强反思)
    reflection_lm = DeepSeekClient(model="deepseek-chat", temperature=0.7, max_tokens=2048)

    print("DeepSeek API 连接成功")
    print()

    # ── 2. 准备数据集 ──
    # 逻辑推理 + 严格格式要求的任务
    # 答案必须严格匹配（精确字符串），所以格式不对就是 0 分
    # 这给反思优化留出空间：LLM 需要学会精确格式
    trainset = [
        {"question": "There are 5 people in a room. Each shakes hands with every other person exactly once. How many handshakes occur? Answer with ONLY the number.", "answer": "10"},
        {"question": "If today is Wednesday, what day will it be 100 days from now? Answer with ONLY the day name in lowercase.", "answer": "friday"},
        {"question": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Answer with ONLY the number.", "answer": "9"},
        {"question": "How many months have 28 days? Answer with ONLY the number.", "answer": "12"},
        {"question": "If you have a bowl with six apples and you take away four, how many do you have? Answer with ONLY the number.", "answer": "4"},
        {"question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Answer with ONLY the number.", "answer": "5"},
        {"question": "If there are 3 apples and you take away 2, how many apples do YOU have? Answer with ONLY the number.", "answer": "2"},
        {"question": "What is the next number in the sequence: 1, 1, 2, 3, 5, 8, 13, ...? Answer with ONLY the number.", "answer": "21"},
        {"question": "How many times can you subtract 5 from 25? Answer with ONLY the number.", "answer": "1"},
        {"question": "If a doctor gives you 3 pills and tells you to take one every 30 minutes, how many minutes will it take to finish all pills? Answer with ONLY the number.", "answer": "60"},
    ]

    valset = [
        {"question": "A clock strikes 6 in 5 seconds. How many seconds will it take to strike 12? Answer with ONLY the number.", "answer": "11"},
        {"question": "How many 9s are there between 1 and 100? Answer with ONLY the number.", "answer": "20"},
        {"question": "If you rearrange the letters 'CIFAIPC' you get the name of a(n)? Answer with ONLY the word in lowercase.", "answer": "pacific"},
        {"question": "Two fathers and two sons go fishing. They each catch one fish. They catch 3 fish total. How is this possible? Answer: because there are actually ___ people (grandfather, father, son). Fill in the number ONLY.", "answer": "3"},
        {"question": "What is half of 2+2? Answer with ONLY the number.", "answer": "3"},
        {"question": "If you have 6 oranges and give half to a friend, how many do you have? Answer with ONLY the number.", "answer": "3"},
    ]

    print(f"训练集大小: {len(trainset)}")
    print(f"验证集大小: {len(valset)}")
    print()

    # ── 3. 配置初始智能体 ──
    # 故意用一个较弱的提示词，让优化过程有提升空间
    initial_config = {
        "system_prompt": "You are a helpful assistant. Answer the question."
    }
    # 注意: 这个提示词没有要求模型只返回数字，LLM 可能返回冗长解释

    print("初始提示词:")
    print(f"  {initial_config['system_prompt']}")
    print()

    # ── 4. 创建适配器 (使用真实 LLM) ──
    adapter = QAAdapter(llm_client=task_lm)

    # ── 5. 创建评估函数 ──
    # 使用严格精确匹配: LLM 输出必须与答案完全一致才得分
    def evaluator(output, data):
        predicted = output.get("answer", "").strip().lower()
        expected = data.get("answer", "").strip().lower()
        return 1.0 if predicted == expected else 0.0

    # ── 6. 创建反思学习策略 (使用真实 LLM 做反思) ──
    strategy = ReflectiveLearningStrategy(
        reflection_lm=reflection_lm,
        max_failures=5,
    )

    # ── 7. 创建优化器 ──
    optimizer = AgentOptimizer(
        initial_agent_config=initial_config,
        learning_strategy=strategy,
        adapter=adapter,
        evaluator=evaluator,
        max_iterations=3,
        min_improvement=0.001,  # 降低停止阈值，允许更多迭代
        failure_threshold=1.0,  # 非满分都视为需要改进
        verbose=True,
    )

    # ── 8. 运行优化 ──
    print("开始优化...\n")
    result = optimizer.optimize(trainset=trainset, valset=valset)

    # ── 9. 显示结果 ──
    print("\n" + "=" * 60)
    print("优化结果摘要")
    print("=" * 60)
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

    # ── 10. 保存结果 ──
    result_file = os.path.join(os.path.dirname(__file__), "deepseek_result.json")
    result.save(result_file)
    print(f"结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
