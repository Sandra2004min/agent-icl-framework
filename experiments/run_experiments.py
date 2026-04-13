"""
ICL-Agent 系统化实验脚本

实验 A: 逻辑推理题 - 反思策略优化
实验 B: 代码 Debug 题 - 反思策略优化
实验 C: 三种策略对比 (反思 vs 少样本 vs 检索)

运行:
  cd icl-agent
  python experiments/run_experiments.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 加载 .env
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(_env_path):
    with open(_env_path, encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ.setdefault(_key.strip(), _val.strip())

from icl_agent.core import AgentOptimizer
from icl_agent.strategies import (
    ReflectiveLearningStrategy,
    FewShotLearningStrategy,
    RetrievalLearningStrategy,
)
from icl_agent.adapters import QAAdapter
from icl_agent.utils.llm_client import DeepSeekClient


# ─── 数据集定义 ───

LOGIC_TRAINSET = [
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

LOGIC_VALSET = [
    {"question": "A clock strikes 6 in 5 seconds. How many seconds will it take to strike 12? Answer with ONLY the number.", "answer": "11"},
    {"question": "How many 9s are there between 1 and 100? Answer with ONLY the number.", "answer": "20"},
    {"question": "If you rearrange the letters 'CIFAIPC' you get the name of a(n)? Answer with ONLY the word in lowercase.", "answer": "pacific"},
    {"question": "Two fathers and two sons go fishing. They each catch one fish. They catch 3 fish total. How is this possible? Answer: because there are actually ___ people (grandfather, father, son). Fill in the number ONLY.", "answer": "3"},
    {"question": "What is half of 2+2? Answer with ONLY the number.", "answer": "3"},
    {"question": "If you have 6 oranges and give half to a friend, how many do you have? Answer with ONLY the number.", "answer": "3"},
]

CODE_TRAINSET = [
    {"question": "What is the bug in this Python code?\ndef factorial(n):\n    result = 0\n    for i in range(1, n+1):\n        result *= i\n    return result\nAnswer with ONLY the fix in one sentence.", "answer": "result should be initialized to 1 instead of 0"},
    {"question": "What is the bug in this Python code?\ndef is_palindrome(s):\n    return s == s.reverse()\nAnswer with ONLY the fix in one sentence.", "answer": "strings don't have a reverse method, use s[::-1] instead"},
    {"question": "What is the bug in this Python code?\ndef average(numbers):\n    return sum(numbers) / len(numbers)\naverage([])\nAnswer with ONLY the fix in one sentence.", "answer": "add a check for empty list to avoid division by zero"},
    {"question": "What is the bug in this Python code?\ndef find_max(lst):\n    max_val = 0\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val\nfind_max([-1, -5, -3])\nAnswer with ONLY the fix in one sentence.", "answer": "initialize max_val to float('-inf') or lst[0] instead of 0"},
    {"question": "What is the bug in this Python code?\ndef count_vowels(s):\n    vowels = 'aeiou'\n    count = 0\n    for char in s:\n        if char in vowels:\n            count += 1\n    return count\ncount_vowels('HELLO')\nAnswer with ONLY the fix in one sentence.", "answer": "convert char to lowercase before checking or add uppercase vowels"},
    {"question": "What is the bug in this Python code?\ndef remove_duplicates(lst):\n    for item in lst:\n        if lst.count(item) > 1:\n            lst.remove(item)\n    return lst\nAnswer with ONLY the fix in one sentence.", "answer": "don't modify a list while iterating over it, use a set or new list instead"},
]

CODE_VALSET = [
    {"question": "What is the bug in this Python code?\ndef binary_search(arr, target):\n    low, high = 0, len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid\n    return -1\nAnswer with ONLY the fix in one sentence.", "answer": "low should be updated to mid + 1 to avoid infinite loop"},
    {"question": "What is the bug in this Python code?\ndef flatten(nested_list):\n    result = []\n    for item in nested_list:\n        if type(item) == list:\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\nAnswer with ONLY the fix in one sentence.", "answer": "use isinstance(item, list) instead of type(item) == list to handle subclasses"},
    {"question": "What is the bug in this Python code?\ndef fib(n):\n    if n == 0: return 0\n    if n == 1: return 1\n    return fib(n-1) + fib(n-2)\nfib(50)\nAnswer with ONLY the fix in one sentence.", "answer": "add memoization or use iterative approach to avoid exponential time complexity"},
]


def strict_evaluator(output, data):
    """严格精确匹配评估"""
    predicted = output.get("answer", "").strip().lower()
    expected = data.get("answer", "").strip().lower()
    return 1.0 if predicted == expected else 0.0


def semantic_evaluator(output, data):
    """语义包含评估 -- 预测中包含答案关键词即得分"""
    predicted = output.get("answer", "").strip().lower()
    expected = data.get("answer", "").strip().lower()
    if predicted == expected:
        return 1.0
    # 关键词匹配: 答案中的主要词都出现在预测中
    expected_words = set(expected.split())
    predicted_words = set(predicted.split())
    if len(expected_words) == 0:
        return 0.0
    overlap = len(expected_words & predicted_words) / len(expected_words)
    return round(overlap, 2)


def run_single_experiment(
    name, trainset, valset, strategy, evaluator_fn, task_lm, reflection_lm=None
):
    """运行单个实验并返回结果"""
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"{'='*60}\n")

    adapter = QAAdapter(llm_client=task_lm)
    initial_config = {
        "system_prompt": "You are a helpful assistant. Answer the question."
    }

    optimizer = AgentOptimizer(
        initial_agent_config=initial_config,
        learning_strategy=strategy,
        adapter=adapter,
        evaluator=evaluator_fn,
        max_iterations=3,
        min_improvement=0.001,
        failure_threshold=1.0,
        verbose=True,
    )

    start = time.time()
    result = optimizer.optimize(trainset=trainset, valset=valset)
    elapsed = time.time() - start

    summary = {
        "experiment": name,
        "strategy": type(strategy).__name__,
        "initial_score": round(result.initial_score, 4),
        "final_score": round(result.final_score, 4),
        "best_score": round(result.best_score, 4),
        "improvement_pct": round(result.improvement, 2),
        "iterations": result.total_iterations,
        "elapsed_seconds": round(elapsed, 1),
        "best_instruction": result.best_instruction,
        "score_history": [round(s, 4) for s in result.score_history],
    }

    print(f"\n--- {name} Summary ---")
    print(f"  Strategy:    {summary['strategy']}")
    print(f"  Initial:     {summary['initial_score']}")
    print(f"  Best:        {summary['best_score']}")
    print(f"  Improvement: {summary['improvement_pct']}%")
    print(f"  Iterations:  {summary['iterations']}")
    print(f"  Time:        {summary['elapsed_seconds']}s")
    print()

    return summary


def main():
    print("=" * 60)
    print("  ICL-Agent Systematic Experiments")
    print("=" * 60)

    task_lm = DeepSeekClient(model="deepseek-chat", temperature=0.0, max_tokens=256)
    reflection_lm = DeepSeekClient(model="deepseek-chat", temperature=0.7, max_tokens=2048)

    all_results = []

    # ── Experiment A: Logic Puzzles + Reflective ──
    exp_a = run_single_experiment(
        name="A: Logic Puzzles (Reflective)",
        trainset=LOGIC_TRAINSET,
        valset=LOGIC_VALSET,
        strategy=ReflectiveLearningStrategy(
            reflection_lm=reflection_lm, max_failures=5
        ),
        evaluator_fn=strict_evaluator,
        task_lm=task_lm,
    )
    all_results.append(exp_a)

    # ── Experiment B: Code Debug + Reflective ──
    exp_b = run_single_experiment(
        name="B: Code Debug (Reflective)",
        trainset=CODE_TRAINSET,
        valset=CODE_VALSET,
        strategy=ReflectiveLearningStrategy(
            reflection_lm=reflection_lm, max_failures=5
        ),
        evaluator_fn=semantic_evaluator,
        task_lm=task_lm,
    )
    all_results.append(exp_b)

    # ── Experiment C: Strategy Comparison on Logic Puzzles ──
    # C1: Few-Shot
    exp_c1 = run_single_experiment(
        name="C1: Logic Puzzles (Few-Shot)",
        trainset=LOGIC_TRAINSET,
        valset=LOGIC_VALSET,
        strategy=FewShotLearningStrategy(num_shots=3),
        evaluator_fn=strict_evaluator,
        task_lm=task_lm,
    )
    all_results.append(exp_c1)

    # C2: Retrieval
    exp_c2 = run_single_experiment(
        name="C2: Logic Puzzles (Retrieval)",
        trainset=LOGIC_TRAINSET,
        valset=LOGIC_VALSET,
        strategy=RetrievalLearningStrategy(top_k=3),
        evaluator_fn=strict_evaluator,
        task_lm=task_lm,
    )
    all_results.append(exp_c2)

    # ── Final Report ──
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"{'Experiment':<35} {'Init':>6} {'Best':>6} {'Improv':>8} {'Iters':>6} {'Time':>6}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['experiment']:<35} {r['initial_score']:>6.2f} {r['best_score']:>6.2f} "
            f"{r['improvement_pct']:>7.1f}% {r['iterations']:>5} {r['elapsed_seconds']:>5.0f}s"
        )

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
