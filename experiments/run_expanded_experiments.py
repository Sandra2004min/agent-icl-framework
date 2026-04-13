"""
ICL-Agent 扩大规模实验脚本

实验矩阵:
  D: GSM8K 数学推理 x {Reflective, FewShot-CoT, Retrieval-Active}
  E: 扩展逻辑推理 x {Reflective, FewShot-CoT, Retrieval-Active}
  F: 代码修复 x {Reflective, CodeAdapter+LLM-Judge}

运行:
  cd icl-agent
  python experiments/run_expanded_experiments.py
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
from icl_agent.adapters import QAAdapter, MathAdapter, CodeAdapter
from icl_agent.utils.llm_client import DeepSeekClient

from datasets import (
    GSM8K_TRAINSET, GSM8K_VALSET,
    LOGIC_TRAINSET_EXTENDED, LOGIC_VALSET_EXTENDED,
    CODE_FIX_TRAINSET, CODE_FIX_VALSET,
)


# ─── 评估函数 ───

def strict_evaluator(output, data):
    """严格精确匹配"""
    predicted = output.get("answer", "").strip().lower()
    expected = data.get("answer", "").strip().lower()
    return 1.0 if predicted == expected else 0.0


def math_evaluator_factory(adapter):
    """创建数学评估器（利用 MathAdapter 的数字提取能力）"""
    def evaluator(output, data):
        return adapter.evaluate(output, data)
    return evaluator


def code_evaluator_factory(adapter):
    """创建代码评估器（利用 CodeAdapter 的多级匹配）"""
    def evaluator(output, data):
        return adapter.evaluate(output, data)
    return evaluator


def run_experiment(name, trainset, valset, strategy, evaluator_fn, adapter, initial_prompt, max_iter=3):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Strategy: {type(strategy).__name__}")
    print(f"  Train: {len(trainset)}, Val: {len(valset)}")
    print(f"{'='*60}\n")

    initial_config = {"system_prompt": initial_prompt}

    optimizer = AgentOptimizer(
        initial_agent_config=initial_config,
        learning_strategy=strategy,
        adapter=adapter,
        evaluator=evaluator_fn,
        max_iterations=max_iter,
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
        "adapter": type(adapter).__name__,
        "train_size": len(trainset),
        "val_size": len(valset),
        "initial_score": round(result.initial_score, 4),
        "final_score": round(result.final_score, 4),
        "best_score": round(result.best_score, 4),
        "improvement_pct": round(result.improvement, 2),
        "iterations": result.total_iterations,
        "elapsed_seconds": round(elapsed, 1),
        "best_instruction": result.best_instruction[:500],
        "score_history": [round(s, 4) for s in result.score_history],
    }

    print(f"\n--- {name} Summary ---")
    print(f"  Strategy:    {summary['strategy']}")
    print(f"  Adapter:     {summary['adapter']}")
    print(f"  Initial:     {summary['initial_score']}")
    print(f"  Best:        {summary['best_score']}")
    print(f"  Improvement: {summary['improvement_pct']}%")
    print(f"  Iterations:  {summary['iterations']}")
    print(f"  Time:        {summary['elapsed_seconds']}s")
    print()

    return summary


def main():
    print("=" * 60)
    print("  ICL-Agent Expanded Experiments")
    print("  Scale: GSM8K(20+10) + Logic(20+10) + Code(10+5)")
    print("=" * 60)

    # LLM 客户端
    task_lm = DeepSeekClient(model="deepseek-chat", temperature=0.0, max_tokens=512)
    reflection_lm = DeepSeekClient(model="deepseek-chat", temperature=0.7, max_tokens=2048)

    all_results = []

    # ─── D: GSM8K 数学推理 ───
    math_prompt = (
        "You are a math problem solver. Solve the problem step by step. "
        "At the end, write your final numerical answer after '#### '. "
        "For example: #### 42"
    )

    math_adapter = MathAdapter(llm_client=task_lm)
    math_eval = math_evaluator_factory(math_adapter)

    # D1: Reflective
    exp_d1 = run_experiment(
        name="D1: GSM8K (Reflective)",
        trainset=GSM8K_TRAINSET,
        valset=GSM8K_VALSET,
        strategy=ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=8),
        evaluator_fn=math_eval,
        adapter=math_adapter,
        initial_prompt=math_prompt,
        max_iter=3,
    )
    all_results.append(exp_d1)

    # D2: FewShot-CoT (改进版)
    exp_d2 = run_experiment(
        name="D2: GSM8K (FewShot-CoT)",
        trainset=GSM8K_TRAINSET,
        valset=GSM8K_VALSET,
        strategy=FewShotLearningStrategy(
            num_shots=3,
            include_reasoning=True,
            include_negative=True,
            max_negative=2,
            reflection_lm=reflection_lm,
        ),
        evaluator_fn=math_eval,
        adapter=math_adapter,
        initial_prompt=math_prompt,
        max_iter=3,
    )
    all_results.append(exp_d2)

    # D3: Retrieval-Active (改进版)
    exp_d3 = run_experiment(
        name="D3: GSM8K (Retrieval-Active)",
        trainset=GSM8K_TRAINSET,
        valset=GSM8K_VALSET,
        strategy=RetrievalLearningStrategy(
            top_k=3,
            auto_extract=True,
            reflection_lm=reflection_lm,
        ),
        evaluator_fn=math_eval,
        adapter=math_adapter,
        initial_prompt=math_prompt,
        max_iter=3,
    )
    all_results.append(exp_d3)

    # ─── E: 扩展逻辑推理 ───
    logic_prompt = "You are a helpful assistant. Answer the question."

    logic_adapter = QAAdapter(llm_client=task_lm)

    # E1: Reflective
    exp_e1 = run_experiment(
        name="E1: Logic-Extended (Reflective)",
        trainset=LOGIC_TRAINSET_EXTENDED,
        valset=LOGIC_VALSET_EXTENDED,
        strategy=ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=8),
        evaluator_fn=strict_evaluator,
        adapter=logic_adapter,
        initial_prompt=logic_prompt,
        max_iter=3,
    )
    all_results.append(exp_e1)

    # E2: FewShot-CoT (改进版)
    exp_e2 = run_experiment(
        name="E2: Logic-Extended (FewShot-CoT)",
        trainset=LOGIC_TRAINSET_EXTENDED,
        valset=LOGIC_VALSET_EXTENDED,
        strategy=FewShotLearningStrategy(
            num_shots=5,
            include_reasoning=True,
            include_negative=True,
            max_negative=2,
            reflection_lm=reflection_lm,
        ),
        evaluator_fn=strict_evaluator,
        adapter=logic_adapter,
        initial_prompt=logic_prompt,
        max_iter=3,
    )
    all_results.append(exp_e2)

    # E3: Retrieval-Active (改进版)
    exp_e3 = run_experiment(
        name="E3: Logic-Extended (Retrieval-Active)",
        trainset=LOGIC_TRAINSET_EXTENDED,
        valset=LOGIC_VALSET_EXTENDED,
        strategy=RetrievalLearningStrategy(
            top_k=3,
            auto_extract=True,
            reflection_lm=reflection_lm,
        ),
        evaluator_fn=strict_evaluator,
        adapter=logic_adapter,
        initial_prompt=logic_prompt,
        max_iter=3,
    )
    all_results.append(exp_e3)

    # ─── F: 代码修复 ───
    code_prompt = (
        "You are a Python code reviewer. "
        "When asked about a bug, identify the issue and explain the fix concisely."
    )

    # F1: Reflective + CodeAdapter + LLM-Judge
    code_adapter_with_judge = CodeAdapter(
        llm_client=task_lm,
        judge_lm=reflection_lm,
        task_type="code_fix",
    )
    code_eval_judge = code_evaluator_factory(code_adapter_with_judge)

    exp_f1 = run_experiment(
        name="F1: Code-Fix (Reflective+LLM-Judge)",
        trainset=CODE_FIX_TRAINSET,
        valset=CODE_FIX_VALSET,
        strategy=ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=5),
        evaluator_fn=code_eval_judge,
        adapter=code_adapter_with_judge,
        initial_prompt=code_prompt,
        max_iter=3,
    )
    all_results.append(exp_f1)

    # F2: CodeAdapter without LLM-Judge (keyword match only, as baseline)
    code_adapter_keyword = CodeAdapter(
        llm_client=task_lm,
        judge_lm=None,
        task_type="code_fix",
    )
    code_eval_keyword = code_evaluator_factory(code_adapter_keyword)

    exp_f2 = run_experiment(
        name="F2: Code-Fix (Reflective+Keyword)",
        trainset=CODE_FIX_TRAINSET,
        valset=CODE_FIX_VALSET,
        strategy=ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=5),
        evaluator_fn=code_eval_keyword,
        adapter=code_adapter_keyword,
        initial_prompt=code_prompt,
        max_iter=3,
    )
    all_results.append(exp_f2)

    # ─── Final Report ───
    print("\n" + "=" * 80)
    print("  FINAL RESULTS - EXPANDED EXPERIMENTS")
    print("=" * 80)
    header = f"{'Experiment':<40} {'Adapter':<14} {'Init':>6} {'Best':>6} {'Improv':>8} {'Iter':>5} {'Time':>6}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['experiment']:<40} {r['adapter']:<14} "
            f"{r['initial_score']:>6.2f} {r['best_score']:>6.2f} "
            f"{r['improvement_pct']:>7.1f}% {r['iterations']:>4} {r['elapsed_seconds']:>5.0f}s"
        )

    # 分组对比
    print("\n--- Strategy Comparison (per task) ---")
    tasks = {}
    for r in all_results:
        task = r['experiment'].split(":")[0]
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(r)

    for task, results in tasks.items():
        print(f"\n  Task Group {task}:")
        for r in results:
            print(f"    {r['strategy']:<30} {r['initial_score']:.2f} -> {r['best_score']:.2f} ({r['improvement_pct']:+.1f}%)")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "expanded_experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
