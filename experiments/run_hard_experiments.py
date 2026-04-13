"""
ICL-Agent 高难度实验脚本

目的: 上一轮 GSM8K 数学题初始满分，无法区分策略。
本轮使用竞赛级数学 / 高难度逻辑 / 算法级代码 Bug 数据集。

实验矩阵:
  G: MATH-Hard x {Reflective, FewShot-CoT, Retrieval-Active}   -- 数学策略对比
  H: Logic-Hard x {Reflective, FewShot-CoT, Retrieval-Active}  -- 逻辑策略对比
  I: Code-Hard x {Reflective+LLM-Judge, Reflective+Keyword}    -- 代码评估对比

运行:
  cd icl-agent
  python experiments/run_hard_experiments.py
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

from datasets_hard import (
    MATH_HARD_TRAINSET, MATH_HARD_VALSET,
    LOGIC_HARD_TRAINSET, LOGIC_HARD_VALSET,
    CODE_HARD_TRAINSET, CODE_HARD_VALSET,
)


# ─── 评估函数 ───

def strict_evaluator(output, data):
    """严格精确匹配"""
    predicted = output.get("answer", "").strip().lower()
    expected = data.get("answer", "").strip().lower()
    return 1.0 if predicted == expected else 0.0


def math_evaluator_factory(adapter):
    def evaluator(output, data):
        return adapter.evaluate(output, data)
    return evaluator


def code_evaluator_factory(adapter):
    def evaluator(output, data):
        return adapter.evaluate(output, data)
    return evaluator


def run_experiment(name, trainset, valset, strategy, evaluator_fn, adapter,
                   initial_prompt, max_iter=3):
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
    print("  ICL-Agent Hard-Difficulty Experiments")
    print("  MATH-Hard(20+10) + Logic-Hard(20+10) + Code-Hard(10+5)")
    print("=" * 60)

    # LLM 客户端
    task_lm = DeepSeekClient(
        model="deepseek-chat", temperature=0.0, max_tokens=1024
    )
    reflection_lm = DeepSeekClient(
        model="deepseek-chat", temperature=0.7, max_tokens=2048
    )

    all_results = []

    # ─── G: MATH-Hard 竞赛数学 ───
    math_prompt = (
        "You are a math competition solver. Solve the problem step by step "
        "using rigorous mathematical reasoning. "
        "At the end, write your final numerical answer after '#### '. "
        "For example: #### 42"
    )

    math_adapter = MathAdapter(llm_client=task_lm)
    math_eval = math_evaluator_factory(math_adapter)

    # G1: Reflective
    all_results.append(run_experiment(
        name="G1: MATH-Hard (Reflective)",
        trainset=MATH_HARD_TRAINSET,
        valset=MATH_HARD_VALSET,
        strategy=ReflectiveLearningStrategy(
            reflection_lm=reflection_lm, max_failures=10
        ),
        evaluator_fn=math_eval,
        adapter=math_adapter,
        initial_prompt=math_prompt,
        max_iter=3,
    ))

    # G2: FewShot-CoT
    all_results.append(run_experiment(
        name="G2: MATH-Hard (FewShot-CoT)",
        trainset=MATH_HARD_TRAINSET,
        valset=MATH_HARD_VALSET,
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
    ))

    # G3: Retrieval-Active
    all_results.append(run_experiment(
        name="G3: MATH-Hard (Retrieval-Active)",
        trainset=MATH_HARD_TRAINSET,
        valset=MATH_HARD_VALSET,
        strategy=RetrievalLearningStrategy(
            top_k=3,
            auto_extract=True,
            reflection_lm=reflection_lm,
        ),
        evaluator_fn=math_eval,
        adapter=math_adapter,
        initial_prompt=math_prompt,
        max_iter=3,
    ))

    # ─── H: Logic-Hard 高难度逻辑 ───
    logic_prompt = (
        "You are a logic puzzle expert. Think step by step and reason carefully. "
        "Answer the question following the exact format requested."
    )

    logic_adapter = QAAdapter(llm_client=task_lm)

    # H1: Reflective
    all_results.append(run_experiment(
        name="H1: Logic-Hard (Reflective)",
        trainset=LOGIC_HARD_TRAINSET,
        valset=LOGIC_HARD_VALSET,
        strategy=ReflectiveLearningStrategy(
            reflection_lm=reflection_lm, max_failures=10
        ),
        evaluator_fn=strict_evaluator,
        adapter=logic_adapter,
        initial_prompt=logic_prompt,
        max_iter=3,
    ))

    # H2: FewShot-CoT
    all_results.append(run_experiment(
        name="H2: Logic-Hard (FewShot-CoT)",
        trainset=LOGIC_HARD_TRAINSET,
        valset=LOGIC_HARD_VALSET,
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
    ))

    # H3: Retrieval-Active
    all_results.append(run_experiment(
        name="H3: Logic-Hard (Retrieval-Active)",
        trainset=LOGIC_HARD_TRAINSET,
        valset=LOGIC_HARD_VALSET,
        strategy=RetrievalLearningStrategy(
            top_k=3,
            auto_extract=True,
            reflection_lm=reflection_lm,
        ),
        evaluator_fn=strict_evaluator,
        adapter=logic_adapter,
        initial_prompt=logic_prompt,
        max_iter=3,
    ))

    # ─── I: Code-Hard 算法级代码 Bug ───
    code_prompt = (
        "You are an expert software engineer specializing in debugging. "
        "Identify the bug precisely and explain the fix concisely."
    )

    # I1: Reflective + LLM-Judge
    code_adapter_judge = CodeAdapter(
        llm_client=task_lm, judge_lm=reflection_lm, task_type="code_fix"
    )
    all_results.append(run_experiment(
        name="I1: Code-Hard (Reflective+LLM-Judge)",
        trainset=CODE_HARD_TRAINSET,
        valset=CODE_HARD_VALSET,
        strategy=ReflectiveLearningStrategy(
            reflection_lm=reflection_lm, max_failures=5
        ),
        evaluator_fn=code_evaluator_factory(code_adapter_judge),
        adapter=code_adapter_judge,
        initial_prompt=code_prompt,
        max_iter=3,
    ))

    # I2: Reflective + Keyword
    code_adapter_kw = CodeAdapter(
        llm_client=task_lm, judge_lm=None, task_type="code_fix"
    )
    all_results.append(run_experiment(
        name="I2: Code-Hard (Reflective+Keyword)",
        trainset=CODE_HARD_TRAINSET,
        valset=CODE_HARD_VALSET,
        strategy=ReflectiveLearningStrategy(
            reflection_lm=reflection_lm, max_failures=5
        ),
        evaluator_fn=code_evaluator_factory(code_adapter_kw),
        adapter=code_adapter_kw,
        initial_prompt=code_prompt,
        max_iter=3,
    ))

    # ─── Final Report ───
    print("\n" + "=" * 80)
    print("  FINAL RESULTS - HARD-DIFFICULTY EXPERIMENTS")
    print("=" * 80)
    header = (
        f"{'Experiment':<42} {'Adapter':<14} "
        f"{'Init':>6} {'Best':>6} {'Improv':>8} {'Iter':>5} {'Time':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['experiment']:<42} {r['adapter']:<14} "
            f"{r['initial_score']:>6.2f} {r['best_score']:>6.2f} "
            f"{r['improvement_pct']:>7.1f}% "
            f"{r['iterations']:>4} {r['elapsed_seconds']:>5.0f}s"
        )

    # 分组对比
    print("\n--- Strategy Comparison (per task) ---")

    groups = {}
    for r in all_results:
        # 提取任务名称: "G1: MATH-Hard" -> "MATH-Hard"
        task = r['experiment'].split(": ", 1)[1].split(" (")[0]
        groups.setdefault(task, []).append(r)

    for task, results in groups.items():
        print(f"\n  {task}:")
        for r in results:
            strat = r['experiment'].split("(")[1].rstrip(")")
            print(
                f"    {strat:<30} "
                f"{r['initial_score']:.2f} -> {r['best_score']:.2f} "
                f"({r['improvement_pct']:+.1f}%)"
            )

    # Save
    out_path = os.path.join(
        os.path.dirname(__file__), "hard_experiment_results.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
