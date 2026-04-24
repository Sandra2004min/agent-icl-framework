"""
ICL-Agent 验证集扩充(20题)重跑实验脚本

目的: 验证集从10题扩充到20题后,重跑6组核心Logic实验(DeepSeek)
  E1-E3: Logic-Standard x {Reflective, FewShot-CoT, Retrieval-v3}
  H1-H3: Logic-Hard     x {Reflective, FewShot-CoT, Retrieval-v3}

运行:
  cd icl-agent
  python experiments/run_valset20_experiments.py
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

from datasets import LOGIC_TRAINSET_EXTENDED, LOGIC_VALSET_EXTENDED
from datasets_hard import LOGIC_HARD_TRAINSET, LOGIC_HARD_VALSET


# --- 评估函数 ---

def strict_evaluator(output, data):
    """精确匹配(与之前实验一致)"""
    predicted = output.get("answer", "").strip().lower()
    expected = data.get("answer", "").strip().lower()
    return 1.0 if predicted == expected else 0.0


def run_experiment(name, trainset, valset, strategy, adapter, initial_prompt, max_iter=3):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Strategy:   {type(strategy).__name__}")
    print(f"  Train: {len(trainset)}, Val: {len(valset)}")
    print(f"{'='*60}\n")

    initial_config = {"system_prompt": initial_prompt}

    optimizer = AgentOptimizer(
        initial_agent_config=initial_config,
        learning_strategy=strategy,
        adapter=adapter,
        evaluator=strict_evaluator,
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
        "train_size": len(trainset),
        "val_size": len(valset),
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
    print(f"  Initial:     {summary['initial_score']}")
    print(f"  Best:        {summary['best_score']}")
    print(f"  Final:       {summary['final_score']}")
    print(f"  Improvement: {summary['improvement_pct']}%")
    print(f"  Iterations:  {summary['iterations']}")
    print(f"  Time:        {summary['elapsed_seconds']}s")
    print(f"  History:     {summary['score_history']}")
    print()

    return summary


def main():
    print("=" * 60)
    print("  ICL-Agent ValSet-20 Experiments")
    print(f"  Logic-Std:  train={len(LOGIC_TRAINSET_EXTENDED)}, val={len(LOGIC_VALSET_EXTENDED)}")
    print(f"  Logic-Hard: train={len(LOGIC_HARD_TRAINSET)}, val={len(LOGIC_HARD_VALSET)}")
    print("  Model: DeepSeek-chat")
    print("  Strategies: Reflective, FewShot-CoT, Retrieval-v3")
    print("=" * 60)

    # LLM 客户端
    task_lm = DeepSeekClient(model="deepseek-chat", temperature=0.0, max_tokens=512)
    reflection_lm = DeepSeekClient(model="deepseek-chat", temperature=0.7, max_tokens=2048)

    all_results = []

    # --- E: Logic-Standard (20 val) ---
    logic_prompt = "You are a helpful assistant. Answer the question."
    logic_adapter = QAAdapter(llm_client=task_lm)

    # E1: Reflective
    all_results.append(run_experiment(
        name="E1: Logic-Std (Reflective) [val=20]",
        trainset=LOGIC_TRAINSET_EXTENDED,
        valset=LOGIC_VALSET_EXTENDED,
        strategy=ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=8),
        adapter=logic_adapter,
        initial_prompt=logic_prompt,
    ))

    # E2: FewShot-CoT
    all_results.append(run_experiment(
        name="E2: Logic-Std (FewShot-CoT) [val=20]",
        trainset=LOGIC_TRAINSET_EXTENDED,
        valset=LOGIC_VALSET_EXTENDED,
        strategy=FewShotLearningStrategy(
            num_shots=5,
            include_reasoning=True,
            include_negative=True,
            max_negative=2,
            reflection_lm=reflection_lm,
        ),
        adapter=logic_adapter,
        initial_prompt=logic_prompt,
    ))

    # E3: Retrieval-v3
    all_results.append(run_experiment(
        name="E3: Logic-Std (Retrieval-v3) [val=20]",
        trainset=LOGIC_TRAINSET_EXTENDED,
        valset=LOGIC_VALSET_EXTENDED,
        strategy=RetrievalLearningStrategy(
            top_k=3,
            auto_extract=True,
            reflection_lm=reflection_lm,
        ),
        adapter=logic_adapter,
        initial_prompt=logic_prompt,
    ))

    # --- H: Logic-Hard (20 val) ---
    hard_prompt = "You are a logic puzzle expert. Think step by step and reason carefully. Answer the question following the exact format requested."
    hard_adapter = QAAdapter(llm_client=task_lm)

    # H1: Reflective
    all_results.append(run_experiment(
        name="H1: Logic-Hard (Reflective) [val=20]",
        trainset=LOGIC_HARD_TRAINSET,
        valset=LOGIC_HARD_VALSET,
        strategy=ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=8),
        adapter=hard_adapter,
        initial_prompt=hard_prompt,
    ))

    # H2: FewShot-CoT
    all_results.append(run_experiment(
        name="H2: Logic-Hard (FewShot-CoT) [val=20]",
        trainset=LOGIC_HARD_TRAINSET,
        valset=LOGIC_HARD_VALSET,
        strategy=FewShotLearningStrategy(
            num_shots=5,
            include_reasoning=True,
            include_negative=True,
            max_negative=2,
            reflection_lm=reflection_lm,
        ),
        adapter=hard_adapter,
        initial_prompt=hard_prompt,
    ))

    # H3: Retrieval-v3
    all_results.append(run_experiment(
        name="H3: Logic-Hard (Retrieval-v3) [val=20]",
        trainset=LOGIC_HARD_TRAINSET,
        valset=LOGIC_HARD_VALSET,
        strategy=RetrievalLearningStrategy(
            top_k=3,
            auto_extract=True,
            reflection_lm=reflection_lm,
        ),
        adapter=hard_adapter,
        initial_prompt=hard_prompt,
    ))

    # --- 汇总报告 ---
    print("\n" + "=" * 80)
    print("  FINAL RESULTS - ValSet-20 Experiments")
    print("=" * 80)
    header = f"{'Experiment':<45} {'Init':>6} {'Best':>6} {'Final':>6} {'Improv':>8} {'Iter':>5} {'Time':>6}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['experiment']:<45} "
            f"{r['initial_score']:>6.3f} {r['best_score']:>6.3f} {r['final_score']:>6.3f} "
            f"{r['improvement_pct']:>7.1f}% {r['iterations']:>4} {r['elapsed_seconds']:>5.0f}s"
        )

    # 与旧数据(val=10)对比
    print("\n--- Comparison with val=10 baselines ---")
    old_baselines = {
        "E1": {"init": 0.7, "best": 1.0, "pct": 42.86},
        "E2": {"init": 0.7, "best": 0.8, "pct": 14.29},
        "E3": {"init": 0.7, "best": 0.9, "pct": 28.57},
        "H1": {"init": 0.6, "best": 0.9, "pct": 50.0},
        "H2": {"init": 0.6, "best": 0.8, "pct": 33.33},
        "H3": {"init": 0.6, "best": 0.6, "pct": 0.0},
    }
    for r in all_results:
        tag = r['experiment'][:2]
        old = old_baselines.get(tag, {})
        if old:
            print(
                f"  {tag}: val=10 -> {old['init']:.1f}/{old['best']:.1f} ({old['pct']:+.1f}%)  |  "
                f"val=20 -> {r['initial_score']:.3f}/{r['best_score']:.3f} ({r['improvement_pct']:+.1f}%)"
            )

    # 保存
    out_path = os.path.join(os.path.dirname(__file__), "valset20_experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
