"""
ICL-Agent 标准基准实验脚本 (MMLU + BBH)

基准数据集:
  1. MMLU (多学科知识推理, 50 train + 25 val)
     学科: logical_fallacies, abstract_algebra, high_school_physics, world_religions, elementary_mathematics
  2. BBH (Big-Bench Hard 推理任务, 50 train + 25 val)
     任务: boolean_expressions, causal_judgement, date_understanding, logical_deduction, navigate

策略: Reflective, FewShot-CoT, Retrieval-v3
模型: DeepSeek-chat

运行:
  cd icl-agent
  python experiments/run_benchmark_mmlu_bbh.py
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


# ─── 评估函数 ───

def mmlu_evaluator(output, data):
    """MMLU: 精确匹配选项字母 (A/B/C/D)"""
    predicted = output.get("answer", "").strip().upper()
    expected = data.get("answer", "").strip().upper()
    # 从回答中提取第一个出现的 A/B/C/D
    import re
    match = re.search(r'\b([ABCD])\b', predicted)
    if match:
        predicted = match.group(1)
    return 1.0 if predicted == expected else 0.0


def bbh_evaluator(output, data):
    """BBH: 精确匹配 (多种格式归一化)"""
    import re
    predicted = output.get("answer", "").strip()
    expected = data.get("answer", "").strip()

    pred_low = predicted.lower()
    exp_low = expected.lower()

    # 直接匹配
    if pred_low == exp_low:
        return 1.0

    # 选项格式: (A)/(B)/... 匹配
    pred_opt = re.search(r'\(([a-zA-Z])\)', predicted)
    exp_opt = re.search(r'\(([a-zA-Z])\)', expected)
    if pred_opt and exp_opt:
        return 1.0 if pred_opt.group(1).lower() == exp_opt.group(1).lower() else 0.0

    # True/False, Yes/No 归一化
    tf_map = {'true': 'true', 'false': 'false', 'yes': 'yes', 'no': 'no',
              'valid': 'valid', 'invalid': 'invalid'}
    for word, norm in tf_map.items():
        if pred_low.startswith(word) and exp_low == norm:
            return 1.0

    # 期望是选项格式但模型输出了实际内容 - 尝试在回答中找选项标签
    if exp_opt:
        # 模型可能输出了 "A" 而非 "(A)"
        bare = re.search(r'^([a-zA-Z])$', predicted.strip())
        if bare and bare.group(1).lower() == exp_opt.group(1).lower():
            return 1.0

    # 包含匹配
    if exp_low in pred_low and len(pred_low) < len(exp_low) * 5:
        return 1.0

    return 0.0


def run_experiment(name, trainset, valset, strategy, adapter, evaluator_fn, initial_prompt, max_iter=3):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Strategy:   {type(strategy).__name__}")
    print(f"  Train: {len(trainset)}, Val: {len(valset)}")
    print(f"{'='*60}\n", flush=True)

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
    print(f"  History:     {summary['score_history']}", flush=True)
    print()

    return summary


def main():
    data_dir = os.path.dirname(__file__)

    # LLM
    task_lm = DeepSeekClient(model="deepseek-chat", temperature=0.0, max_tokens=512)
    reflection_lm = DeepSeekClient(model="deepseek-chat", temperature=0.7, max_tokens=2048)

    all_results = []

    # ─────────────────────────────────────────
    #  Part 1: MMLU (多学科知识推理)
    # ─────────────────────────────────────────
    print("=" * 60)
    print("  Part 1: MMLU Benchmark")
    print("=" * 60, flush=True)

    with open(os.path.join(data_dir, "mmlu_data.json"), encoding="utf-8") as f:
        mmlu = json.load(f)

    mmlu_train = mmlu["train"]
    mmlu_val = mmlu["val"]
    print(f"  Train: {len(mmlu_train)}, Val: {len(mmlu_val)}")

    mmlu_prompt = (
        "You are a knowledgeable expert. Read the question and the four options carefully. "
        "Answer with ONLY the letter of the correct option (A, B, C, or D). "
        "Do not include any explanation."
    )
    mmlu_adapter = QAAdapter(llm_client=task_lm)

    # MMLU - Reflective
    all_results.append(run_experiment(
        name="MMLU (Reflective)",
        trainset=mmlu_train, valset=mmlu_val,
        strategy=ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=10),
        adapter=mmlu_adapter, evaluator_fn=mmlu_evaluator,
        initial_prompt=mmlu_prompt,
    ))

    # MMLU - FewShot-CoT
    all_results.append(run_experiment(
        name="MMLU (FewShot-CoT)",
        trainset=mmlu_train, valset=mmlu_val,
        strategy=FewShotLearningStrategy(
            num_shots=5, include_reasoning=True,
            include_negative=True, max_negative=2,
            reflection_lm=reflection_lm,
        ),
        adapter=mmlu_adapter, evaluator_fn=mmlu_evaluator,
        initial_prompt=mmlu_prompt,
    ))

    # MMLU - Retrieval-v3
    all_results.append(run_experiment(
        name="MMLU (Retrieval-v3)",
        trainset=mmlu_train, valset=mmlu_val,
        strategy=RetrievalLearningStrategy(
            top_k=3, auto_extract=True, reflection_lm=reflection_lm,
        ),
        adapter=mmlu_adapter, evaluator_fn=mmlu_evaluator,
        initial_prompt=mmlu_prompt,
    ))

    # ─────────────────────────────────────────
    #  Part 2: BBH (Big-Bench Hard)
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Part 2: BBH Benchmark")
    print("=" * 60, flush=True)

    with open(os.path.join(data_dir, "bbh_data.json"), encoding="utf-8") as f:
        bbh = json.load(f)

    bbh_train = bbh["train"]
    bbh_val = bbh["val"]
    print(f"  Train: {len(bbh_train)}, Val: {len(bbh_val)}")

    bbh_prompt = (
        "You are a careful reasoner. Read the question and options, then provide your final answer. "
        "If the question has options like (A), (B), (C)..., you MUST answer with ONLY the option label "
        "in parentheses, e.g. (A) or (B). "
        "If the question asks for True/False or Yes/No, answer with ONLY that word. "
        "Do NOT include any explanation or reasoning in your output."
    )
    bbh_adapter = QAAdapter(llm_client=task_lm)

    # BBH - Reflective
    all_results.append(run_experiment(
        name="BBH (Reflective)",
        trainset=bbh_train, valset=bbh_val,
        strategy=ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=10),
        adapter=bbh_adapter, evaluator_fn=bbh_evaluator,
        initial_prompt=bbh_prompt,
    ))

    # BBH - FewShot-CoT
    all_results.append(run_experiment(
        name="BBH (FewShot-CoT)",
        trainset=bbh_train, valset=bbh_val,
        strategy=FewShotLearningStrategy(
            num_shots=5, include_reasoning=True,
            include_negative=True, max_negative=2,
            reflection_lm=reflection_lm,
        ),
        adapter=bbh_adapter, evaluator_fn=bbh_evaluator,
        initial_prompt=bbh_prompt,
    ))

    # BBH - Retrieval-v3
    all_results.append(run_experiment(
        name="BBH (Retrieval-v3)",
        trainset=bbh_train, valset=bbh_val,
        strategy=RetrievalLearningStrategy(
            top_k=3, auto_extract=True, reflection_lm=reflection_lm,
        ),
        adapter=bbh_adapter, evaluator_fn=bbh_evaluator,
        initial_prompt=bbh_prompt,
    ))

    # ─────────────────────────────────────────
    #  汇总
    # ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL RESULTS - MMLU + BBH Benchmark Experiments")
    print("=" * 80, flush=True)

    header = f"{'Experiment':<30} {'Init':>6} {'Best':>6} {'Final':>6} {'Improv':>8} {'Iter':>5} {'Time':>7}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['experiment']:<30} "
            f"{r['initial_score']:>6.3f} {r['best_score']:>6.3f} {r['final_score']:>6.3f} "
            f"{r['improvement_pct']:>7.1f}% {r['iterations']:>4} {r['elapsed_seconds']:>6.0f}s"
        )

    out_path = os.path.join(data_dir, "mmlu_bbh_experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
