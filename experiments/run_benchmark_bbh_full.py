"""
ICL-Agent BBH 全量基准实验

  BBH: 100 train + 1087 val (5个任务全量)
  任务: boolean_expressions, causal_judgement, date_understanding,
        logical_deduction_five_objects, navigate

预估: ~7-8 小时

运行: python -u experiments/run_benchmark_bbh_full.py
"""

import sys, os, json, time, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(_env_path):
    with open(_env_path, encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ.setdefault(_key.strip(), _val.strip())

from icl_agent.core import AgentOptimizer
from icl_agent.strategies import ReflectiveLearningStrategy, FewShotLearningStrategy, RetrievalLearningStrategy
from icl_agent.adapters import QAAdapter
from icl_agent.utils.llm_client import DeepSeekClient


def bbh_evaluator(output, data):
    predicted = output.get("answer", "").strip()
    expected = data.get("answer", "").strip()
    pred_low = predicted.lower()
    exp_low = expected.lower()
    if pred_low == exp_low:
        return 1.0
    pred_opt = re.search(r'\(([a-zA-Z])\)', predicted)
    exp_opt = re.search(r'\(([a-zA-Z])\)', expected)
    if pred_opt and exp_opt:
        return 1.0 if pred_opt.group(1).lower() == exp_opt.group(1).lower() else 0.0
    tf_map = {'true': 'true', 'false': 'false', 'yes': 'yes', 'no': 'no', 'valid': 'valid', 'invalid': 'invalid'}
    for word, norm in tf_map.items():
        if pred_low.startswith(word) and exp_low == norm:
            return 1.0
    if exp_opt:
        bare = re.search(r'^([a-zA-Z])$', predicted.strip())
        if bare and bare.group(1).lower() == exp_opt.group(1).lower():
            return 1.0
    if exp_low in pred_low and len(pred_low) < len(exp_low) * 5:
        return 1.0
    return 0.0


def run_experiment(name, trainset, valset, strategy, adapter, evaluator_fn, initial_prompt, max_iter=3):
    print(f"\n{'='*60}", flush=True)
    print(f"  {name}")
    print(f"  Strategy: {type(strategy).__name__}, Train: {len(trainset)}, Val: {len(valset)}")
    print(f"{'='*60}\n", flush=True)

    optimizer = AgentOptimizer(
        initial_agent_config={"system_prompt": initial_prompt},
        learning_strategy=strategy, adapter=adapter, evaluator=evaluator_fn,
        max_iterations=max_iter, min_improvement=0.001, failure_threshold=1.0, verbose=True,
    )
    start = time.time()
    result = optimizer.optimize(trainset=trainset, valset=valset)
    elapsed = time.time() - start

    summary = {
        "experiment": name, "strategy": type(strategy).__name__,
        "train_size": len(trainset), "val_size": len(valset),
        "initial_score": round(result.initial_score, 4),
        "final_score": round(result.final_score, 4),
        "best_score": round(result.best_score, 4),
        "improvement_pct": round(result.improvement, 2),
        "iterations": result.total_iterations,
        "elapsed_seconds": round(elapsed, 1),
        "best_instruction": result.best_instruction,
        "score_history": [round(s, 4) for s in result.score_history],
    }
    print(f"\n>>> {name}: {summary['initial_score']} -> {summary['best_score']} ({summary['improvement_pct']:+.1f}%), {summary['iterations']} iters, {elapsed:.0f}s", flush=True)
    print(f"    History: {summary['score_history']}\n", flush=True)
    return summary


def main():
    data_dir = os.path.dirname(__file__)
    task_lm = DeepSeekClient(model="deepseek-chat", temperature=0.0, max_tokens=512)
    reflection_lm = DeepSeekClient(model="deepseek-chat", temperature=0.7, max_tokens=2048)

    with open(os.path.join(data_dir, "bbh_data_full.json"), encoding="utf-8") as f:
        d = json.load(f)

    bbh_train = d["train"]
    bbh_val = d["val"]

    print("="*60)
    print(f"  BBH Full Benchmark")
    print(f"  Train: {len(bbh_train)}, Val: {len(bbh_val)} (total: {len(bbh_train)+len(bbh_val)})")
    print(f"  Tasks: boolean_expressions, causal_judgement, date_understanding,")
    print(f"         logical_deduction_five_objects, navigate")
    print("="*60, flush=True)

    bbh_prompt = (
        "You are a careful reasoner. Read the question and options, then provide your final answer. "
        "If the question has options like (A), (B), (C)..., you MUST answer with ONLY the option label "
        "in parentheses, e.g. (A) or (B). "
        "If the question asks for True/False or Yes/No, answer with ONLY that word. "
        "Do NOT include any explanation or reasoning in your output."
    )
    bbh_adapter = QAAdapter(llm_client=task_lm)

    all_results = []
    strategies = [
        ("Reflective", ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=10)),
        ("FewShot-CoT", FewShotLearningStrategy(num_shots=5, include_reasoning=True, include_negative=True, max_negative=2, reflection_lm=reflection_lm)),
        ("Retrieval-v3", RetrievalLearningStrategy(top_k=3, auto_extract=True, reflection_lm=reflection_lm)),
    ]

    for sname, strategy in strategies:
        all_results.append(run_experiment(
            f"BBH-Full ({sname})", bbh_train, bbh_val, strategy,
            bbh_adapter, bbh_evaluator, bbh_prompt))

    # 汇总
    print("\n" + "="*80, flush=True)
    print("  FINAL RESULTS: BBH Full Benchmark")
    print("="*80)
    print(f"{'Experiment':<30} {'Train':>5} {'Val':>5} {'Init':>6} {'Best':>6} {'Improv':>8} {'Iter':>4} {'Time':>7}")
    print("-"*78)
    for r in all_results:
        print(f"{r['experiment']:<30} {r['train_size']:>5} {r['val_size']:>5} {r['initial_score']:>6.3f} {r['best_score']:>6.3f} {r['improvement_pct']:>7.1f}% {r['iterations']:>4} {r['elapsed_seconds']:>6.0f}s")

    out_path = os.path.join(data_dir, "bbh_full_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
