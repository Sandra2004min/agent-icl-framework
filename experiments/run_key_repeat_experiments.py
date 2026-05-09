"""
Repeat the key improved-framework configurations for stability analysis.

Default matrix:
  - DeepSeek / Logic-Hard / Reflective
  - Qwen     / Logic-Hard / Reflective
  - Qwen     / Logic-Hard / Retrieval-v3
  - Kimi     / Logic-Hard / Reflective

Each configuration is repeated 3 times by default. Results and aggregate
statistics are saved after every run so the script can resume safely.
"""

import argparse
import json
import os
import statistics
import sys
import time


EXPERIMENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")

sys.path.insert(0, EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)


_env_path = os.path.join(REPO_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())


from icl_agent.adapters import QAAdapter
from icl_agent.core import AgentOptimizer
from icl_agent.strategies import ReflectiveLearningStrategy, RetrievalLearningStrategy
from icl_agent.utils.llm_client import DeepSeekClient

from datasets_hard import LOGIC_HARD_TRAINSET, LOGIC_HARD_VALSET


MODEL_CONFIGS = {
    "DeepSeek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "Qwen": {
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "Kimi": {
        "api_key_env": "KIMI_API_KEY",
        "base_url_env": "KIMI_BASE_URL",
        "default_base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
    },
}


KEY_CONFIGS = [
    ("DeepSeek", "Reflective"),
    ("Qwen", "Reflective"),
    ("Qwen", "Retrieval-v3"),
    ("Kimi", "Reflective"),
]


def strict_evaluator(output, data):
    predicted = output.get("answer", "").strip().lower()
    expected = data.get("answer", "").strip().lower()
    return 1.0 if predicted == expected else 0.0


def make_client(model_name, temperature, max_tokens):
    cfg = MODEL_CONFIGS[model_name]
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise ValueError(f"Missing API key env: {cfg['api_key_env']}")
    return DeepSeekClient(
        api_key=api_key,
        base_url=os.getenv(cfg["base_url_env"], cfg["default_base_url"]),
        model=cfg["model"],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def make_strategy(strategy_name, reflection_lm):
    if strategy_name == "Reflective":
        return ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=8)
    if strategy_name == "Retrieval-v3":
        return RetrievalLearningStrategy(top_k=3, auto_extract=True, reflection_lm=reflection_lm)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def load_json(path, fallback):
    if not os.path.exists(path):
        return fallback
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_stats(results):
    grouped = {}
    for item in results:
        if "error" in item:
            continue
        key = f"{item['model']}/Logic-Hard/{item['strategy']}"
        grouped.setdefault(key, []).append(item)

    stats = {}
    for key, items in grouped.items():
        best_scores = [item["best_score"] for item in items]
        final_scores = [item["final_score"] for item in items]
        improvements = [item["improvement_pct"] for item in items]
        stats[key] = {
            "runs": len(items),
            "initial_scores": [item["initial_score"] for item in items],
            "best_scores": best_scores,
            "best_mean": round(statistics.mean(best_scores), 4),
            "best_std": round(statistics.stdev(best_scores), 4) if len(best_scores) > 1 else 0.0,
            "final_scores": final_scores,
            "final_mean": round(statistics.mean(final_scores), 4),
            "final_std": round(statistics.stdev(final_scores), 4) if len(final_scores) > 1 else 0.0,
            "improvement_pcts": improvements,
            "improvement_mean": round(statistics.mean(improvements), 2),
            "improvement_std": round(statistics.stdev(improvements), 2) if len(improvements) > 1 else 0.0,
            "score_histories": [item["score_history"] for item in items],
        }
    return stats


def run_once(model_name, strategy_name, repeat_id, max_iter):
    experiment = f"Repeat{repeat_id}/{model_name}/Logic-Hard/{strategy_name}"
    print("\n" + "=" * 72, flush=True)
    print(f"Experiment: {experiment}", flush=True)
    print("=" * 72 + "\n", flush=True)

    task_lm = make_client(model_name, temperature=0.0, max_tokens=512)
    reflection_lm = make_client(model_name, temperature=0.7, max_tokens=2048)
    adapter = QAAdapter(llm_client=task_lm)
    strategy = make_strategy(strategy_name, reflection_lm)
    initial_prompt = (
        "You are a logic puzzle expert. Think step by step and reason carefully. "
        "Answer the question following the exact format requested."
    )

    optimizer = AgentOptimizer(
        initial_agent_config={"system_prompt": initial_prompt},
        learning_strategy=strategy,
        adapter=adapter,
        evaluator=strict_evaluator,
        max_iterations=max_iter,
        min_improvement=0.001,
        failure_threshold=1.0,
        verbose=True,
    )

    start = time.time()
    result = optimizer.optimize(trainset=LOGIC_HARD_TRAINSET, valset=LOGIC_HARD_VALSET)
    elapsed = time.time() - start

    summary = {
        "experiment": experiment,
        "model": model_name,
        "dataset": "Logic-Hard",
        "strategy": strategy_name,
        "repeat_id": repeat_id,
        "train_size": len(LOGIC_HARD_TRAINSET),
        "val_size": len(LOGIC_HARD_VALSET),
        "initial_score": round(result.initial_score, 4),
        "final_score": round(result.final_score, 4),
        "best_score": round(result.best_score, 4),
        "improvement_pct": round(result.improvement, 2),
        "iterations": result.total_iterations,
        "elapsed_seconds": round(elapsed, 1),
        "score_history": [round(score, 4) for score in result.score_history],
        "candidate_pool_size": len(result.candidate_pool),
        "best_instruction": result.best_instruction,
    }
    print(
        f"\n>>> {experiment}: {summary['initial_score']:.4f} -> "
        f"{summary['best_score']:.4f} ({summary['improvement_pct']:+.1f}%), "
        f"{summary['iterations']} iters, {elapsed:.0f}s",
        flush=True,
    )
    print(f"    History: {summary['score_history']}\n", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--output",
        default=os.path.join(EXPERIMENT_DIR, "key_repeat_results.json"),
    )
    parser.add_argument(
        "--stats-output",
        default=os.path.join(EXPERIMENT_DIR, "key_repeat_stats.json"),
    )
    args = parser.parse_args(argv)

    configs = KEY_CONFIGS
    if args.only:
        wanted = [item.lower() for item in args.only]
        configs = [
            cfg for cfg in configs
            if any(item in f"{cfg[0]}/{cfg[1]}".lower() for item in wanted)
        ]

    results = load_json(args.output, []) if args.resume else []
    completed = {item["experiment"] for item in results}

    print("=" * 72, flush=True)
    print("Key Configuration Repeat Experiments", flush=True)
    print(f"Configs: {len(configs)}, repeats: {args.repeats}, max_iter: {args.max_iter}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)
    print("=" * 72, flush=True)

    for repeat_id in range(1, args.repeats + 1):
        for model_name, strategy_name in configs:
            experiment = f"Repeat{repeat_id}/{model_name}/Logic-Hard/{strategy_name}"
            if args.resume and experiment in completed:
                print(f"\nSkipping completed: {experiment}", flush=True)
                continue
            try:
                summary = run_once(model_name, strategy_name, repeat_id, args.max_iter)
            except Exception as exc:
                summary = {
                    "experiment": experiment,
                    "model": model_name,
                    "dataset": "Logic-Hard",
                    "strategy": strategy_name,
                    "repeat_id": repeat_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"\n[ERROR] {experiment}: {summary['error']}", flush=True)
            results.append(summary)
            completed.add(experiment)
            save_json(args.output, results)
            save_json(args.stats_output, compute_stats(results))
            print(f"Checkpoint saved: {args.output}", flush=True)

    stats = compute_stats(results)
    print("\n" + "=" * 80, flush=True)
    print("REPEAT STATS", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Config':<36} {'Runs':>4} {'Best Mean':>10} {'Best Std':>9} {'Imp Mean':>9}")
    print("-" * 80, flush=True)
    for key, item in stats.items():
        print(
            f"{key:<36} {item['runs']:>4} {item['best_mean']:>10.4f} "
            f"{item['best_std']:>9.4f} {item['improvement_mean']:>8.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
