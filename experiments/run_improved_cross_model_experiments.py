"""
Run improved-framework cross-model comparison experiments.

Matrix:
  Models: DeepSeek, Qwen, Kimi
  Datasets: Logic-Std, Logic-Hard
  Strategies: Reflective, FewShot-CoT, Retrieval-v3

The script checkpoints after every experiment and can be resumed safely.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List


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
from icl_agent.strategies import (
    FewShotLearningStrategy,
    ReflectiveLearningStrategy,
    RetrievalLearningStrategy,
)
from icl_agent.utils.llm_client import DeepSeekClient

from datasets import LOGIC_TRAINSET_EXTENDED, LOGIC_VALSET_EXTENDED
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
    if strategy_name == "FewShot-CoT":
        return FewShotLearningStrategy(
            num_shots=5,
            include_reasoning=True,
            include_negative=True,
            max_negative=2,
            reflection_lm=reflection_lm,
        )
    if strategy_name == "Retrieval-v3":
        return RetrievalLearningStrategy(
            top_k=3,
            auto_extract=True,
            reflection_lm=reflection_lm,
        )
    raise ValueError(f"Unknown strategy: {strategy_name}")


def build_specs(model_names):
    datasets = [
        {
            "dataset": "Logic-Std",
            "trainset": LOGIC_TRAINSET_EXTENDED,
            "valset": LOGIC_VALSET_EXTENDED,
            "initial_prompt": "You are a helpful assistant. Answer the question.",
        },
        {
            "dataset": "Logic-Hard",
            "trainset": LOGIC_HARD_TRAINSET,
            "valset": LOGIC_HARD_VALSET,
            "initial_prompt": (
                "You are a logic puzzle expert. Think step by step and reason carefully. "
                "Answer the question following the exact format requested."
            ),
        },
    ]
    strategies = ["Reflective", "FewShot-CoT", "Retrieval-v3"]

    specs = []
    for model_name in model_names:
        for dataset in datasets:
            for strategy_name in strategies:
                specs.append({
                    **dataset,
                    "model_name": model_name,
                    "strategy_name": strategy_name,
                    "experiment": f"{model_name}/{dataset['dataset']}/{strategy_name}",
                })
    return specs


def load_existing(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_results(path, results):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_experiment(spec, max_iter):
    print("\n" + "=" * 72, flush=True)
    print(f"Experiment: {spec['experiment']}", flush=True)
    print(f"Train: {len(spec['trainset'])}, Val: {len(spec['valset'])}", flush=True)
    print("=" * 72 + "\n", flush=True)

    task_lm = make_client(spec["model_name"], temperature=0.0, max_tokens=512)
    reflection_lm = make_client(spec["model_name"], temperature=0.7, max_tokens=2048)
    adapter = QAAdapter(llm_client=task_lm)
    strategy = make_strategy(spec["strategy_name"], reflection_lm)

    optimizer = AgentOptimizer(
        initial_agent_config={"system_prompt": spec["initial_prompt"]},
        learning_strategy=strategy,
        adapter=adapter,
        evaluator=strict_evaluator,
        max_iterations=max_iter,
        min_improvement=0.001,
        failure_threshold=1.0,
        verbose=True,
    )

    start = time.time()
    result = optimizer.optimize(trainset=spec["trainset"], valset=spec["valset"])
    elapsed = time.time() - start

    summary = {
        "experiment": spec["experiment"],
        "model": spec["model_name"],
        "dataset": spec["dataset"],
        "strategy": spec["strategy_name"],
        "train_size": len(spec["trainset"]),
        "val_size": len(spec["valset"]),
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
        f"\n>>> {spec['experiment']}: {summary['initial_score']:.4f} -> "
        f"{summary['best_score']:.4f} ({summary['improvement_pct']:+.1f}%), "
        f"{summary['iterations']} iters, {elapsed:.0f}s",
        flush=True,
    )
    print(f"    History: {summary['score_history']}\n", flush=True)
    return summary


def filter_specs(specs, only_patterns):
    if not only_patterns:
        return specs
    lowered = [pattern.lower() for pattern in only_patterns]
    return [
        spec for spec in specs
        if any(pattern in spec["experiment"].lower() for pattern in lowered)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument(
        "--output",
        default=os.path.join(EXPERIMENT_DIR, "improved_cross_model_results.json"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", action="append", choices=sorted(MODEL_CONFIGS), default=[])
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args(argv)

    model_names = args.model if args.model else ["DeepSeek", "Qwen", "Kimi"]
    specs = filter_specs(build_specs(model_names), args.only)
    results = load_existing(args.output) if args.resume else []
    completed = {item["experiment"] for item in results}

    print("=" * 72, flush=True)
    print("Improved Framework Cross-Model Experiments", flush=True)
    print(f"Models: {', '.join(model_names)}", flush=True)
    print(f"Max iterations: {args.max_iter}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Planned experiments: {len(specs)}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)
    print("=" * 72, flush=True)

    for spec in specs:
        if args.resume and spec["experiment"] in completed:
            print(f"\nSkipping completed: {spec['experiment']}", flush=True)
            continue
        try:
            summary = run_experiment(spec, args.max_iter)
        except Exception as exc:
            summary = {
                "experiment": spec["experiment"],
                "model": spec["model_name"],
                "dataset": spec["dataset"],
                "strategy": spec["strategy_name"],
                "train_size": len(spec["trainset"]),
                "val_size": len(spec["valset"]),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"\n[ERROR] {spec['experiment']}: {summary['error']}", flush=True)
        results.append(summary)
        save_results(args.output, results)
        completed.add(spec["experiment"])
        print(f"Checkpoint saved: {args.output}", flush=True)

    print("\n" + "=" * 92, flush=True)
    print("FINAL CROSS-MODEL RESULTS", flush=True)
    print("=" * 92, flush=True)
    print(f"{'Experiment':<36} {'Train':>5} {'Val':>5} {'Init':>7} {'Best':>7} {'Final':>7} {'Imp':>8} {'Iter':>4}")
    print("-" * 92, flush=True)
    for item in results:
        if "error" in item:
            print(f"{item['experiment']:<36} ERROR: {item['error']}", flush=True)
            continue
        print(
            f"{item['experiment']:<36} {item['train_size']:>5} {item['val_size']:>5} "
            f"{item['initial_score']:>7.4f} {item['best_score']:>7.4f} "
            f"{item['final_score']:>7.4f} {item['improvement_pct']:>7.1f}% "
            f"{item['iterations']:>4}",
            flush=True,
        )


if __name__ == "__main__":
    main()
