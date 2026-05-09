"""
Controlled ablation for Retrieval ContextBuilder integration.

This experiment keeps the model, datasets, optimizer settings, Retrieval
parameters, and knowledge extraction path fixed. The only intended variable is:

  Full Retrieval:        use_context_builder=True
  w/o ContextBuilder:    use_context_builder=False

By default, reflection_lm is not passed to Retrieval so knowledge extraction
uses the deterministic heuristic path. This reduces random variation from
LLM-generated rules and makes the ablation cleaner for thesis reporting.
"""

import argparse
import json
import os
import re
import string
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
from icl_agent.core import AgentOptimizer, ContextBudget
from icl_agent.strategies import RetrievalLearningStrategy
from icl_agent.utils.llm_client import DeepSeekClient

from datasets import LOGIC_TRAINSET_EXTENDED, LOGIC_VALSET_EXTENDED
from datasets_hard import LOGIC_HARD_TRAINSET, LOGIC_HARD_VALSET


def load_json_dataset(name):
    with open(os.path.join(EXPERIMENT_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def strict_evaluator(output, data):
    predicted = output.get("answer", "").strip().lower()
    expected = data.get("answer", "").strip().lower()
    return 1.0 if predicted == expected else 0.0


def hotpotqa_evaluator(output, data):
    predicted = output.get("answer", "").strip()
    expected = data.get("answer", "").strip()

    def normalize(text):
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())

    pred_norm = normalize(predicted)
    exp_norm = normalize(expected)
    if pred_norm == exp_norm:
        return 1.0
    pred_tokens = pred_norm.split()
    exp_tokens = exp_norm.split()
    common = set(pred_tokens) & set(exp_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(exp_tokens) if exp_tokens else 0.0
    return round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0


def mmlu_evaluator(output, data):
    predicted = output.get("answer", "").strip().upper()
    expected = data.get("answer", "").strip().upper()
    match = re.search(r"\b([ABCD])\b", predicted)
    if match:
        predicted = match.group(1)
    return 1.0 if predicted == expected else 0.0


def make_strategy(use_context_builder):
    return RetrievalLearningStrategy(
        top_k=3,
        auto_extract=True,
        reflection_lm=None,
        use_context_builder=use_context_builder,
        context_budget=ContextBudget(
            total_chars=2700,
            system_chars=900,
            knowledge_chars=1500,
            memory_chars=0,
            examples_chars=500,
            plan_chars=300,
        ),
    )


def build_specs(task_lm):
    hotpot = load_json_dataset("hotpotqa_data_v2.json")
    mmlu = load_json_dataset("mmlu_data_v2.json")

    qa_adapter = QAAdapter(llm_client=task_lm)
    hotpot_adapter = QAAdapter(llm_client=task_lm)
    mmlu_adapter = QAAdapter(llm_client=task_lm)

    datasets = [
        {
            "dataset": "Logic-Std",
            "trainset": LOGIC_TRAINSET_EXTENDED,
            "valset": LOGIC_VALSET_EXTENDED,
            "adapter": qa_adapter,
            "evaluator": strict_evaluator,
            "initial_prompt": "You are a helpful assistant. Answer the question.",
        },
        {
            "dataset": "Logic-Hard",
            "trainset": LOGIC_HARD_TRAINSET,
            "valset": LOGIC_HARD_VALSET,
            "adapter": qa_adapter,
            "evaluator": strict_evaluator,
            "initial_prompt": (
                "You are a logic puzzle expert. Think step by step and reason carefully. "
                "Answer the question following the exact format requested."
            ),
        },
        {
            "dataset": "HotpotQA-v2",
            "trainset": hotpot["train"],
            "valset": hotpot["val"],
            "adapter": hotpot_adapter,
            "evaluator": hotpotqa_evaluator,
            "initial_prompt": (
                "You are a knowledgeable assistant. Answer the question concisely. "
                "Your answer should be as short as possible - typically a name, date, "
                "number, or a few words. Do NOT provide explanations."
            ),
        },
        {
            "dataset": "MMLU-v2",
            "trainset": mmlu["train"],
            "valset": mmlu["val"],
            "adapter": mmlu_adapter,
            "evaluator": mmlu_evaluator,
            "initial_prompt": (
                "You are a knowledgeable expert. Read the question and the four options carefully. "
                "Answer with ONLY the letter of the correct option (A, B, C, or D). "
                "Do not include any explanation."
            ),
        },
    ]

    specs = []
    for dataset in datasets:
        for label, use_context_builder in [
            ("Full", True),
            ("w/o ContextBuilder", False),
        ]:
            spec = dataset.copy()
            spec["variant"] = label
            spec["use_context_builder"] = use_context_builder
            spec["experiment"] = f"{dataset['dataset']} Retrieval ({label})"
            specs.append(spec)
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
    print(f"use_context_builder={spec['use_context_builder']}", flush=True)
    print("=" * 72 + "\n", flush=True)

    strategy = make_strategy(spec["use_context_builder"])
    optimizer = AgentOptimizer(
        initial_agent_config={"system_prompt": spec["initial_prompt"]},
        learning_strategy=strategy,
        adapter=spec["adapter"],
        evaluator=spec["evaluator"],
        max_iterations=max_iter,
        min_improvement=0.001,
        failure_threshold=1.0,
        verbose=True,
    )

    start = time.time()
    result = optimizer.optimize(trainset=spec["trainset"], valset=spec["valset"])
    elapsed = time.time() - start

    context_package_chars = None
    if strategy.last_context_package is not None:
        context_package_chars = len(strategy.last_context_package.to_prompt())

    summary = {
        "experiment": spec["experiment"],
        "dataset": spec["dataset"],
        "variant": spec["variant"],
        "use_context_builder": spec["use_context_builder"],
        "strategy": type(strategy).__name__,
        "train_size": len(spec["trainset"]),
        "val_size": len(spec["valset"]),
        "initial_score": round(result.initial_score, 4),
        "final_score": round(result.final_score, 4),
        "best_score": round(result.best_score, 4),
        "improvement_pct": round(result.improvement, 2),
        "iterations": result.total_iterations,
        "elapsed_seconds": round(elapsed, 1),
        "score_history": [round(s, 4) for s in result.score_history],
        "candidate_pool_size": len(result.candidate_pool),
        "context_package_chars": context_package_chars,
        "best_instruction_chars": len(result.best_instruction),
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument(
        "--output",
        default=os.path.join(EXPERIMENT_DIR, "controlled_contextbuilder_ablation_results.json"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args(argv)

    task_lm = DeepSeekClient(model="deepseek-chat", temperature=0.0, max_tokens=1024)
    specs = build_specs(task_lm)
    if args.only:
        wanted = [item.lower() for item in args.only]
        specs = [
            spec for spec in specs
            if any(item in spec["experiment"].lower() or item in spec["dataset"].lower() for item in wanted)
        ]

    results = load_existing(args.output) if args.resume else []
    completed = {item["experiment"] for item in results}

    print("=" * 72, flush=True)
    print("Controlled Retrieval ContextBuilder Ablation", flush=True)
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
                "dataset": spec["dataset"],
                "variant": spec["variant"],
                "use_context_builder": spec["use_context_builder"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"\n[ERROR] {spec['experiment']}: {summary['error']}", flush=True)
        results.append(summary)
        save_results(args.output, results)
        completed.add(spec["experiment"])
        print(f"Checkpoint saved: {args.output}", flush=True)

    print("\n" + "=" * 88, flush=True)
    print("FINAL CONTROLLED ABLATION RESULTS", flush=True)
    print("=" * 88, flush=True)
    print(f"{'Experiment':<42} {'Init':>7} {'Best':>7} {'Final':>7} {'Imp':>8} {'Iter':>4}")
    print("-" * 88, flush=True)
    for item in results:
        if "error" in item:
            print(f"{item['experiment']:<42} ERROR: {item['error']}", flush=True)
            continue
        print(
            f"{item['experiment']:<42} {item['initial_score']:>7.4f} "
            f"{item['best_score']:>7.4f} {item['final_score']:>7.4f} "
            f"{item['improvement_pct']:>7.1f}% {item['iterations']:>4}",
            flush=True,
        )


if __name__ == "__main__":
    main()
