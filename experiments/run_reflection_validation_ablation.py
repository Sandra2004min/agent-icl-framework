"""
Controlled ablation for reflection hypothesis validation.

Variants:
  Full validation:
    ReflectiveLearningStrategy with a small performance gate. Before accepting
    a reflected prompt, it evaluates current vs proposed prompt on up to five
    failed training cases and only accepts if the proposed prompt improves.

  w/o validation:
    A subclass that accepts every reflected hypothesis, approximating the old
    direct-apply behavior.

The experiment focuses on Logic-Hard, where reflection has enough room to help
and bad reflections can cause visible regressions.
"""

import argparse
import json
import os
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
from icl_agent.strategies import ReflectiveLearningStrategy
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
}


class NoValidationReflectiveStrategy(ReflectiveLearningStrategy):
    """Reflective strategy variant that accepts every hypothesis."""

    def _validate_hypothesis(
        self,
        hypothesis,
        current_config,
        proposed_config,
        contexts,
        failed_contexts,
    ):
        hypothesis.mark_validation(
            accepted=True,
            reason="validation_disabled",
        )
        return True


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


def make_gate_validator(adapter, max_cases, min_delta):
    def validator(current_config, proposed_config, hypothesis, contexts, failed_contexts):
        gate_contexts = failed_contexts[:max_cases] if failed_contexts else contexts[:max_cases]
        if not gate_contexts:
            return {
                "accepted": False,
                "reason": "no_gate_contexts",
                "score_before": 0.0,
                "score_after": 0.0,
            }

        before_scores = []
        after_scores = []
        for ctx in gate_contexts:
            before_output = adapter.execute(current_config, ctx.input_data)
            after_output = adapter.execute(proposed_config, ctx.input_data)
            before_scores.append(strict_evaluator(before_output, ctx.input_data))
            after_scores.append(strict_evaluator(after_output, ctx.input_data))

        score_before = sum(before_scores) / len(before_scores)
        score_after = sum(after_scores) / len(after_scores)
        return {
            "accepted": score_after - score_before >= min_delta,
            "reason": "gate_improved" if score_after > score_before else "gate_not_improved",
            "score_before": round(score_before, 4),
            "score_after": round(score_after, 4),
        }

    return validator


def make_strategy(variant, reflection_lm, adapter, max_cases, min_delta):
    if variant == "FullValidation":
        return ReflectiveLearningStrategy(
            reflection_lm=reflection_lm,
            max_failures=8,
            hypothesis_validator=make_gate_validator(adapter, max_cases, min_delta),
            validation_min_delta=min_delta,
        )
    if variant == "NoValidation":
        return NoValidationReflectiveStrategy(
            reflection_lm=reflection_lm,
            max_failures=8,
        )
    raise ValueError(f"Unknown variant: {variant}")


def load_json(path, fallback):
    if not os.path.exists(path):
        return fallback
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_experiment(model_name, variant, max_iter, max_cases, min_delta):
    experiment = f"{model_name}/Logic-Hard/Reflective/{variant}"
    print("\n" + "=" * 72, flush=True)
    print(f"Experiment: {experiment}", flush=True)
    print("=" * 72 + "\n", flush=True)

    task_lm = make_client(model_name, temperature=0.0, max_tokens=512)
    reflection_lm = make_client(model_name, temperature=0.7, max_tokens=2048)
    adapter = QAAdapter(llm_client=task_lm)
    strategy = make_strategy(variant, reflection_lm, adapter, max_cases, min_delta)
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

    hypothesis = getattr(strategy, "last_hypothesis", None)
    summary = {
        "experiment": experiment,
        "model": model_name,
        "dataset": "Logic-Hard",
        "strategy": "Reflective",
        "variant": variant,
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
        "last_hypothesis": hypothesis.to_dict() if hypothesis is not None else None,
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", action="append", choices=sorted(MODEL_CONFIGS), default=[])
    parser.add_argument("--gate-cases", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument(
        "--output",
        default=os.path.join(EXPERIMENT_DIR, "reflection_validation_ablation_results.json"),
    )
    args = parser.parse_args(argv)

    models = args.model if args.model else ["DeepSeek", "Qwen"]
    variants = ["FullValidation", "NoValidation"]
    results = load_json(args.output, []) if args.resume else []
    completed = {item["experiment"] for item in results}

    print("=" * 72, flush=True)
    print("Reflection Validation Ablation", flush=True)
    print(f"Models: {', '.join(models)}", flush=True)
    print(f"Variants: {', '.join(variants)}", flush=True)
    print(f"Gate cases: {args.gate_cases}, min_delta: {args.min_delta}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)
    print("=" * 72, flush=True)

    for model_name in models:
        for variant in variants:
            experiment = f"{model_name}/Logic-Hard/Reflective/{variant}"
            if args.resume and experiment in completed:
                print(f"\nSkipping completed: {experiment}", flush=True)
                continue
            try:
                summary = run_experiment(
                    model_name=model_name,
                    variant=variant,
                    max_iter=args.max_iter,
                    max_cases=args.gate_cases,
                    min_delta=args.min_delta,
                )
            except Exception as exc:
                summary = {
                    "experiment": experiment,
                    "model": model_name,
                    "dataset": "Logic-Hard",
                    "strategy": "Reflective",
                    "variant": variant,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"\n[ERROR] {experiment}: {summary['error']}", flush=True)
            results.append(summary)
            completed.add(experiment)
            save_json(args.output, results)
            print(f"Checkpoint saved: {args.output}", flush=True)

    print("\n" + "=" * 88, flush=True)
    print("FINAL REFLECTION VALIDATION ABLATION", flush=True)
    print("=" * 88, flush=True)
    print(f"{'Experiment':<48} {'Init':>7} {'Best':>7} {'Final':>7} {'Imp':>8} {'Iter':>4}")
    print("-" * 88, flush=True)
    for item in results:
        if "error" in item:
            print(f"{item['experiment']:<48} ERROR: {item['error']}", flush=True)
            continue
        print(
            f"{item['experiment']:<48} {item['initial_score']:>7.4f} "
            f"{item['best_score']:>7.4f} {item['final_score']:>7.4f} "
            f"{item['improvement_pct']:>7.1f}% {item['iterations']:>4}",
            flush=True,
        )


if __name__ == "__main__":
    main()
