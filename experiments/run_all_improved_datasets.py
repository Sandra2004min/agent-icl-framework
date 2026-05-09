"""
Run the improved ICL-Agent framework across the project datasets.

The script checkpoints after every experiment so long runs can resume safely.
By default it covers the local core datasets, the larger v2 benchmark datasets,
and the small BBH split. The very large BBH-full split is available with
--include-bbh-full because it can take many hours by itself.
"""

import argparse
import json
import os
import re
import string
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List


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


from icl_agent.adapters import CodeAdapter, MathAdapter, QAAdapter
from icl_agent.adapters.base_adapter import BaseAdapter
from icl_agent.core import AgentOptimizer
from icl_agent.strategies import (
    FewShotLearningStrategy,
    ReflectiveLearningStrategy,
    RetrievalLearningStrategy,
)
from icl_agent.utils.llm_client import DeepSeekClient

from datasets import (
    CODE_FIX_TRAINSET,
    CODE_FIX_VALSET,
    GSM8K_TRAINSET,
    GSM8K_VALSET,
    LOGIC_TRAINSET_EXTENDED,
    LOGIC_VALSET_EXTENDED,
)
from datasets_hard import (
    CODE_HARD_TRAINSET,
    CODE_HARD_VALSET,
    LOGIC_HARD_TRAINSET,
    LOGIC_HARD_VALSET,
    MATH_HARD_TRAINSET,
    MATH_HARD_VALSET,
)


class HumanEvalAdapter(BaseAdapter):
    def __init__(self, llm_client):
        super().__init__(name="HumanEvalAdapter")
        self.llm_client = llm_client

    def execute(self, agent_config, input_data):
        prompt = input_data["prompt"]
        messages = [
            {"role": "system", "content": agent_config.get("system_prompt", "")},
            {
                "role": "user",
                "content": (
                    "Complete the following Python function. Return ONLY the "
                    "function body (the implementation lines), without repeating "
                    f"the function signature or any extra text.\n\n{prompt}"
                ),
            },
        ]
        try:
            response = self.llm_client(messages)
        except Exception as exc:
            response = f"# Error: {exc}"
        return {"code": response, "prompt": prompt}

    def evaluate(self, output, ground_truth):
        prompt = ground_truth["prompt"]
        test_code = ground_truth["test"]
        entry_point = ground_truth["entry_point"]
        code_body = self._extract_code(output.get("code", ""))
        full_code = prompt + code_body + "\n" + test_code + f"\ncheck({entry_point})\n"
        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return 1.0 if result.returncode == 0 else 0.0
        except Exception:
            return 0.0

    def _extract_code(self, text):
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0]
            return "\n" + code.strip() + "\n"
        if "```" in text:
            code = text.split("```")[1].split("```")[0]
            lines = code.strip().split("\n")
            if lines and lines[0].strip() in ("python", "py", ""):
                lines = lines[1:]
            return "\n" + "\n".join(lines) + "\n"
        lines = text.strip().split("\n")
        indented = []
        for line in lines:
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                indented.append("    " + line)
            else:
                indented.append(line)
        return "\n" + "\n".join(indented) + "\n"


def strict_evaluator(output, data):
    predicted = output.get("answer", "").strip().lower()
    expected = data.get("answer", "").strip().lower()
    return 1.0 if predicted == expected else 0.0


def math_evaluator(adapter):
    def evaluator(output, data):
        return adapter.evaluate(output, data)

    return evaluator


def code_evaluator(adapter):
    def evaluator(output, data):
        return adapter.evaluate(output, data)

    return evaluator


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
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(exp_tokens) if exp_tokens else 0
    return round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0


def mmlu_evaluator(output, data):
    predicted = output.get("answer", "").strip().upper()
    expected = data.get("answer", "").strip().upper()
    match = re.search(r"\b([ABCD])\b", predicted)
    if match:
        predicted = match.group(1)
    return 1.0 if predicted == expected else 0.0


def bbh_evaluator(output, data):
    predicted = output.get("answer", "").strip()
    expected = data.get("answer", "").strip()
    pred_low = predicted.lower()
    exp_low = expected.lower()
    if pred_low == exp_low:
        return 1.0
    pred_opt = re.search(r"\(([a-zA-Z])\)", predicted)
    exp_opt = re.search(r"\(([a-zA-Z])\)", expected)
    if pred_opt and exp_opt:
        return 1.0 if pred_opt.group(1).lower() == exp_opt.group(1).lower() else 0.0
    tf_map = {
        "true": "true",
        "false": "false",
        "yes": "yes",
        "no": "no",
        "valid": "valid",
        "invalid": "invalid",
    }
    for word, norm in tf_map.items():
        if pred_low.startswith(word) and exp_low == norm:
            return 1.0
    if exp_opt:
        bare = re.search(r"^([a-zA-Z])$", predicted.strip())
        if bare and bare.group(1).lower() == exp_opt.group(1).lower():
            return 1.0
    if exp_low in pred_low and len(pred_low) < len(exp_low) * 5:
        return 1.0
    return 0.0


def load_json_dataset(name):
    with open(os.path.join(EXPERIMENT_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def make_strategy(strategy_name, reflection_lm, task_type):
    if strategy_name == "Reflective":
        max_failures = 5 if task_type == "code" else 10
        return ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=max_failures)
    if strategy_name == "FewShot-CoT":
        return FewShotLearningStrategy(
            num_shots=3 if task_type in {"math", "codegen"} else 5,
            include_reasoning=True,
            include_negative=True,
            max_negative=2,
            reflection_lm=reflection_lm,
        )
    if strategy_name == "Retrieval-v3":
        return RetrievalLearningStrategy(top_k=3, auto_extract=True, reflection_lm=reflection_lm)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def build_specs(task_lm, reflection_lm, include_bbh_full=False):
    specs = []

    math_prompt = (
        "You are a math problem solver. Solve the problem step by step. "
        "At the end, write your final numerical answer after '#### '. For example: #### 42"
    )
    hard_math_prompt = (
        "You are a math competition solver. Solve the problem step by step using rigorous "
        "mathematical reasoning. At the end, write your final numerical answer after '#### '. "
        "For example: #### 42"
    )
    logic_prompt = "You are a helpful assistant. Answer the question."
    hard_logic_prompt = (
        "You are a logic puzzle expert. Think step by step and reason carefully. "
        "Answer the question following the exact format requested."
    )
    code_prompt = (
        "You are an expert software engineer specializing in debugging. "
        "Identify the bug precisely and explain the fix concisely."
    )
    hotpot_prompt = (
        "You are a knowledgeable assistant. Answer the question concisely. Your answer should "
        "be as short as possible - typically a name, date, number, or a few words. "
        "Do NOT provide explanations."
    )
    humaneval_prompt = (
        "You are an expert Python programmer. Complete the function implementation based on "
        "the docstring. Return ONLY the function body code (properly indented), without "
        "repeating the function signature, adding tests, or any explanation."
    )
    mmlu_prompt = (
        "You are a knowledgeable expert. Read the question and the four options carefully. "
        "Answer with ONLY the letter of the correct option (A, B, C, or D). Do not include any explanation."
    )
    bbh_prompt = (
        "You are a careful reasoner. Read the question and options, then provide your final answer. "
        "If the question has options like (A), (B), (C)..., you MUST answer with ONLY the option label "
        "in parentheses, e.g. (A) or (B). If the question asks for True/False or Yes/No, answer with ONLY "
        "that word. Do NOT include any explanation or reasoning in your output."
    )

    def add_dataset(dataset_name, trainset, valset, adapter, evaluator_fn, prompt, task_type, strategies):
        for strategy_name in strategies:
            specs.append(
                {
                    "experiment": f"{dataset_name} ({strategy_name})",
                    "dataset": dataset_name,
                    "strategy_name": strategy_name,
                    "strategy": make_strategy(strategy_name, reflection_lm, task_type),
                    "trainset": trainset,
                    "valset": valset,
                    "adapter": adapter,
                    "evaluator": evaluator_fn,
                    "initial_prompt": prompt,
                }
            )

    math_adapter = MathAdapter(llm_client=task_lm)
    logic_adapter = QAAdapter(llm_client=task_lm)
    hard_logic_adapter = QAAdapter(llm_client=task_lm)
    hotpot_adapter = QAAdapter(llm_client=task_lm)
    mmlu_adapter = QAAdapter(llm_client=task_lm)
    bbh_adapter = QAAdapter(llm_client=task_lm)
    he_adapter = HumanEvalAdapter(llm_client=task_lm)
    code_adapter_judge = CodeAdapter(llm_client=task_lm, judge_lm=reflection_lm, task_type="code_fix")
    code_adapter_keyword = CodeAdapter(llm_client=task_lm, judge_lm=None, task_type="code_fix")

    all_strategies = ["Reflective", "FewShot-CoT", "Retrieval-v3"]
    add_dataset("GSM8K", GSM8K_TRAINSET, GSM8K_VALSET, math_adapter, math_evaluator(math_adapter), math_prompt, "math", all_strategies)
    add_dataset("Logic-Std", LOGIC_TRAINSET_EXTENDED, LOGIC_VALSET_EXTENDED, logic_adapter, strict_evaluator, logic_prompt, "qa", all_strategies)
    add_dataset("MATH-Hard", MATH_HARD_TRAINSET, MATH_HARD_VALSET, math_adapter, math_evaluator(math_adapter), hard_math_prompt, "math", all_strategies)
    add_dataset("Logic-Hard", LOGIC_HARD_TRAINSET, LOGIC_HARD_VALSET, hard_logic_adapter, strict_evaluator, hard_logic_prompt, "qa", all_strategies)
    add_dataset("Code-Fix", CODE_FIX_TRAINSET, CODE_FIX_VALSET, code_adapter_judge, code_evaluator(code_adapter_judge), code_prompt, "code", ["Reflective"])
    add_dataset("Code-Fix-Keyword", CODE_FIX_TRAINSET, CODE_FIX_VALSET, code_adapter_keyword, code_evaluator(code_adapter_keyword), code_prompt, "code", ["Reflective"])
    add_dataset("Code-Hard", CODE_HARD_TRAINSET, CODE_HARD_VALSET, code_adapter_judge, code_evaluator(code_adapter_judge), code_prompt, "code", ["Reflective"])
    add_dataset("Code-Hard-Keyword", CODE_HARD_TRAINSET, CODE_HARD_VALSET, code_adapter_keyword, code_evaluator(code_adapter_keyword), code_prompt, "code", ["Reflective"])

    hotpot = load_json_dataset("hotpotqa_data_v2.json")
    humaneval = load_json_dataset("humaneval_data_v2.json")
    mmlu = load_json_dataset("mmlu_data_v2.json")
    bbh_small = load_json_dataset("bbh_data.json")
    add_dataset("HotpotQA-v2", hotpot["train"], hotpot["val"], hotpot_adapter, hotpotqa_evaluator, hotpot_prompt, "qa", all_strategies)
    add_dataset("HumanEval-v2", humaneval["train"], humaneval["val"], he_adapter, he_adapter.evaluate, humaneval_prompt, "codegen", all_strategies)
    add_dataset("MMLU-v2", mmlu["train"], mmlu["val"], mmlu_adapter, mmlu_evaluator, mmlu_prompt, "qa", all_strategies)
    add_dataset("BBH-small", bbh_small["train"], bbh_small["val"], bbh_adapter, bbh_evaluator, bbh_prompt, "qa", all_strategies)

    if include_bbh_full:
        bbh_full = load_json_dataset("bbh_data_full.json")
        add_dataset("BBH-full", bbh_full["train"], bbh_full["val"], bbh_adapter, bbh_evaluator, bbh_prompt, "qa", all_strategies)

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
    print(f"Strategy: {type(spec['strategy']).__name__}", flush=True)
    print("=" * 72 + "\n", flush=True)

    optimizer = AgentOptimizer(
        initial_agent_config={"system_prompt": spec["initial_prompt"]},
        learning_strategy=spec["strategy"],
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

    summary = {
        "experiment": spec["experiment"],
        "dataset": spec["dataset"],
        "strategy": type(spec["strategy"]).__name__,
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
        spec
        for spec in specs
        if any(pattern in spec["experiment"].lower() or pattern in spec["dataset"].lower() for pattern in lowered)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument(
        "--output",
        default=os.path.join(EXPERIMENT_DIR, "all_improved_datasets_results.json"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-bbh-full", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args(argv)

    print("=" * 72, flush=True)
    print("Improved ICL-Agent All-Dataset Training", flush=True)
    print(f"Max iterations: {args.max_iter}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Include BBH-full: {args.include_bbh_full}", flush=True)
    print("=" * 72, flush=True)

    task_lm = DeepSeekClient(model="deepseek-chat", temperature=0.0, max_tokens=1024)
    reflection_lm = DeepSeekClient(model="deepseek-chat", temperature=0.7, max_tokens=2048)
    specs = filter_specs(build_specs(task_lm, reflection_lm, args.include_bbh_full), args.only)

    results = load_existing(args.output) if args.resume else []
    completed = {item["experiment"] for item in results}
    print(f"Planned experiments: {len(specs)}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)

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
                "strategy": type(spec["strategy"]).__name__,
                "train_size": len(spec["trainset"]),
                "val_size": len(spec["valset"]),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"\n[ERROR] {spec['experiment']}: {summary['error']}", flush=True)
        results.append(summary)
        save_results(args.output, results)
        completed.add(spec["experiment"])
        print(f"Checkpoint saved: {args.output}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Experiment':<36} {'Train':>5} {'Val':>5} {'Init':>7} {'Best':>7} {'Final':>7} {'Imp':>8} {'Iter':>4}")
    print("-" * 90, flush=True)
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
