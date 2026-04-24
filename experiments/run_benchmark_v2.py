"""
ICL-Agent 标准基准实验 v2 (大验证集)

  HotpotQA:  50 train + 100 val
  HumanEval: 40 train + 124 val (全量164题)
  MMLU:      50 train + 200 val (5学科)

运行: python -u experiments/run_benchmark_v2.py
"""

import sys, os, json, time, subprocess, re, string

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
from icl_agent.adapters.base_adapter import BaseAdapter
from icl_agent.utils.llm_client import DeepSeekClient


# ─── HumanEval Adapter ───
class HumanEvalAdapter(BaseAdapter):
    def __init__(self, llm_client):
        super().__init__(name="HumanEvalAdapter")
        self.llm_client = llm_client

    def execute(self, agent_config, input_data):
        prompt = input_data["prompt"]
        messages = [
            {"role": "system", "content": agent_config.get("system_prompt", "")},
            {"role": "user", "content": f"Complete the following Python function. Return ONLY the function body (the implementation lines), without repeating the function signature or any extra text.\n\n{prompt}"},
        ]
        try:
            response = self.llm_client(messages)
        except Exception as e:
            response = f"# Error: {e}"
        return {"code": response, "prompt": prompt}

    def evaluate(self, output, ground_truth):
        prompt = ground_truth["prompt"]
        test_code = ground_truth["test"]
        entry_point = ground_truth["entry_point"]
        code_body = self._extract_code(output.get("code", ""))
        full_code = prompt + code_body + "\n" + test_code + f"\ncheck({entry_point})\n"
        try:
            result = subprocess.run([sys.executable, "-c", full_code], capture_output=True, text=True, timeout=5)
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


# ─── Evaluators ───
def hotpotqa_evaluator(output, data):
    predicted = output.get("answer", "").strip()
    expected = data.get("answer", "").strip()
    def normalize(s):
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = s.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(s.split())
    if normalize(predicted) == normalize(expected):
        return 1.0
    pred_tokens = normalize(predicted).split()
    exp_tokens = normalize(expected).split()
    common = set(pred_tokens) & set(exp_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(exp_tokens) if exp_tokens else 0
    return round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0

def mmlu_evaluator(output, data):
    predicted = output.get("answer", "").strip().upper()
    expected = data.get("answer", "").strip().upper()
    match = re.search(r'\b([ABCD])\b', predicted)
    if match:
        predicted = match.group(1)
    return 1.0 if predicted == expected else 0.0


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
    task_lm = DeepSeekClient(model="deepseek-chat", temperature=0.0, max_tokens=1024)
    reflection_lm = DeepSeekClient(model="deepseek-chat", temperature=0.7, max_tokens=2048)
    all_results = []

    def make_strategies():
        return [
            ("Reflective", ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=10)),
            ("FewShot-CoT", FewShotLearningStrategy(num_shots=5, include_reasoning=True, include_negative=True, max_negative=2, reflection_lm=reflection_lm)),
            ("Retrieval-v3", RetrievalLearningStrategy(top_k=3, auto_extract=True, reflection_lm=reflection_lm)),
        ]

    # ─── HotpotQA (50 train + 100 val) ───
    print("\n" + "="*60 + "\n  PART 1: HotpotQA (val=100)\n" + "="*60, flush=True)
    with open(os.path.join(data_dir, "hotpotqa_data_v2.json"), encoding="utf-8") as f:
        d = json.load(f)
    hotpot_adapter = QAAdapter(llm_client=task_lm)
    hotpot_prompt = "You are a knowledgeable assistant. Answer the question concisely. Your answer should be as short as possible - typically a name, date, number, or a few words. Do NOT provide explanations."
    for sname, strategy in make_strategies():
        all_results.append(run_experiment(
            f"HotpotQA ({sname})", d["train"], d["val"], strategy,
            hotpot_adapter, hotpotqa_evaluator, hotpot_prompt))

    # ─── HumanEval (40 train + 124 val) ───
    print("\n" + "="*60 + "\n  PART 2: HumanEval (val=124, full)\n" + "="*60, flush=True)
    with open(os.path.join(data_dir, "humaneval_data_v2.json"), encoding="utf-8") as f:
        d = json.load(f)
    he_adapter = HumanEvalAdapter(llm_client=task_lm)
    he_prompt = "You are an expert Python programmer. Complete the function implementation based on the docstring. Return ONLY the function body code (properly indented), without repeating the function signature, adding tests, or any explanation."
    def he_eval(output, data): return he_adapter.evaluate(output, data)
    for sname, strategy in make_strategies():
        all_results.append(run_experiment(
            f"HumanEval ({sname})", d["train"], d["val"], strategy,
            he_adapter, he_eval, he_prompt))

    # ─── MMLU (50 train + 200 val) ───
    print("\n" + "="*60 + "\n  PART 3: MMLU (val=200)\n" + "="*60, flush=True)
    with open(os.path.join(data_dir, "mmlu_data_v2.json"), encoding="utf-8") as f:
        d = json.load(f)
    mmlu_adapter = QAAdapter(llm_client=task_lm)
    mmlu_prompt = "You are a knowledgeable expert. Read the question and the four options carefully. Answer with ONLY the letter of the correct option (A, B, C, or D). Do not include any explanation."
    for sname, strategy in make_strategies():
        all_results.append(run_experiment(
            f"MMLU ({sname})", d["train"], d["val"], strategy,
            mmlu_adapter, mmlu_evaluator, mmlu_prompt))

    # ─── 汇总 ───
    print("\n" + "="*80, flush=True)
    print("  FINAL RESULTS: HotpotQA + HumanEval + MMLU (v2, large val)")
    print("="*80)
    print(f"{'Experiment':<30} {'Train':>5} {'Val':>5} {'Init':>6} {'Best':>6} {'Improv':>8} {'Iter':>4} {'Time':>7}")
    print("-"*78)
    for r in all_results:
        print(f"{r['experiment']:<30} {r['train_size']:>5} {r['val_size']:>5} {r['initial_score']:>6.3f} {r['best_score']:>6.3f} {r['improvement_pct']:>7.1f}% {r['iterations']:>4} {r['elapsed_seconds']:>6.0f}s")

    out_path = os.path.join(data_dir, "benchmark_v2_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
