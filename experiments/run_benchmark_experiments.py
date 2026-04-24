"""
ICL-Agent 标准基准实验脚本

基准数据集:
  1. HotpotQA (多跳问答, 50 train + 25 val)
  2. HumanEval (代码生成, 40 train + 20 val)

策略: Reflective, FewShot-CoT, Retrieval-v3
模型: DeepSeek-chat

运行:
  cd icl-agent
  python experiments/run_benchmark_experiments.py
"""

import sys
import os
import json
import time
import tempfile
import subprocess

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
from icl_agent.adapters.base_adapter import BaseAdapter
from icl_agent.utils.llm_client import DeepSeekClient


# ============================================================
#  HumanEval 适配器 (代码生成 + 执行验证)
# ============================================================

class HumanEvalAdapter(BaseAdapter):
    """HumanEval 代码生成适配器: LLM 生成函数体, subprocess 执行测试用例"""

    def __init__(self, llm_client):
        super().__init__(name="HumanEvalAdapter")
        self.llm_client = llm_client

    def execute(self, agent_config, input_data):
        system_prompt = agent_config.get("system_prompt", "")
        prompt = input_data["prompt"]
        user_msg = (
            f"Complete the following Python function. "
            f"Return ONLY the function body (the implementation lines), "
            f"without repeating the function signature or any extra text.\n\n{prompt}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        try:
            response = self.llm_client(messages)
        except Exception as e:
            response = f"# Error: {e}"
        return {"code": response, "prompt": prompt}

    def evaluate(self, output, ground_truth):
        """执行 LLM 生成的代码 + 测试用例, pass=1.0, fail=0.0"""
        prompt = ground_truth["prompt"]
        test_code = ground_truth["test"]
        entry_point = ground_truth["entry_point"]
        llm_code = output.get("code", "")

        # 从 LLM 输出中提取代码块
        code_body = self._extract_code(llm_code)

        # 组装完整代码: 函数签名 + 生成的函数体 + 测试
        full_code = prompt + code_body + "\n" + test_code + f"\ncheck({entry_point})\n"

        # 在子进程中执行, 5秒超时
        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True, text=True, timeout=5,
            )
            return 1.0 if result.returncode == 0 else 0.0
        except (subprocess.TimeoutExpired, Exception):
            return 0.0

    def _extract_code(self, text):
        """从 LLM 输出中提取代码"""
        # 尝试提取 ```python ... ``` 块
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return "\n" + code.strip() + "\n"
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                # 去掉可能的语言标记
                lines = code.strip().split("\n")
                if lines and lines[0].strip() in ("python", "py", ""):
                    lines = lines[1:]
                return "\n" + "\n".join(lines) + "\n"
        # 直接使用 (加缩进)
        lines = text.strip().split("\n")
        # 如果没缩进, 加4空格
        indented = []
        for line in lines:
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                indented.append("    " + line)
            else:
                indented.append(line)
        return "\n" + "\n".join(indented) + "\n"


# ============================================================
#  通用实验运行器
# ============================================================

def hotpotqa_evaluator(output, data):
    """HotpotQA 评估: EM + F1 (标准 HotpotQA 评估方式)"""
    import re, string
    predicted = output.get("answer", "").strip()
    expected = data.get("answer", "").strip()

    def normalize(s):
        """标准 HotpotQA 归一化: 小写, 去冠词/标点/多余空格"""
        s = s.lower()
        # 去冠词
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        # 去标点
        s = s.translate(str.maketrans('', '', string.punctuation))
        # 合并空格
        return ' '.join(s.split())

    pred_norm = normalize(predicted)
    exp_norm = normalize(expected)

    # EM
    if pred_norm == exp_norm:
        return 1.0

    # Token-level F1
    pred_tokens = pred_norm.split()
    exp_tokens = exp_norm.split()
    common = set(pred_tokens) & set(exp_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(exp_tokens) if exp_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return round(f1, 4)


def humaneval_evaluator(output, data):
    """HumanEval 评估: 通过 HumanEvalAdapter.evaluate()"""
    # 这个由 adapter.evaluate() 处理, 这里是 optimizer 需要的包装
    adapter = HumanEvalAdapter.__instances__[-1]  # hack
    return adapter.evaluate(output, data)


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


# ============================================================
#  主函数
# ============================================================

def main():
    data_dir = os.path.dirname(__file__)

    # LLM 客户端
    task_lm = DeepSeekClient(model="deepseek-chat", temperature=0.0, max_tokens=1024)
    reflection_lm = DeepSeekClient(model="deepseek-chat", temperature=0.7, max_tokens=2048)

    all_results = []

    # ─────────────────────────────────────────
    #  Part 1: HotpotQA (多跳问答)
    # ─────────────────────────────────────────
    print("=" * 60)
    print("  Part 1: HotpotQA Benchmark")
    print("=" * 60, flush=True)

    with open(os.path.join(data_dir, "hotpotqa_data.json"), encoding="utf-8") as f:
        hotpot = json.load(f)

    hotpot_train = hotpot["train"]  # 50
    hotpot_val = hotpot["val"]      # 25

    hotpot_prompt = (
        "You are a knowledgeable assistant. Answer the question concisely. "
        "Your answer should be as short as possible - typically a name, date, number, "
        "or a few words. Do NOT provide explanations."
    )
    hotpot_adapter = QAAdapter(llm_client=task_lm)

    # HotpotQA - Reflective
    all_results.append(run_experiment(
        name="HotpotQA (Reflective)",
        trainset=hotpot_train, valset=hotpot_val,
        strategy=ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=10),
        adapter=hotpot_adapter, evaluator_fn=hotpotqa_evaluator,
        initial_prompt=hotpot_prompt,
    ))

    # HotpotQA - FewShot-CoT
    all_results.append(run_experiment(
        name="HotpotQA (FewShot-CoT)",
        trainset=hotpot_train, valset=hotpot_val,
        strategy=FewShotLearningStrategy(
            num_shots=5, include_reasoning=True,
            include_negative=True, max_negative=2,
            reflection_lm=reflection_lm,
        ),
        adapter=hotpot_adapter, evaluator_fn=hotpotqa_evaluator,
        initial_prompt=hotpot_prompt,
    ))

    # HotpotQA - Retrieval-v3
    all_results.append(run_experiment(
        name="HotpotQA (Retrieval-v3)",
        trainset=hotpot_train, valset=hotpot_val,
        strategy=RetrievalLearningStrategy(
            top_k=3, auto_extract=True, reflection_lm=reflection_lm,
        ),
        adapter=hotpot_adapter, evaluator_fn=hotpotqa_evaluator,
        initial_prompt=hotpot_prompt,
    ))

    # ─────────────────────────────────────────
    #  Part 2: HumanEval (代码生成)
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Part 2: HumanEval Benchmark")
    print("=" * 60, flush=True)

    with open(os.path.join(data_dir, "humaneval_data.json"), encoding="utf-8") as f:
        humaneval = json.load(f)

    he_train = humaneval["train"]  # 40
    he_val = humaneval["val"]      # 20

    he_prompt = (
        "You are an expert Python programmer. "
        "Complete the function implementation based on the docstring. "
        "Return ONLY the function body code (properly indented), "
        "without repeating the function signature, adding tests, or any explanation."
    )
    he_adapter = HumanEvalAdapter(llm_client=task_lm)

    def he_evaluator(output, data):
        return he_adapter.evaluate(output, data)

    # HumanEval - Reflective
    all_results.append(run_experiment(
        name="HumanEval (Reflective)",
        trainset=he_train, valset=he_val,
        strategy=ReflectiveLearningStrategy(reflection_lm=reflection_lm, max_failures=10),
        adapter=he_adapter, evaluator_fn=he_evaluator,
        initial_prompt=he_prompt,
    ))

    # HumanEval - FewShot-CoT
    all_results.append(run_experiment(
        name="HumanEval (FewShot-CoT)",
        trainset=he_train, valset=he_val,
        strategy=FewShotLearningStrategy(
            num_shots=3, include_reasoning=True,
            include_negative=True, max_negative=2,
            reflection_lm=reflection_lm,
        ),
        adapter=he_adapter, evaluator_fn=he_evaluator,
        initial_prompt=he_prompt,
    ))

    # HumanEval - Retrieval-v3
    all_results.append(run_experiment(
        name="HumanEval (Retrieval-v3)",
        trainset=he_train, valset=he_val,
        strategy=RetrievalLearningStrategy(
            top_k=3, auto_extract=True, reflection_lm=reflection_lm,
        ),
        adapter=he_adapter, evaluator_fn=he_evaluator,
        initial_prompt=he_prompt,
    ))

    # ─────────────────────────────────────────
    #  汇总
    # ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL RESULTS - Standard Benchmark Experiments")
    print("=" * 80, flush=True)

    header = f"{'Experiment':<35} {'Init':>6} {'Best':>6} {'Final':>6} {'Improv':>8} {'Iter':>5} {'Time':>7}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['experiment']:<35} "
            f"{r['initial_score']:>6.3f} {r['best_score']:>6.3f} {r['final_score']:>6.3f} "
            f"{r['improvement_pct']:>7.1f}% {r['iterations']:>4} {r['elapsed_seconds']:>6.0f}s"
        )

    # 保存
    out_path = os.path.join(data_dir, "benchmark_experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
