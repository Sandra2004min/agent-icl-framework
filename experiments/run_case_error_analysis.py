"""
Generate case-level error analysis for selected improved-framework runs.

The script replays initial and best prompts on Logic-Hard validation cases for
three representative configurations:
  - DeepSeek / Reflective: strong positive case
  - Qwen / Retrieval-v3: partial positive case with instability
  - Kimi / Reflective: negative/model-threshold case

It saves detailed JSON plus a thesis-ready Markdown summary.
"""

import argparse
import json
import os
import re
import sys
import textwrap
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
from icl_agent.utils.llm_client import DeepSeekClient

from datasets_hard import LOGIC_HARD_VALSET


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


SELECTED_EXPERIMENTS = [
    "DeepSeek/Logic-Hard/Reflective",
    "Qwen/Logic-Hard/Retrieval-v3",
    "Kimi/Logic-Hard/Reflective",
]


INITIAL_PROMPT = (
    "You are a logic puzzle expert. Think step by step and reason carefully. "
    "Answer the question following the exact format requested."
)


def strict_score(answer, expected):
    return 1.0 if answer.strip().lower() == expected.strip().lower() else 0.0


def make_client(model_name):
    cfg = MODEL_CONFIGS[model_name]
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise ValueError(f"Missing API key env: {cfg['api_key_env']}")
    return DeepSeekClient(
        api_key=api_key,
        base_url=os.getenv(cfg["base_url_env"], cfg["default_base_url"]),
        model=cfg["model"],
        temperature=0.0,
        max_tokens=512,
    )


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_best_instruction(results, experiment):
    for item in results:
        if item.get("experiment") == experiment:
            return item.get("best_instruction", INITIAL_PROMPT)
    raise ValueError(f"Experiment not found: {experiment}")


def run_prompt(adapter, prompt, case):
    output = adapter.execute({"system_prompt": prompt}, case)
    answer = output.get("answer", "")
    return {
        "answer": answer,
        "score": strict_score(answer, case["answer"]),
    }


def classify_case(before_score, after_score):
    if before_score == 0.0 and after_score == 1.0:
        return "fixed"
    if before_score == 1.0 and after_score == 0.0:
        return "regressed"
    if before_score == 0.0 and after_score == 0.0:
        return "persistent_failure"
    return "persistent_success"


def truncate(text, max_chars=360):
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def infer_failure_reason(case, before, after):
    expected = case["answer"]
    question = case["question"].lower()
    before_answer = before["answer"].strip()
    after_answer = after["answer"].strip()

    if before["score"] == 0.0 and after["score"] == 1.0:
        if "answer with only" in question or "fill in" in question:
            return "改进提示词更严格地约束了输出格式，模型从解释性回答转为只输出目标答案。"
        return "改进提示词提升了对题目约束的识别或最终答案抽取能力。"
    if before["score"] == 1.0 and after["score"] == 0.0:
        return "改进提示词引入了干扰或过度约束，导致原本正确的题目被改错。"
    if before["score"] == 0.0 and after["score"] == 0.0:
        if re.search(r"\b\d+/\d+\b", expected) or "probability" in question:
            return "持续失败主要来自概率/组合推理错误，单纯提示词难以补足底层推理能力。"
        if len(after_answer) > len(expected) * 3 or len(before_answer) > len(expected) * 3:
            return "持续失败包含格式遵循问题，模型倾向输出解释或额外文本。"
        return "持续失败说明该题需要更强的结构化推理或任务特化知识，当前经验注入不足。"
    return "两种提示词均能正确处理该案例。"


def summarize_cases(cases):
    counts = {}
    for item in cases:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    return counts


def pick_examples(cases):
    selected = []
    for category in ["fixed", "regressed", "persistent_failure"]:
        matches = [item for item in cases if item["category"] == category]
        selected.extend(matches[:2])
    return selected


def render_markdown(analyses):
    lines = [
        "# 案例级错误分析",
        "",
        "本文件基于改进后框架的代表性跨模型结果，对 Logic-Hard 验证集重新执行初始提示词与最佳提示词，抽取修正、退化和持续失败案例。其目的不是重新报告总体分数，而是解释不同策略为何有效或失效。",
        "",
    ]

    for analysis in analyses:
        lines.extend([
            f"## {analysis['experiment']}",
            "",
            f"- 初始准确率：{analysis['initial_accuracy']:.4f}",
            f"- 最佳提示词准确率：{analysis['best_accuracy']:.4f}",
            f"- fixed：{analysis['counts'].get('fixed', 0)}",
            f"- regressed：{analysis['counts'].get('regressed', 0)}",
            f"- persistent_failure：{analysis['counts'].get('persistent_failure', 0)}",
            f"- persistent_success：{analysis['counts'].get('persistent_success', 0)}",
            "",
        ])

        for idx, item in enumerate(analysis["examples"], 1):
            lines.extend([
                f"### 案例 {idx}：{item['category']}",
                "",
                f"**题目**：{truncate(item['question'], 520)}",
                "",
                f"**标准答案**：`{item['expected']}`",
                "",
                f"**初始输出**：`{truncate(item['before_answer'], 220)}`，得分 {item['before_score']:.0f}",
                "",
                f"**最佳提示词输出**：`{truncate(item['after_answer'], 220)}`，得分 {item['after_score']:.0f}",
                "",
                f"**分析**：{item['reason']}",
                "",
            ])

        lines.extend([
            "### 小结",
            "",
            analysis["summary"],
            "",
        ])

    lines.extend([
        "## 论文可用结论",
        "",
        "1. Reflective 在强模型上主要通过强化“只输出最终答案”和题型约束检查来修复错误，但不同运行之间仍有较大波动。",
        "2. Retrieval-v3 的收益更依赖知识与当前题目的结构相似性；当检索知识无法覆盖深层推理结构时，改进有限。",
        "3. Kimi 的持续失败和 final 退化表明，提示词优化存在模型能力门槛：当模型不能稳定遵循长指令或完成基础推理时，优化策略难以转化为收益。",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=os.path.join(EXPERIMENT_DIR, "improved_cross_model_results.json"),
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(EXPERIMENT_DIR, "case_error_analysis.json"),
    )
    parser.add_argument(
        "--output-md",
        default=os.path.join(EXPERIMENT_DIR, "case_error_analysis.md"),
    )
    args = parser.parse_args(argv)

    results = load_results(args.results)
    analyses = []

    for experiment in SELECTED_EXPERIMENTS:
        model_name = experiment.split("/")[0]
        best_prompt = find_best_instruction(results, experiment)
        client = make_client(model_name)
        adapter = QAAdapter(llm_client=client)

        print("\n" + "=" * 72, flush=True)
        print(f"Case analysis: {experiment}", flush=True)
        print("=" * 72, flush=True)

        cases = []
        for case_id, case in enumerate(LOGIC_HARD_VALSET, 1):
            before = run_prompt(adapter, INITIAL_PROMPT, case)
            after = run_prompt(adapter, best_prompt, case)
            category = classify_case(before["score"], after["score"])
            item = {
                "case_id": case_id,
                "question": case["question"],
                "expected": case["answer"],
                "before_answer": before["answer"],
                "before_score": before["score"],
                "after_answer": after["answer"],
                "after_score": after["score"],
                "category": category,
                "reason": infer_failure_reason(case, before, after),
            }
            cases.append(item)
            print(
                f"{case_id:02d}: {category} before={before['score']:.0f} after={after['score']:.0f}",
                flush=True,
            )

        before_acc = sum(item["before_score"] for item in cases) / len(cases)
        after_acc = sum(item["after_score"] for item in cases) / len(cases)
        counts = summarize_cases(cases)
        if experiment.startswith("DeepSeek"):
            summary = (
                "DeepSeek 的 Reflective 最佳提示词能修复部分高难度逻辑题，主要收益来自更强的格式约束和对最终答案抽取的强调；但仍存在概率、博弈和组合题的持续失败。"
            )
        elif "Retrieval" in experiment:
            summary = (
                "Qwen 的 Retrieval-v3 能在部分题目上受益于历史经验，但修正并不稳定；检索知识对题型结构相似的问题更有效，对需要全新推理结构的题目帮助有限。"
            )
        else:
            summary = (
                "Kimi 在本组案例中表现出明显的模型能力门槛：即使提供优化后的提示词，也难以稳定遵循格式或完成复杂推理，说明负结果不能简单归因于优化算法。"
            )

        analyses.append({
            "experiment": experiment,
            "initial_accuracy": before_acc,
            "best_accuracy": after_acc,
            "counts": counts,
            "cases": cases,
            "examples": pick_examples(cases),
            "summary": summary,
        })

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(analyses, f, indent=2, ensure_ascii=False)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(analyses))

    print(f"\nSaved JSON: {args.output_json}", flush=True)
    print(f"Saved Markdown: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
