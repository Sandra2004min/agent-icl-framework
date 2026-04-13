"""
Code Adapter
代码生成适配器

适配代码生成与修复任务，支持:
- 代码生成（给定描述生成代码）
- 代码修复（给定有 bug 的代码，输出修复建议/修复代码）
- 基于 LLM-as-Judge 的语义评估
"""

import re
from typing import Dict, Any, Optional, Callable
from .base_adapter import BaseAdapter


class CodeAdapter(BaseAdapter):
    """
    代码生成/修复适配器

    特性:
    - 从 LLM 输出中提取代码块
    - 支持多级评估: 精确匹配 / 关键词匹配 / LLM-as-Judge
    - 适用于 HumanEval 风格和代码修复任务
    """

    def __init__(
        self,
        llm_client: Optional[Callable] = None,
        judge_lm: Optional[Callable] = None,
        task_type: str = "code_fix",
    ):
        """
        Args:
            llm_client: 任务 LLM 客户端（生成代码）
            judge_lm: 评判 LLM 客户端（LLM-as-Judge），可选
            task_type: "code_fix" | "code_gen"
        """
        super().__init__(name="CodeAdapter")
        self.llm_client = llm_client
        self.judge_lm = judge_lm
        self.task_type = task_type

    def execute(
        self,
        agent_config: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行代码任务

        input_data 格式:
            code_fix:  {"question": "What is the bug in...?", "answer": "..."}
            code_gen:  {"question": "Write a function...", "answer": "def ..."}
        """
        system_prompt = agent_config.get("system_prompt", "")
        question = input_data.get("question", "")

        if self.llm_client is None:
            answer = "[Simulated code response]"
        else:
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
                answer = self.llm_client(messages)
            except Exception as e:
                answer = f"[Error: {str(e)}]"

        return {"answer": answer}

    def evaluate(
        self,
        output: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """
        评估代码输出

        评估策略（按优先级）:
        1. LLM-as-Judge（如果提供了 judge_lm）
        2. 关键词语义匹配
        3. 代码块提取后匹配
        """
        predicted = output.get("answer", "").strip()
        expected = ground_truth.get("answer", "").strip()

        if not predicted or not expected:
            return 0.0

        # 策略 1: LLM-as-Judge
        if self.judge_lm is not None:
            judge_score = self._llm_judge(predicted, expected, ground_truth)
            if judge_score is not None:
                return judge_score

        # 策略 2: 精确匹配
        if predicted.lower() == expected.lower():
            return 1.0

        # 策略 3: 关键词语义匹配
        keyword_score = self._keyword_match(predicted, expected)

        # 策略 4: 代码块内容匹配
        code_score = self._code_block_match(predicted, expected)

        return max(keyword_score, code_score)

    def _llm_judge(
        self,
        predicted: str,
        expected: str,
        ground_truth: Dict[str, Any]
    ) -> Optional[float]:
        """使用 LLM 作为评判者"""
        try:
            question = ground_truth.get("question", "")
            prompt = (
                "You are a code review expert. Judge whether the predicted answer "
                "correctly addresses the question.\n\n"
                f"Question: {question[:500]}\n\n"
                f"Reference Answer: {expected[:500]}\n\n"
                f"Predicted Answer: {predicted[:500]}\n\n"
                "Score the predicted answer from 0.0 to 1.0:\n"
                "- 1.0: Correct and complete\n"
                "- 0.7-0.9: Mostly correct with minor issues\n"
                "- 0.4-0.6: Partially correct\n"
                "- 0.1-0.3: Has the right idea but significant issues\n"
                "- 0.0: Completely wrong\n\n"
                "Reply with ONLY a number between 0.0 and 1.0."
            )
            response = self.judge_lm(prompt)
            # 提取分数
            numbers = re.findall(r'[01]\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)
        except Exception:
            pass
        return None

    def _keyword_match(self, predicted: str, expected: str) -> float:
        """基于关键词的语义匹配"""
        pred_lower = predicted.lower()
        exp_lower = expected.lower()

        # 提取关键词（移除停用词）
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "it", "this", "that", "and", "or", "not", "but", "should",
            "you", "your", "we", "they", "he", "she", "use", "instead",
        }

        def extract_keywords(text):
            words = re.findall(r'[a-z_][a-z0-9_]*', text.lower())
            return set(w for w in words if w not in stopwords and len(w) > 1)

        exp_keywords = extract_keywords(exp_lower)
        pred_keywords = extract_keywords(pred_lower)

        if not exp_keywords:
            return 0.0

        overlap = len(exp_keywords & pred_keywords)
        recall = overlap / len(exp_keywords)
        precision = overlap / len(pred_keywords) if pred_keywords else 0.0

        # F1 score
        if recall + precision == 0:
            return 0.0
        f1 = 2 * recall * precision / (recall + precision)

        return round(f1, 2)

    def _code_block_match(self, predicted: str, expected: str) -> float:
        """提取代码块后比较"""
        pred_code = self._extract_code_block(predicted)
        exp_code = self._extract_code_block(expected)

        if pred_code and exp_code:
            # 标准化后比较
            pred_norm = self._normalize_code(pred_code)
            exp_norm = self._normalize_code(exp_code)

            if pred_norm == exp_norm:
                return 1.0

            # 检查关键改动是否包含
            return self._keyword_match(pred_norm, exp_norm) * 0.8

        return 0.0

    def _extract_code_block(self, text: str) -> str:
        """从文本中提取代码块"""
        # 匹配 ```python ... ``` 或 ``` ... ```
        blocks = re.findall(r'```(?:python)?\n?(.*?)\n?```', text, re.DOTALL)
        if blocks:
            return blocks[0].strip()
        return ""

    def _normalize_code(self, code: str) -> str:
        """标准化代码（去除注释和多余空白）"""
        lines = []
        for line in code.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
        return "\n".join(lines)
