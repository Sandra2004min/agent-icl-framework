"""
Math Adapter
数学推理适配器

适配 GSM8K 等数学推理数据集，支持数字提取和多格式答案匹配
"""

import re
from typing import Dict, Any, Optional, Callable
from .base_adapter import BaseAdapter


class MathAdapter(BaseAdapter):
    """
    数学推理适配器

    特性:
    - 支持 GSM8K 格式 (question / answer 含 #### 标记)
    - 从 LLM 输出中智能提取最终数字答案
    - 支持整数、小数、负数、分数等格式
    """

    def __init__(
        self,
        llm_client: Optional[Callable] = None,
        answer_prefix: str = "#### ",
    ):
        """
        Args:
            llm_client: LLM 客户端
            answer_prefix: GSM8K 答案前缀，用于从原始答案中提取数字
        """
        super().__init__(name="MathAdapter")
        self.llm_client = llm_client
        self.answer_prefix = answer_prefix

    def execute(
        self,
        agent_config: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行数学推理

        input_data 格式:
            {"question": "...", "answer": "..."}
        """
        system_prompt = agent_config.get("system_prompt", "")
        question = input_data.get("question", "")

        if self.llm_client is None:
            answer = f"[Simulated: 42]"
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
        评估数学答案

        支持多种格式:
        1. 精确匹配提取的数字
        2. GSM8K 格式: 从 "#### 123" 中提取答案
        3. 容忍逗号分隔的数字 (如 1,000 -> 1000)
        """
        predicted = output.get("answer", "")
        expected_raw = ground_truth.get("answer", "")

        # 提取期望的数字答案
        expected_num = self._extract_answer(expected_raw)
        # 提取预测的数字答案
        predicted_num = self._extract_final_number(predicted)

        if expected_num is None:
            # 无法解析期望答案，回退到字符串匹配
            return 1.0 if predicted.strip().lower() == expected_raw.strip().lower() else 0.0

        if predicted_num is None:
            return 0.0

        # 数值比较（容忍浮点误差）
        try:
            if abs(float(predicted_num) - float(expected_num)) < 1e-6:
                return 1.0
        except (ValueError, TypeError):
            pass

        # 字符串比较
        if predicted_num.strip() == expected_num.strip():
            return 1.0

        return 0.0

    def _extract_answer(self, text: str) -> Optional[str]:
        """从 GSM8K 格式中提取答案数字"""
        text = text.strip()

        # GSM8K 格式: "... #### 123"
        if self.answer_prefix in text:
            parts = text.split(self.answer_prefix)
            return self._normalize_number(parts[-1].strip())

        # 纯数字
        num = self._normalize_number(text)
        if num is not None:
            return num

        # 尝试提取最后一个数字
        return self._extract_last_number(text)

    def _extract_final_number(self, text: str) -> Optional[str]:
        """从 LLM 输出中提取最终答案数字"""
        text = text.strip()

        # 1. 查找 "#### xxx" 格式
        match = re.search(r'####\s*(.+)', text)
        if match:
            return self._normalize_number(match.group(1).strip())

        # 2. 查找 "answer is xxx" / "answer: xxx" 等模式
        patterns = [
            r'(?:the\s+)?answer\s*(?:is|=|:)\s*\$?([+-]?[\d,]+\.?\d*)',
            r'(?:result|total|sum)\s*(?:is|=|:)\s*\$?([+-]?[\d,]+\.?\d*)',
            r'\\boxed\{([^}]+)\}',
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return self._normalize_number(match.group(1).strip())

        # 3. 提取最后一个数字
        return self._extract_last_number(text)

    def _extract_last_number(self, text: str) -> Optional[str]:
        """提取文本中最后一个数字"""
        numbers = re.findall(r'[+-]?[\d,]+\.?\d*', text)
        if numbers:
            return self._normalize_number(numbers[-1])
        return None

    def _normalize_number(self, text: str) -> Optional[str]:
        """标准化数字字符串"""
        if text is None:
            return None
        # 移除逗号、美元符号、空格
        text = text.replace(",", "").replace("$", "").replace(" ", "").strip()
        # 验证是否为有效数字
        try:
            num = float(text)
            # 如果是整数，返回整数形式
            if num == int(num):
                return str(int(num))
            return str(num)
        except ValueError:
            return text if text else None
