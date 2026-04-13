"""
QA Adapter
问答适配器

适配问答任务
"""

from typing import Dict, Any, Optional, Callable
from .base_adapter import BaseAdapter


class QAAdapter(BaseAdapter):
    """
    问答任务适配器

    支持简单的问答任务
    """

    def __init__(
        self,
        llm_client: Optional[Callable] = None
    ):
        """
        初始化QA适配器

        Args:
            llm_client: LLM客户端函数
        """
        super().__init__(name="QAAdapter")
        self.llm_client = llm_client

    def execute(
        self,
        agent_config: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行问答

        Args:
            agent_config: 包含 system_prompt 的配置
            input_data: 包含 question 的输入

        Returns:
            包含 answer 的输出
        """
        system_prompt = agent_config.get("system_prompt", "")
        question = input_data.get("question", "")

        if self.llm_client is None:
            # 简单的模拟回答（用于测试）
            answer = f"[Simulated Answer to: {question}]"
        else:
            # 调用实际的LLM
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
        评估答案

        评估逻辑:
        1. 精确匹配: 1.0
        2. 数字提取后匹配: 1.0 (处理 LLM 回答中包含多余文本的情况)
        3. 部分文本匹配: 0.5
        4. 无匹配: 0.0
        """
        import re

        predicted_answer = output.get("answer", "").strip().lower()
        true_answer = ground_truth.get("answer", "").strip().lower()

        if predicted_answer == true_answer:
            return 1.0

        # 从预测答案中提取数字，与真实答案比较
        pred_numbers = re.findall(r'-?\d+\.?\d*', predicted_answer)
        true_numbers = re.findall(r'-?\d+\.?\d*', true_answer)
        if pred_numbers and true_numbers and pred_numbers[-1] == true_numbers[-1]:
            return 1.0

        # 部分匹配
        if true_answer in predicted_answer or predicted_answer in true_answer:
            return 0.5

        return 0.0


# 示例使用
if __name__ == "__main__":
    # 创建适配器
    adapter = QAAdapter()

    # 配置
    config = {
        "system_prompt": "You are a helpful assistant."
    }

    # 输入
    input_data = {
        "question": "What is 2+2?"
    }

    # 执行
    output = adapter.execute(config, input_data)
    print("Output:", output)

    # 评估
    ground_truth = {"answer": "[Simulated Answer to: What is 2+2?]"}
    score = adapter.evaluate(output, ground_truth)
    print("Score:", score)
