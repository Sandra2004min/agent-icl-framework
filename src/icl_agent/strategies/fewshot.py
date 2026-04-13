"""
Few-Shot Learning Strategy
Few-Shot学习策略

从成功案例中提取示例（含推理过程）注入提示词
"""

from typing import List, Dict, Any, Optional
from .base import LearningStrategy
from ..core.context import ContextData
from ..core.knowledge import KnowledgeExtractor


class FewShotLearningStrategy(LearningStrategy):
    """
    Few-Shot学习策略（改进版）

    改进点:
    1. 不仅提取 input/output，还提取推理过程 (Chain-of-Thought)
    2. 优先选择多样性高的示例，而非仅按分数排序
    3. 同时利用失败案例生成"反面教材"
    """

    def __init__(
        self,
        num_shots: int = 5,
        include_reasoning: bool = True,
        include_negative: bool = True,
        max_negative: int = 2,
        reflection_lm: Any = None,
    ):
        """
        Args:
            num_shots: 正面示例数量
            include_reasoning: 是否包含推理过程（CoT）
            include_negative: 是否包含反面示例
            max_negative: 最多反面示例数
            reflection_lm: 可选 LLM，用于为成功案例生成推理链
        """
        super().__init__(name="FewShotLearning")
        self.num_shots = num_shots
        self.include_reasoning = include_reasoning
        self.include_negative = include_negative
        self.max_negative = max_negative
        self.reflection_lm = reflection_lm

    def learn(
        self,
        current_config: Dict[str, Any],
        contexts: List[ContextData],
        failed_contexts: List[ContextData],
        knowledge_extractor: KnowledgeExtractor
    ) -> Dict[str, Any]:
        """通过 Few-Shot + CoT 学习改进配置"""

        # 1. 选择成功案例
        successful_contexts = [
            ctx for ctx in contexts
            if not ctx.is_failure and ctx.score > 0.5
        ]

        # 2. 选择多样性高的示例
        selected_examples = self._select_diverse_examples(
            successful_contexts,
            n=self.num_shots
        )

        # 3. 格式化正面示例（含 CoT）
        positive_section = self._format_positive_examples(selected_examples)

        # 4. 格式化反面示例
        negative_section = ""
        if self.include_negative and failed_contexts:
            negative_examples = failed_contexts[:self.max_negative]
            negative_section = self._format_negative_examples(negative_examples)

        # 5. 组装增强提示
        new_config = current_config.copy()
        current_prompt = new_config.get("system_prompt", "")

        parts = [current_prompt]

        if positive_section:
            parts.append(
                "Here are examples of correct responses with step-by-step reasoning:\n\n"
                + positive_section
            )

        if negative_section:
            parts.append(
                "Here are examples of INCORRECT responses to avoid:\n\n"
                + negative_section
            )

        if positive_section or negative_section:
            parts.append(
                "Follow the reasoning patterns shown in the correct examples. "
                "Avoid the mistakes shown in the incorrect examples."
            )

        new_config["system_prompt"] = "\n\n".join(parts)

        # 6. 提取知识
        knowledge_extractor.extract_from_examples(
            examples=[ctx.to_dict() for ctx in selected_examples],
            context="few_shot_cot_learning"
        )

        return new_config

    def _select_diverse_examples(
        self,
        contexts: List[ContextData],
        n: int
    ) -> List[ContextData]:
        """选择多样性高的示例（按分数排序 + 简单去重）"""
        if not contexts:
            return []

        sorted_contexts = sorted(contexts, key=lambda x: x.score, reverse=True)

        selected = []
        seen_inputs = set()
        for ctx in sorted_contexts:
            input_key = str(ctx.input_data)[:100]
            if input_key not in seen_inputs:
                selected.append(ctx)
                seen_inputs.add(input_key)
            if len(selected) >= n:
                break

        return selected

    def _format_positive_examples(self, contexts: List[ContextData]) -> str:
        """格式化正面示例，包含推理过程"""
        if not contexts:
            return ""

        formatted = []
        for i, ctx in enumerate(contexts, 1):
            example = f"--- Example {i} ---\n"

            # 输入
            if isinstance(ctx.input_data, dict):
                q = ctx.input_data.get("question", str(ctx.input_data))
            else:
                q = str(ctx.input_data)
            example += f"Question: {q}\n"

            # 推理过程（如果有推理步骤或 LLM 可生成）
            if self.include_reasoning:
                reasoning = self._get_reasoning(ctx)
                if reasoning:
                    example += f"Reasoning: {reasoning}\n"

            # 输出
            if isinstance(ctx.output_data, dict):
                a = ctx.output_data.get("answer", str(ctx.output_data))
            else:
                a = str(ctx.output_data)
            example += f"Answer: {a}\n"

            formatted.append(example)

        return "\n".join(formatted)

    def _format_negative_examples(self, contexts: List[ContextData]) -> str:
        """格式化反面示例"""
        if not contexts:
            return ""

        formatted = []
        for i, ctx in enumerate(contexts, 1):
            example = f"--- Mistake {i} ---\n"

            if isinstance(ctx.input_data, dict):
                q = ctx.input_data.get("question", str(ctx.input_data))
            else:
                q = str(ctx.input_data)
            example += f"Question: {q}\n"

            if isinstance(ctx.output_data, dict):
                a = ctx.output_data.get("answer", str(ctx.output_data))
            else:
                a = str(ctx.output_data)
            example += f"Wrong Answer: {a}\n"

            if ctx.feedback:
                example += f"Why it's wrong: {ctx.feedback}\n"

            formatted.append(example)

        return "\n".join(formatted)

    def _get_reasoning(self, ctx: ContextData) -> str:
        """获取推理过程"""
        # 1. 尝试从轨迹元数据中获取
        if ctx.reasoning_summary:
            return ctx.reasoning_summary

        # 2. 如果有 LLM，为成功案例生成推理链
        if self.reflection_lm is not None:
            return self._generate_cot(ctx)

        # 3. 构造简单推理说明
        if isinstance(ctx.input_data, dict) and isinstance(ctx.output_data, dict):
            q = ctx.input_data.get("question", "")
            a = ctx.output_data.get("answer", "")
            if q and a:
                return f"Think step by step to arrive at the answer '{a}'."

        return ""

    def _generate_cot(self, ctx: ContextData) -> str:
        """用 LLM 为成功案例生成推理链"""
        try:
            if isinstance(ctx.input_data, dict):
                q = ctx.input_data.get("question", str(ctx.input_data))
            else:
                q = str(ctx.input_data)

            if isinstance(ctx.output_data, dict):
                a = ctx.output_data.get("answer", str(ctx.output_data))
            else:
                a = str(ctx.output_data)

            prompt = (
                f"Question: {q}\n"
                f"Correct Answer: {a}\n\n"
                "Provide a brief step-by-step reasoning (2-3 sentences) "
                "that leads to the correct answer. Be concise."
            )
            reasoning = self.reflection_lm(prompt)
            # 截断过长的推理
            if len(reasoning) > 300:
                reasoning = reasoning[:300].rsplit(".", 1)[0] + "."
            return reasoning
        except Exception:
            return ""
