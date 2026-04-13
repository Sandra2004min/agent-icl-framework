"""
Reflective Learning Strategy
反思学习策略

基于GEPA的反思机制，通过LLM分析失败案例并生成改进建议
"""

from typing import List, Dict, Any, Optional
from .base import LearningStrategy
from ..core.context import ContextData
from ..core.knowledge import KnowledgeExtractor, KnowledgeType


class ReflectiveLearningStrategy(LearningStrategy):
    """
    反思学习策略

    核心思想：
    1. 收集失败案例
    2. 构建反思提示
    3. 调用强LLM反思
    4. 提取改进建议
    5. 更新智能体配置
    """

    def __init__(
        self,
        reflection_lm: Any = None,  # LLM客户端
        max_failures: int = 10,
        reflection_prompt_template: Optional[str] = None
    ):
        """
        初始化反思学习策略

        Args:
            reflection_lm: 用于反思的LLM
            max_failures: 最多分析的失败案例数
            reflection_prompt_template: 自定义反思提示模板
        """
        super().__init__(name="ReflectiveLearning")
        self.reflection_lm = reflection_lm
        self.max_failures = max_failures
        self.reflection_prompt_template = reflection_prompt_template or self._default_prompt_template()

    def learn(
        self,
        current_config: Dict[str, Any],
        contexts: List[ContextData],
        failed_contexts: List[ContextData],
        knowledge_extractor: KnowledgeExtractor
    ) -> Dict[str, Any]:
        """
        通过反思学习改进配置

        流程：
        1. 选择失败案例
        2. 构建反思数据集
        3. 生成反思提示
        4. LLM反思生成改进
        5. 提取并应用改进
        """
        # 1. 选择失败案例（最多max_failures个）
        selected_failures = failed_contexts[:self.max_failures]

        if not selected_failures:
            # 没有失败案例，返回原配置
            return current_config.copy()

        # 2. 构建反思数据集
        reflective_dataset = self._build_reflective_dataset(selected_failures)

        # 3. 生成反思提示
        current_instruction = current_config.get("system_prompt", "")
        reflection_prompt = self._generate_reflection_prompt(
            current_instruction,
            reflective_dataset
        )

        # 4. LLM反思
        improved_instruction = self._reflect_with_llm(reflection_prompt)

        # 5. 提取知识
        knowledge = knowledge_extractor.extract_from_reflection(
            reflective_data={"failures": [ctx.to_dict() for ctx in selected_failures]},
            improved_instruction=improved_instruction
        )

        # 6. 更新配置
        new_config = current_config.copy()
        new_config["system_prompt"] = improved_instruction

        return new_config

    def _build_reflective_dataset(
        self,
        failed_contexts: List[ContextData]
    ) -> List[Dict[str, Any]]:
        """
        构建反思数据集

        格式化失败案例为结构化的反思材料
        """
        dataset = []

        for ctx in failed_contexts:
            example = {
                "Inputs": ctx.input_data,
                "Generated Outputs": ctx.output_data,
                "Feedback": ctx.feedback,
                "Score": ctx.score,
                "Error Patterns": ctx.error_patterns if ctx.error_patterns else ["No specific errors"],
            }
            dataset.append(example)

        return dataset

    def _generate_reflection_prompt(
        self,
        current_instruction: str,
        reflective_dataset: List[Dict[str, Any]]
    ) -> str:
        """
        生成反思提示

        基于当前指令和失败案例，构建反思提示
        """
        # 格式化失败案例
        formatted_examples = self._format_examples(reflective_dataset)

        # 填充模板
        prompt = self.reflection_prompt_template.format(
            current_instruction=current_instruction,
            failure_examples=formatted_examples,
            num_failures=len(reflective_dataset)
        )

        return prompt

    def _reflect_with_llm(self, prompt: str) -> str:
        """
        使用LLM进行反思

        Args:
            prompt: 反思提示

        Returns:
            str: 改进后的指令
        """
        if self.reflection_lm is None:
            # 如果没有提供LLM，返回简单的改进
            return self._simple_improvement(prompt)

        try:
            # 调用LLM
            response = self.reflection_lm(prompt)

            # 提取改进的指令（假设在代码块中）
            improved_instruction = self._extract_instruction(response)

            return improved_instruction

        except Exception as e:
            print(f"LLM反思失败: {e}")
            return prompt  # 返回原指令

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """格式化示例为Markdown文本"""
        formatted = []

        for i, example in enumerate(examples, 1):
            ex_str = f"## Example {i}\n\n"

            for key, value in example.items():
                ex_str += f"### {key}\n"

                if isinstance(value, dict):
                    for k, v in value.items():
                        ex_str += f"- **{k}**: {v}\n"
                elif isinstance(value, list):
                    for item in value:
                        ex_str += f"- {item}\n"
                else:
                    ex_str += f"{value}\n"

                ex_str += "\n"

            formatted.append(ex_str)

        return "\n".join(formatted)

    def _extract_instruction(self, llm_response: str) -> str:
        """从LLM响应中提取指令"""
        # 查找代码块
        import re

        # 尝试提取```块中的内容
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', llm_response, re.DOTALL)

        if code_blocks:
            return code_blocks[0].strip()

        # 如果没有代码块，返回整个响应
        return llm_response.strip()

    def _simple_improvement(self, original: str) -> str:
        """简单的规则基础改进（当没有LLM时）"""
        # 这是一个后备方案
        improvements = [
            "Be more precise in your responses.",
            "Double-check your calculations.",
            "Consider edge cases.",
        ]

        return original + "\n\nAdditional guidelines:\n" + "\n".join(f"- {imp}" for imp in improvements)

    def _default_prompt_template(self) -> str:
        """默认的反思提示模板"""
        return """You are an expert at improving AI agent instructions based on failure analysis.

I have an AI agent with the following instruction:

```
{current_instruction}
```

The agent was tested on {num_failures} cases and failed. Here are the details:

{failure_examples}

Your task:
1. Analyze the failure patterns
2. Identify what knowledge or guidelines are missing from the current instruction
3. Propose an improved instruction that addresses these failures

Requirements:
- Keep the core purpose of the instruction
- Add specific guidelines to avoid the observed failures
- Be concise but comprehensive
- Format the improved instruction clearly

Please provide the improved instruction within a code block (```).
"""


# 示例使用
if __name__ == "__main__":
    from ..core.context import ContextData

    # 创建示例失败上下文
    ctx1 = ContextData(
        trajectory_id="test1",
        input_data={"question": "What is 2+2?"},
        output_data={"answer": "5"},
        score=0.0,
        is_failure=True,
        feedback="Incorrect answer. Expected: 4, Got: 5"
    )

    ctx2 = ContextData(
        trajectory_id="test2",
        input_data={"question": "What is 3*3?"},
        output_data={"answer": "6"},
        score=0.0,
        is_failure=True,
        feedback="Incorrect answer. Expected: 9, Got: 6"
    )

    # 创建策略（没有LLM）
    strategy = ReflectiveLearningStrategy()

    # 当前配置
    current_config = {
        "system_prompt": "You are a math assistant."
    }

    # 学习
    from ..core.knowledge import KnowledgeExtractor

    extractor = KnowledgeExtractor()

    improved_config = strategy.learn(
        current_config=current_config,
        contexts=[ctx1, ctx2],
        failed_contexts=[ctx1, ctx2],
        knowledge_extractor=extractor
    )

    print("改进后的配置:")
    print(improved_config["system_prompt"])
