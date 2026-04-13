"""
Knowledge Module - 知识提取模块

从上下文中提取可学习的知识：
- 从反思中学习
- 从示例中学习
- 从检索中学习
- 知识表示和存储
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class KnowledgeType(Enum):
    """知识类型枚举"""
    REFLECTION = "reflection"  # 反思型知识
    EXAMPLE = "example"        # 示例型知识
    RETRIEVAL = "retrieval"    # 检索型知识
    RULE = "rule"              # 规则型知识


@dataclass
class Knowledge:
    """
    知识单元

    表示从执行经验中提取的一条知识
    """

    knowledge_id: str
    knowledge_type: KnowledgeType
    content: str
    source: str  # 知识来源
    confidence: float = 1.0  # 置信度 (0-1)
    usage_count: int = 0  # 使用次数
    success_rate: float = 0.0  # 成功率
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "knowledge_id": self.knowledge_id,
            "knowledge_type": self.knowledge_type.value,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "metadata": self.metadata,
        }

    def update_usage(self, success: bool):
        """更新使用统计"""
        self.usage_count += 1
        # 更新成功率（使用移动平均）
        self.success_rate = (
            (self.success_rate * (self.usage_count - 1) + (1.0 if success else 0.0))
            / self.usage_count
        )


class KnowledgeBase:
    """
    知识库

    存储和管理提取的知识
    """

    def __init__(self):
        self.knowledge_items: List[Knowledge] = []
        self._index_by_type: Dict[KnowledgeType, List[Knowledge]] = {
            kt: [] for kt in KnowledgeType
        }

    def add(self, knowledge: Knowledge):
        """添加知识"""
        self.knowledge_items.append(knowledge)
        self._index_by_type[knowledge.knowledge_type].append(knowledge)

    def get_by_type(self, knowledge_type: KnowledgeType) -> List[Knowledge]:
        """按类型获取知识"""
        return self._index_by_type[knowledge_type]

    def get_top_k(self, k: int = 5, by: str = "confidence") -> List[Knowledge]:
        """
        获取Top-K知识

        Args:
            k: 数量
            by: 排序依据 ("confidence", "success_rate", "usage_count")

        Returns:
            List[Knowledge]: 排序后的知识列表
        """
        if by == "confidence":
            sorted_items = sorted(self.knowledge_items, key=lambda x: x.confidence, reverse=True)
        elif by == "success_rate":
            sorted_items = sorted(self.knowledge_items, key=lambda x: x.success_rate, reverse=True)
        elif by == "usage_count":
            sorted_items = sorted(self.knowledge_items, key=lambda x: x.usage_count, reverse=True)
        else:
            sorted_items = self.knowledge_items

        return sorted_items[:k]

    def filter_by_confidence(self, min_confidence: float = 0.5) -> List[Knowledge]:
        """筛选高置信度知识"""
        return [k for k in self.knowledge_items if k.confidence >= min_confidence]

    def save_to_file(self, filepath: str):
        """保存到文件"""
        data = [k.to_dict() for k in self.knowledge_items]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> "KnowledgeBase":
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        kb = cls()
        for item in data:
            item["knowledge_type"] = KnowledgeType(item["knowledge_type"])
            kb.add(Knowledge(**item))

        return kb


class KnowledgeExtractor:
    """
    知识提取器

    从不同来源提取知识
    """

    def __init__(self):
        self.knowledge_base = KnowledgeBase()

    def extract_from_reflection(
        self,
        reflective_data: Dict[str, Any],
        improved_instruction: str
    ) -> Knowledge:
        """
        从反思数据中提取知识

        Args:
            reflective_data: 反思数据
            improved_instruction: 改进后的指令

        Returns:
            Knowledge: 提取的知识
        """
        import uuid

        # 分析反思数据，提取关键洞察
        insights = self._analyze_reflection(reflective_data)

        knowledge = Knowledge(
            knowledge_id=str(uuid.uuid4()),
            knowledge_type=KnowledgeType.REFLECTION,
            content=improved_instruction,
            source="reflection",
            confidence=0.8,  # 初始置信度
            metadata={
                "insights": insights,
                "num_failures_analyzed": len(reflective_data.get("failures", []))
            }
        )

        self.knowledge_base.add(knowledge)
        return knowledge

    def extract_from_examples(
        self,
        examples: List[Dict[str, Any]],
        context: str = ""
    ) -> Knowledge:
        """
        从示例中提取知识

        Args:
            examples: 示例列表
            context: 上下文信息

        Returns:
            Knowledge: 提取的知识
        """
        import uuid

        # 格式化示例为知识
        formatted_examples = self._format_examples(examples)

        knowledge = Knowledge(
            knowledge_id=str(uuid.uuid4()),
            knowledge_type=KnowledgeType.EXAMPLE,
            content=formatted_examples,
            source="few_shot_examples",
            confidence=0.9,
            metadata={
                "num_examples": len(examples),
                "context": context
            }
        )

        self.knowledge_base.add(knowledge)
        return knowledge

    def extract_from_retrieval(
        self,
        retrieved_docs: List[str],
        query: str
    ) -> Knowledge:
        """
        从检索结果中提取知识

        Args:
            retrieved_docs: 检索到的文档列表
            query: 查询

        Returns:
            Knowledge: 提取的知识
        """
        import uuid

        # 整合检索结果
        combined_content = self._combine_retrieval_results(retrieved_docs)

        knowledge = Knowledge(
            knowledge_id=str(uuid.uuid4()),
            knowledge_type=KnowledgeType.RETRIEVAL,
            content=combined_content,
            source="retrieval",
            confidence=0.7,
            metadata={
                "num_docs": len(retrieved_docs),
                "query": query
            }
        )

        self.knowledge_base.add(knowledge)
        return knowledge

    def extract_rules(
        self,
        patterns: Dict[str, Any]
    ) -> List[Knowledge]:
        """
        提取规则型知识

        Args:
            patterns: 发现的模式

        Returns:
            List[Knowledge]: 规则列表
        """
        import uuid

        rules = []

        for pattern_name, pattern_data in patterns.items():
            rule_content = self._formulate_rule(pattern_name, pattern_data)

            knowledge = Knowledge(
                knowledge_id=str(uuid.uuid4()),
                knowledge_type=KnowledgeType.RULE,
                content=rule_content,
                source="pattern_analysis",
                confidence=pattern_data.get("confidence", 0.7),
                metadata={
                    "pattern_name": pattern_name,
                    "pattern_data": pattern_data
                }
            )

            rules.append(knowledge)
            self.knowledge_base.add(knowledge)

        return rules

    def get_knowledge_base(self) -> KnowledgeBase:
        """获取知识库"""
        return self.knowledge_base

    # ===== 私有辅助方法 =====

    def _analyze_reflection(self, reflective_data: Dict[str, Any]) -> List[str]:
        """分析反思数据，提取关键洞察"""
        insights = []

        # 从失败案例中提取模式
        if "failures" in reflective_data:
            failures = reflective_data["failures"]
            if failures:
                insights.append(f"Analyzed {len(failures)} failure cases")

                # 提取常见错误类型
                error_types = [f.get("error_type", "unknown") for f in failures]
                from collections import Counter
                common_errors = Counter(error_types).most_common(3)
                if common_errors:
                    insights.append(f"Common errors: {', '.join(e[0] for e in common_errors)}")

        return insights

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """格式化示例为文本"""
        formatted = []

        for i, example in enumerate(examples, 1):
            ex_str = f"Example {i}:\n"
            for key, value in example.items():
                ex_str += f"  {key}: {value}\n"
            formatted.append(ex_str)

        return "\n".join(formatted)

    def _combine_retrieval_results(self, docs: List[str]) -> str:
        """整合检索结果"""
        # 简单拼接，实际应用中可能需要更复杂的整合策略
        return "\n\n---\n\n".join(docs[:5])  # 只取前5个文档

    def _formulate_rule(self, pattern_name: str, pattern_data: Dict[str, Any]) -> str:
        """根据模式制定规则"""
        # 简单实现
        return f"Rule from pattern '{pattern_name}': {pattern_data.get('description', 'No description')}"


# 示例使用
if __name__ == "__main__":
    # 创建知识提取器
    extractor = KnowledgeExtractor()

    # 从反思中提取知识
    reflective_data = {
        "failures": [
            {"error_type": "MathError", "message": "Wrong calculation"},
            {"error_type": "MathError", "message": "Another wrong calc"},
        ]
    }
    improved_instruction = "Always double-check arithmetic calculations"

    k1 = extractor.extract_from_reflection(reflective_data, improved_instruction)
    print(f"Reflection Knowledge: {k1.content}")
    print(f"  Confidence: {k1.confidence}")
    print(f"  Insights: {k1.metadata['insights']}")

    # 从示例中提取知识
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "3+3", "output": "6"},
    ]
    k2 = extractor.extract_from_examples(examples, context="math problems")
    print(f"\nExample Knowledge:")
    print(k2.content)

    # 获取知识库
    kb = extractor.get_knowledge_base()
    print(f"\nKnowledge Base Size: {len(kb.knowledge_items)}")

    # 获取最高置信度的知识
    top_knowledge = kb.get_top_k(k=2, by="confidence")
    print(f"\nTop-2 Knowledge by Confidence:")
    for k in top_knowledge:
        print(f"  - {k.knowledge_type.value}: {k.confidence}")
