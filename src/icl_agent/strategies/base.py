"""
Base Learning Strategy
学习策略基类

定义所有学习策略的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..core.context import ContextData
from ..core.knowledge import KnowledgeExtractor


class LearningStrategy(ABC):
    """
    学习策略抽象基类

    所有具体的学习策略都应该继承这个类并实现learn方法
    """

    def __init__(self, name: str = "BaseStrategy"):
        """
        初始化学习策略

        Args:
            name: 策略名称
        """
        self.name = name

    @abstractmethod
    def learn(
        self,
        current_config: Dict[str, Any],
        contexts: List[ContextData],
        failed_contexts: List[ContextData],
        knowledge_extractor: KnowledgeExtractor
    ) -> Dict[str, Any]:
        """
        从上下文中学习并生成改进的配置

        Args:
            current_config: 当前智能体配置
            contexts: 所有上下文数据
            failed_contexts: 失败的上下文数据
            knowledge_extractor: 知识提取器

        Returns:
            Dict[str, Any]: 改进后的智能体配置
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
