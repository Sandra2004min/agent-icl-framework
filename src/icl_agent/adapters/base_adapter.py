"""
Base Adapter
基础适配器

定义适配器的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAdapter(ABC):
    """
    适配器抽象基类

    适配器负责：
    1. 执行智能体
    2. 格式化输入输出
    3. 特定领域的评估
    """

    def __init__(self, name: str = "BaseAdapter"):
        self.name = name

    @abstractmethod
    def execute(
        self,
        agent_config: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行智能体

        Args:
            agent_config: 智能体配置
            input_data: 输入数据

        Returns:
            Dict[str, Any]: 输出数据
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        output: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """
        评估输出

        Args:
            output: 智能体输出
            ground_truth: 标准答案

        Returns:
            float: 分数 (0-1)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
