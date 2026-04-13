"""
Adapters Module
适配器模块

提供不同领域的适配器
"""

from .base_adapter import BaseAdapter
from .qa_adapter import QAAdapter
from .math_adapter import MathAdapter
from .code_adapter import CodeAdapter

__all__ = [
    "BaseAdapter",
    "QAAdapter",
    "MathAdapter",
    "CodeAdapter",
]
