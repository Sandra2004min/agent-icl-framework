"""
Core module for ICL-Agent
包含轨迹捕获、上下文分析、知识提取和优化器
"""

from .trajectory import Trajectory, TrajectoryCapture
from .context import ContextAnalyzer, ContextData
from .knowledge import KnowledgeExtractor, Knowledge
from .optimizer import AgentOptimizer, OptimizationResult

__all__ = [
    "Trajectory",
    "TrajectoryCapture",
    "ContextAnalyzer",
    "ContextData",
    "KnowledgeExtractor",
    "Knowledge",
    "AgentOptimizer",
    "OptimizationResult",
]
