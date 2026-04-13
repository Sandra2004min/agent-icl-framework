"""
ICL-Agent: In-Context Learning Agent Framework
智能体上下文学习框架
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .core.trajectory import Trajectory, TrajectoryCapture
from .core.context import ContextAnalyzer, ContextData
from .core.knowledge import KnowledgeExtractor, Knowledge
from .core.optimizer import AgentOptimizer, OptimizationResult
from .strategies.reflective import ReflectiveLearningStrategy
from .strategies.fewshot import FewShotLearningStrategy
from .strategies.retrieval import RetrievalLearningStrategy
from .adapters.base_adapter import BaseAdapter
from .adapters.qa_adapter import QAAdapter
from .adapters.math_adapter import MathAdapter
from .adapters.code_adapter import CodeAdapter
from .utils.llm_client import DeepSeekClient, create_llm_client

__all__ = [
    # Core
    "Trajectory",
    "TrajectoryCapture",
    "ContextAnalyzer",
    "ContextData",
    "KnowledgeExtractor",
    "Knowledge",
    "AgentOptimizer",
    "OptimizationResult",
    # Strategies
    "ReflectiveLearningStrategy",
    "FewShotLearningStrategy",
    "RetrievalLearningStrategy",
    # Adapters
    "BaseAdapter",
    "QAAdapter",
    "MathAdapter",
    "CodeAdapter",
    # Utils
    "DeepSeekClient",
    "create_llm_client",
]
