# ICL-Agent: In-Context Learning Agent Framework

**智能体上下文学习框架 - 毕业设计项目**

> 一个通用的智能体上下文学习框架，支持多种学习策略和跨领域应用

## 🌟 核心特性

- **统一的学习抽象**：将反思学习、Few-Shot学习、检索学习统一到一个框架
- **可插拔的策略**：灵活组合不同的学习策略
- **跨领域支持**：问答、代码生成、工具调用等多个领域
- **完整的工程实现**：包含轨迹捕获、知识提取、优化全流程

## 📦 安装

### 从源码安装

```bash
git clone https://github.com/yourusername/icl-agent.git
cd icl-agent
pip install -e .
```

### 使用pip安装

```bash
pip install icl-agent
```

## 🚀 快速开始

### 基础示例

```python
from icl_agent import ICLAgent
from icl_agent.strategies import ReflectiveLearningStrategy
from icl_agent.adapters import QAAdapter

# 创建智能体
agent = ICLAgent(
    system_prompt="You are a helpful assistant.",
    task_lm="openai/gpt-4o-mini"
)

# 配置学习策略
strategy = ReflectiveLearningStrategy(
    reflection_lm="openai/gpt-4o",
    num_iterations=10
)

# 创建适配器
adapter = QAAdapter()

# 准备数据
trainset = [
    {"question": "What is 2+2?", "answer": "4"},
    # ... more examples
]

# 开始学习优化
result = agent.learn(
    trainset=trainset,
    strategy=strategy,
    adapter=adapter
)

print(f"优化后的提示词: {result.best_prompt}")
print(f"性能提升: {result.improvement}%")
```

## 🏗️ 架构

```
ICL-Agent
├── Core Module          # 核心模块
│   ├── Trajectory       # 轨迹捕获
│   ├── Context          # 上下文分析
│   ├── Knowledge        # 知识提取
│   └── Optimizer        # 优化引擎
├── Learning Strategies  # 学习策略
│   ├── Reflective       # 反思学习
│   ├── Few-Shot         # 示例学习
│   └── Retrieval        # 检索学习
└── Domain Adapters      # 领域适配器
    ├── QA               # 问答
    ├── Code             # 代码生成
    └── Tool             # 工具调用
```

## 📊 实验结果

| 任务 | Baseline | ICL-Agent | 提升 |
|------|----------|-----------|------|
| 数学问答(AIME) | 46.6% | 54.8% | +8.2% |
| 代码生成(HumanEval) | 65.0% | 76.5% | +11.5% |
| 工具调用 | 72.0% | 88.0% | +16.0% |

## 🔬 实验复现

查看 `experiments/` 目录获取完整的实验代码和配置。

```bash
# 运行数学问答实验
cd experiments/math_qa
python run_experiment.py

# 运行代码生成实验
cd experiments/code_gen
python run_experiment.py
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 🙏 致谢

本项目基于以下研究工作：
- [GEPA](https://github.com/gepa-ai/gepa)
- [DSPy](https://dspy.ai/)

---

**毕业设计项目** - 2026年
