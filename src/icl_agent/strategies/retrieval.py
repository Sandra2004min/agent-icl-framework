"""
Retrieval Learning Strategy
检索学习策略（改进版 v3）

从知识库检索相关信息 + 主动从当前轮次提取新知识存入知识库

v3 改进 (修复灾难性退化问题):
1. 高噪声熔断: 失败率 > 70% 时跳过知识提取，避免垃圾规则污染知识库
2. 知识有效性衰减: 连续未带来改进的知识降低置信度，最终被淘汰
3. 规则精炼: LLM 提取时要求输出互不矛盾的独立规则，并逐条存储
4. 保守回退: 若无高质量知识可用，返回原始配置而非注入低质量指南
5. 前序改进保留: 知识质量过滤 + 去重 + 提示词膨胀控制 + 长度上限
"""

from typing import List, Dict, Any
from .base import LearningStrategy
from ..core.context import ContextData
from ..core.knowledge import KnowledgeExtractor, KnowledgeType


class RetrievalLearningStrategy(LearningStrategy):
    """
    检索学习策略（改进版 v3）

    v3 核心改进:
    1. 高噪声熔断机制 -- 失败率过高时停止提取，保护知识库
    2. 置信度衰减 -- 每轮未改进时对已用知识降低置信度
    3. 逐条规则存储 -- 每条规则独立评估，避免整块污染
    4. 保守回退 -- 无合格知识时返回原始 prompt
    """

    def __init__(
        self,
        top_k: int = 3,
        auto_extract: bool = True,
        reflection_lm: Any = None,
        min_confidence: float = 0.7,
        max_knowledge_items: int = 10,
        max_guidelines_chars: int = 1500,
        noise_threshold: float = 0.7,
        decay_rate: float = 0.15,
    ):
        """
        Args:
            top_k: 检索返回的知识数量
            auto_extract: 是否自动从当前上下文提取新知识
            reflection_lm: 可选 LLM，用于从案例中提炼规则
            min_confidence: 最低置信度阈值，低于此值的知识不会被检索
            max_knowledge_items: 知识库最大条目数，超出时淘汰低置信度知识
            max_guidelines_chars: guidelines 文本最大字符数
            noise_threshold: 失败率熔断阈值，超过则跳过知识提取
            decay_rate: 每轮未改进时知识置信度衰减量
        """
        super().__init__(name="RetrievalLearning")
        self.top_k = top_k
        self.auto_extract = auto_extract
        self.reflection_lm = reflection_lm
        self.min_confidence = min_confidence
        self.max_knowledge_items = max_knowledge_items
        self.max_guidelines_chars = max_guidelines_chars
        self.noise_threshold = noise_threshold
        self.decay_rate = decay_rate
        # 保存原始提示词，防止每轮累加膨胀
        self._base_prompt = None
        # 跟踪上一轮分数，用于判断知识是否有效
        self._prev_score = None
        # 跟踪上一轮使用的知识 ID
        self._last_used_knowledge_ids = []

    def learn(
        self,
        current_config: Dict[str, Any],
        contexts: List[ContextData],
        failed_contexts: List[ContextData],
        knowledge_extractor: KnowledgeExtractor
    ) -> Dict[str, Any]:
        """通过检索学习改进配置"""

        # 记录原始提示词（仅首次），后续始终基于原始提示词拼接
        if self._base_prompt is None:
            self._base_prompt = current_config.get("system_prompt", "")

        # 0. 计算当前轮分数，对上一轮使用的知识做有效性衰减
        current_score = self._compute_avg_score(contexts)
        self._decay_ineffective_knowledge(
            current_score, knowledge_extractor
        )
        self._prev_score = current_score

        # 1. 高噪声熔断: 失败率过高时跳过知识提取
        total = len(contexts)
        fail_count = len(failed_contexts)
        fail_ratio = fail_count / total if total > 0 else 0

        if self.auto_extract and fail_ratio <= self.noise_threshold:
            self._extract_knowledge_from_contexts(
                contexts, failed_contexts, knowledge_extractor, fail_ratio
            )

        # 2. 淘汰低质量知识，控制知识库规模
        self._prune_knowledge_base(knowledge_extractor)

        # 3. 从知识库检索（带置信度过滤）
        knowledge_base = knowledge_extractor.get_knowledge_base()
        qualified = knowledge_base.filter_by_confidence(self.min_confidence)
        qualified.sort(key=lambda x: x.confidence, reverse=True)
        top_knowledge = qualified[:self.top_k]

        # 4. 保守回退: 无合格知识时返回原始 prompt
        if not top_knowledge:
            new_config = current_config.copy()
            new_config["system_prompt"] = self._base_prompt
            self._last_used_knowledge_ids = []
            return new_config

        # 记录本轮使用的知识 ID
        self._last_used_knowledge_ids = [k.knowledge_id for k in top_knowledge]

        # 5. 格式化为可操作的指导建议（带长度限制）
        guidelines = self._format_as_guidelines(top_knowledge)
        if len(guidelines) > self.max_guidelines_chars:
            guidelines = guidelines[:self.max_guidelines_chars].rsplit('\n', 1)[0]

        # 6. 基于原始提示词拼接（不在上一轮结果上累加）
        new_config = current_config.copy()

        enhanced_prompt = (
            f"{self._base_prompt}\n\n"
            f"Important guidelines learned from past experience:\n\n"
            f"{guidelines}\n\n"
            f"Apply these guidelines when answering."
        )

        new_config["system_prompt"] = enhanced_prompt

        return new_config

    def _compute_avg_score(self, contexts: List[ContextData]) -> float:
        """计算当前轮次的平均分数"""
        if not contexts:
            return 0.0
        scores = [ctx.score for ctx in contexts if ctx.score is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def _decay_ineffective_knowledge(
        self,
        current_score: float,
        knowledge_extractor: KnowledgeExtractor,
    ):
        """如果上一轮注入知识后分数未改进，对这些知识做置信度衰减"""
        if self._prev_score is None or not self._last_used_knowledge_ids:
            return

        # 分数有改进，标记知识为有效（微增置信度）
        if current_score > self._prev_score:
            kb = knowledge_extractor.get_knowledge_base()
            for item in kb.knowledge_items:
                if item.knowledge_id in self._last_used_knowledge_ids:
                    item.confidence = min(1.0, item.confidence + 0.05)
            return

        # 分数未改进或下降，衰减知识置信度
        kb = knowledge_extractor.get_knowledge_base()
        for item in kb.knowledge_items:
            if item.knowledge_id in self._last_used_knowledge_ids:
                item.confidence = max(0.0, item.confidence - self.decay_rate)

    def _prune_knowledge_base(self, knowledge_extractor: KnowledgeExtractor):
        """淘汰低质量知识，控制知识库规模"""
        kb = knowledge_extractor.get_knowledge_base()
        items = kb.knowledge_items

        if len(items) <= self.max_knowledge_items:
            return

        # 按置信度排序，保留 top max_knowledge_items
        items.sort(key=lambda x: x.confidence, reverse=True)
        kept = items[:self.max_knowledge_items]

        # 重建知识库
        kb.knowledge_items = kept
        kb._index_by_type = {kt: [] for kt in KnowledgeType}
        for k in kept:
            kb._index_by_type[k.knowledge_type].append(k)

    def _is_duplicate(self, new_content: str, knowledge_extractor: KnowledgeExtractor) -> bool:
        """检查新知识是否与已有知识内容高度重复"""
        existing = knowledge_extractor.get_knowledge_base().knowledge_items
        new_words = set(new_content.lower().split())
        if not new_words:
            return True

        for item in existing:
            old_words = set(item.content.lower().split())
            if not old_words:
                continue
            overlap = len(new_words & old_words)
            union = len(new_words | old_words)
            if union > 0 and overlap / union > 0.7:
                return True

        return False

    def _extract_knowledge_from_contexts(
        self,
        contexts: List[ContextData],
        failed_contexts: List[ContextData],
        knowledge_extractor: KnowledgeExtractor,
        fail_ratio: float = 0.0,
    ):
        """从当前轮次的上下文中主动提取知识（已通过熔断检查）"""

        # 从失败案例中提取规则
        if failed_contexts:
            if self.reflection_lm is not None:
                self._extract_rules_with_llm(
                    failed_contexts, knowledge_extractor, fail_ratio
                )
            else:
                self._extract_rules_heuristic(
                    failed_contexts, knowledge_extractor, fail_ratio
                )

        # 从成功案例中提取正面示例知识（仅在高成功率时）
        successful = [ctx for ctx in contexts if not ctx.is_failure and ctx.score > 0.8]
        if successful and fail_ratio < 0.5:
            knowledge_extractor.extract_from_examples(
                examples=[ctx.to_dict() for ctx in successful[:3]],
                context="successful_execution_patterns"
            )

    def _extract_rules_with_llm(
        self,
        failed_contexts: List[ContextData],
        knowledge_extractor: KnowledgeExtractor,
        fail_ratio: float = 0.0,
    ):
        """用 LLM 从失败案例中提炼规则（逐条存储）"""
        try:
            # 构建失败摘要
            failure_summaries = []
            for ctx in failed_contexts[:5]:
                if isinstance(ctx.input_data, dict):
                    q = ctx.input_data.get("question", str(ctx.input_data))[:200]
                else:
                    q = str(ctx.input_data)[:200]
                feedback = ctx.feedback or "No feedback"
                failure_summaries.append(f"- Input: {q}\n  Feedback: {feedback}")

            prompt = (
                "Analyze these failure cases and extract 2-3 concise, "
                "non-contradictory rules that would help avoid similar failures.\n\n"
                "Failures:\n" + "\n".join(failure_summaries) + "\n\n"
                "IMPORTANT: Each rule must be independent and self-contained.\n"
                "Output EXACTLY one rule per line, starting with a number.\n"
                "Keep each rule under 50 words. Be specific and actionable.\n"
                "Do NOT output generic advice like 'be careful' or 'think step by step'."
            )

            rules_text = self.reflection_lm(prompt)

            # 根据失败率动态调整置信度
            base_confidence = 0.85
            if fail_ratio > 0.6:
                base_confidence = 0.55
            elif fail_ratio > 0.4:
                base_confidence = 0.70

            # 逐条解析并独立存储每条规则
            import uuid
            import re
            from ..core.knowledge import Knowledge

            lines = rules_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                # 跳过空行
                if not line:
                    continue
                # 去除编号前缀 (如 "1.", "1)", "- ")
                rule_text = re.sub(r'^[\d]+[.)]\s*', '', line).strip()
                rule_text = re.sub(r'^[-*]\s*', '', rule_text).strip()
                if len(rule_text) < 10:
                    continue

                # 逐条去重检查
                if self._is_duplicate(rule_text, knowledge_extractor):
                    continue

                knowledge = Knowledge(
                    knowledge_id=str(uuid.uuid4()),
                    knowledge_type=KnowledgeType.RULE,
                    content=rule_text,
                    source="llm_rule_extraction",
                    confidence=base_confidence,
                    metadata={
                        "num_failures": len(failed_contexts),
                        "fail_ratio": round(fail_ratio, 2),
                    }
                )
                knowledge_extractor.get_knowledge_base().add(knowledge)

        except Exception:
            # LLM 调用失败时回退到启发式方法
            self._extract_rules_heuristic(
                failed_contexts, knowledge_extractor, fail_ratio
            )

    def _extract_rules_heuristic(
        self,
        failed_contexts: List[ContextData],
        knowledge_extractor: KnowledgeExtractor,
        fail_ratio: float = 0.0,
    ):
        """从失败案例中启发式提取规则（不依赖 LLM）"""
        if not failed_contexts:
            return

        # 根据失败率动态调整置信度
        base_confidence = 0.65
        if fail_ratio > 0.6:
            base_confidence = 0.45
        elif fail_ratio > 0.4:
            base_confidence = 0.55

        # 收集错误模式
        error_patterns = []
        for ctx in failed_contexts:
            if ctx.error_patterns:
                error_patterns.extend(ctx.error_patterns)

        if not error_patterns:
            # 从反馈中提取通用规则
            feedbacks = [ctx.feedback for ctx in failed_contexts if ctx.feedback]
            if feedbacks:
                rule_content = "Common failure feedback:\n" + "\n".join(
                    f"- {fb[:200]}" for fb in feedbacks[:5]
                )

                # 去重检查
                if self._is_duplicate(rule_content, knowledge_extractor):
                    return

                import uuid
                from ..core.knowledge import Knowledge
                knowledge = Knowledge(
                    knowledge_id=str(uuid.uuid4()),
                    knowledge_type=KnowledgeType.RULE,
                    content=rule_content,
                    source="heuristic_rule_extraction",
                    confidence=base_confidence,
                    metadata={
                        "num_failures": len(failed_contexts),
                        "fail_ratio": round(fail_ratio, 2),
                    }
                )
                knowledge_extractor.get_knowledge_base().add(knowledge)
            return

        # 统计错误频率，生成规则
        from collections import Counter
        import uuid
        from ..core.knowledge import Knowledge

        error_counts = Counter(error_patterns)
        rules = []
        for error, count in error_counts.most_common(3):
            rules.append(f"- Avoid '{error}' (occurred {count} times)")

        rule_content = "Rules to avoid common errors:\n" + "\n".join(rules)

        # 去重检查
        if self._is_duplicate(rule_content, knowledge_extractor):
            return

        knowledge = Knowledge(
            knowledge_id=str(uuid.uuid4()),
            knowledge_type=KnowledgeType.RULE,
            content=rule_content,
            source="heuristic_rule_extraction",
            confidence=base_confidence,
            metadata={
                "error_counts": dict(error_counts),
                "fail_ratio": round(fail_ratio, 2),
            }
        )
        knowledge_extractor.get_knowledge_base().add(knowledge)

    def _format_as_guidelines(self, knowledge_items) -> str:
        """将知识条目格式化为可操作的指导建议"""
        guidelines = []
        for i, k in enumerate(knowledge_items, 1):
            # 只取内容前 500 字符，避免提示词过长
            content = k.content[:500]
            source_label = k.knowledge_type.value.capitalize()
            guidelines.append(
                f"{i}. [{source_label}, confidence={k.confidence:.1f}] {content}"
            )
        return "\n".join(guidelines)
