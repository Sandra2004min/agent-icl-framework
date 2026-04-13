"""
LLM Client - DeepSeek API 客户端

通过 OpenAI 兼容接口调用 DeepSeek 模型
"""

import os
from typing import List, Dict, Optional
import httpx
from openai import OpenAI


class DeepSeekClient:
    """
    DeepSeek API 客户端

    使用 OpenAI 兼容接口调用 DeepSeek 模型
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "未提供 API Key。请设置 DEEPSEEK_API_KEY 环境变量或传入 api_key 参数"
            )

        # 处理系统代理：将 https:// 协议的代理改为 http:// 以避免 SSL 握手问题
        http_client = None
        try:
            import urllib.request
            proxies = urllib.request.getproxies()
            if proxies.get("https", "").startswith("https://"):
                proxy_url = proxies["https"].replace("https://", "http://", 1)
                http_client = httpx.Client(proxy=proxy_url)
        except Exception:
            pass

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            http_client=http_client,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages_or_prompt) -> str:
        """
        调用 LLM

        支持两种输入格式:
        - str: 单条提示，自动包装为 user message
        - List[Dict]: OpenAI 格式的 messages 列表
        """
        if isinstance(messages_or_prompt, str):
            messages = [{"role": "user", "content": messages_or_prompt}]
        else:
            messages = messages_or_prompt

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content


def create_llm_client(
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> DeepSeekClient:
    """创建 LLM 客户端的便捷函数"""
    return DeepSeekClient(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
