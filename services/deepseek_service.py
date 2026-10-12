"""DeepSeek 服务 —— 封装 LLM 调用（替代 Java DeepSeekService）"""

import logging
from typing import Optional, List, Dict
from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)

# 全局 OpenAI 兼容客户端
_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
    return _client


async def chat(
    system_prompt: str,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """
    调用 DeepSeek 生成回复。

    Args:
        system_prompt: 系统提示词
        user_message: 用户输入
        history: 对话历史 [{role, content}, ...]，最多取最近5轮

    Returns:
        AI 回复文本，失败返回 None
    """
    client = _get_client()

    messages = [{"role": "system", "content": system_prompt}]

    # 追加最近对话历史（最多 10 条 = 5轮）
    if history:
        start = max(0, len(history) - 10)
        messages.extend(history[start:])

    messages.append({"role": "user", "content": user_message})

    try:
        response = await client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=messages,
            temperature=settings.DEEPSEEK_TEMPERATURE,
            max_tokens=settings.DEEPSEEK_MAX_TOKENS,
        )
        choice = response.choices[0] if response.choices else None
        if choice and choice.message:
            return choice.message.content
        logger.warning("DeepSeek 返回格式异常: %s", response)
        return None
    except Exception as e:
        logger.error("DeepSeek API 调用失败: %s", e)
        return None
