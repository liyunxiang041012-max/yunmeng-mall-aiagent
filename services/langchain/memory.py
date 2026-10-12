"""LangChain Memory — 基于 Redis 的对话历史管理"""

import json
import logging
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from utils.redis_client import get_redis

logger = logging.getLogger(__name__)

HISTORY_KEY_PREFIX = "chat:history:"


class RedisChatMessageHistory(BaseChatMessageHistory):
    """
    LangChain BaseChatMessageHistory 的 Redis 实现。
    与现有 redis_client.py 中的 chat:history:{sessionId} 结构兼容。
    """

    def __init__(self, session_id: str, ttl: int = 86400):
        self.session_id = session_id
        self.key = HISTORY_KEY_PREFIX + session_id
        self.ttl = ttl

    @property
    def messages(self) -> List[BaseMessage]:
        """同步读取消息（兼容接口），本项目用异步版本"""
        return []

    async def aget_messages(self) -> List[BaseMessage]:
        """异步读取对话历史"""
        r = await get_redis()
        raw_list = await r.lrange(self.key, 0, -1)
        if not raw_list:
            return []

        messages = []
        for raw in raw_list:
            try:
                msg = json.loads(raw)
                role = msg.get("role", "user")
                text = msg.get("text", "")
                if role == "ai":
                    messages.append(AIMessage(content=text))
                else:
                    messages.append(HumanMessage(content=text))
            except (json.JSONDecodeError, Exception):
                continue
        return messages

    async def aadd_messages(self, messages: List[BaseMessage]) -> None:
        """异步追加消息到 Redis"""
        r = await get_redis()
        for msg in messages:
            role = "ai" if isinstance(msg, AIMessage) else "user"
            data = json.dumps({
                "role": role,
                "text": msg.content,
                "time": _now(),
            }, ensure_ascii=False)
            await r.rpush(self.key, data)
        await r.expire(self.key, self.ttl)

    # 同步兼容方法（LangChain 调用 async 时会走这些）
    def add_messages(self, messages: List[BaseMessage]) -> None:
        """同步添加（降级为不存储）"""
        logger.warning("同步 add_messages 不被支持，请使用 aadd_messages")

    async def aclear(self) -> None:
        """清空对话历史"""
        r = await get_redis()
        await r.delete(self.key)

    def clear(self) -> None:
        """同步清空（兼容 LangChain 抽象接口）"""
        logger.warning("同步 clear 不被支持，请使用 aclear")


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """
    LangChain RunnableWithMessageHistory 所需的工厂函数。
    注意：此函数必须是同步的！RunnableWithMessageHistory 不支持异步工厂。
    RedisChatMessageHistory 的 __init__ 只存 session_id，
    实际的 Redis 操作在 aget_messages / aadd_messages 中异步执行。
    """
    return RedisChatMessageHistory(session_id=session_id)
