"""Redis 客户端 —— 与 Java 端连接同一 Redis 实例"""

import json
from typing import List, Optional
import redis.asyncio as aioredis
from config import settings

# 全局 Redis 连接
_redis: Optional[aioredis.Redis] = None

HISTORY_KEY_PREFIX = "chat:history:"
HISTORY_TTL = settings.CHAT_HISTORY_TTL_HOURS * 3600  # 秒


async def get_redis() -> aioredis.Redis:
    """获取 Redis 连接（单例）"""
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis


async def close_redis():
    """关闭 Redis 连接"""
    global _redis
    if _redis:
        await _redis.close()
        _redis = None


# ==================== 对话历史操作（与 Java AiChatService 一致） ====================

async def save_message(session_id: str, role: str, text: str, time_str: str):
    """保存消息到 Redis List"""
    r = await get_redis()
    key = HISTORY_KEY_PREFIX + session_id
    msg = json.dumps({"role": role, "text": text, "time": time_str}, ensure_ascii=False)
    await r.rpush(key, msg)
    await r.expire(key, HISTORY_TTL)


async def get_history(session_id: str) -> Optional[List[dict]]:
    """获取完整对话历史"""
    r = await get_redis()
    key = HISTORY_KEY_PREFIX + session_id
    raw_list = await r.lrange(key, 0, -1)
    if not raw_list:
        return None
    messages = []
    for raw in raw_list:
        try:
            messages.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return messages if messages else None


async def load_history_for_llm(session_id: str) -> List[dict]:
    """加载最近10条历史（转为 DeepSeek API 格式）"""
    r = await get_redis()
    key = HISTORY_KEY_PREFIX + session_id
    raw_list = await r.lrange(key, -10, -1)
    if not raw_list:
        return []
    result = []
    for raw in raw_list:
        try:
            msg = json.loads(raw)
            role = "assistant" if msg.get("role") == "ai" else msg.get("role", "user")
            result.append({"role": role, "content": msg.get("text", "")})
        except json.JSONDecodeError:
            continue
    return result


async def clear_history(session_id: str):
    """清除对话历史"""
    r = await get_redis()
    key = HISTORY_KEY_PREFIX + session_id
    await r.delete(key)
