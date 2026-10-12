"""核心 AI 对话服务 —— LangChain Agent 驱动

v3.0 改造：
  - 关键词意图识别(intent_parser) → Agent 自主 Tool Calling
  - if-else 数据获取 → Agent 按需调用 Tool
  - f-string prompt → ChatPromptTemplate
  - 手动历史管理 → RunnableWithMessageHistory + RedisChatMessageHistory
  - 降级模板 → Tool 内置错误处理
"""

import logging
import uuid
import time
from typing import Optional

from services.langchain.chain import chat_with_agent
from utils.redis_client import get_history, clear_history

logger = logging.getLogger(__name__)


async def handle_chat(
    message: str,
    session_id: Optional[str],
    auth_header: Optional[str],
    user_id: Optional[str],
) -> dict:
    """
    处理用户消息（Agent 模式）。

    Agent 自主决策调用哪个 Tool：
      - query_user_orders    → 查订单
      - query_user_coupons   → 查优惠券
      - query_user_cart      → 查购物车
      - search_mall_products → 搜商品
      - query_knowledge_base → 查知识库
      - 无需任何 Tool        → 直接闲聊回复

    Returns:
        {"reply": str, "sessionId": str}
    """
    # 1. 生成或复用 sessionId
    if not session_id or not session_id.strip():
        session_id = f"ai_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

    logger.info("Agent 处理消息: session=%s, message=%s, user=%s", session_id, message, user_id)

    # 2. 使用 LangChain Agent 处理
    try:
        reply = await chat_with_agent(
            message=message,
            session_id=session_id,
            auth_header=auth_header,
            user_id=user_id,
        )
    except Exception as e:
        logger.exception("Agent 调用异常")
        reply = "抱歉，AI 服务暂时不可用，请稍后重试。"

    return {"reply": reply, "sessionId": session_id}


async def handle_history(session_id: str) -> Optional[dict]:
    """获取对话历史（兼容旧接口）"""
    messages = await get_history(session_id)
    if not messages:
        return None
    return {"sessionId": session_id, "messages": messages}


async def handle_clear(session_id: str):
    """清除对话历史"""
    await clear_history(session_id)
