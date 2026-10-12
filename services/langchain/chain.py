"""LangChain Chain 组装 —— Agent + Tools + Memory + Retriever 的完整流水线"""

import logging
import time
from typing import Optional, AsyncGenerator

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from services.langchain.chat_model import build_chat_model
from services.langchain.prompts import build_agent_prompt
from services.langchain.tools import ALL_TOOLS, set_auth_context
from services.langchain.memory import get_session_history

logger = logging.getLogger(__name__)

# 全局单例（Agent 和 LLM 是线程安全的，可以复用）
_agent_executor: Optional[AgentExecutor] = None


def _get_agent() -> AgentExecutor:
    """懒加载 AgentExecutor 单例"""
    global _agent_executor
    if _agent_executor is None:
        llm = build_chat_model(streaming=True)
        prompt = build_agent_prompt()

        agent = create_tool_calling_agent(
            llm=llm,
            tools=ALL_TOOLS,
            prompt=prompt,
        )

        _agent_executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=False,
        )
        logger.info("LangChain Agent 初始化完成，工具数: %d", len(ALL_TOOLS))

    return _agent_executor


def build_chain_with_memory():
    """
    构建带记忆的完整对话链。

    链结构：
      RunnableWithMessageHistory(
        AgentExecutor(Agent + Tools),
        get_session_history -> RedisChatMessageHistory
      )

    调用方式：
      result = await chain.ainvoke(
        {"input": "帮我查订单"},
        config={"configurable": {"session_id": "xxx", "auth_header": "...", "user_id": "2"}}
      )
    """
    agent = _get_agent()

    chain = RunnableWithMessageHistory(
        agent,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return chain


# ========== 高层 API ==========


async def chat_with_agent(
    message: str,
    session_id: str,
    auth_header: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    使用 Agent 处理用户消息（非流式）。

    Args:
        message: 用户输入
        session_id: 会话ID
        auth_header: Authorization header
        user_id: 用户ID

    Returns:
        AI 回复文本
    """
    set_auth_context(auth_header, user_id)

    chain = build_chain_with_memory()

    result = await chain.ainvoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )

    return result["output"]


async def chat_with_agent_stream(
    message: str,
    session_id: str,
    auth_header: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """
    使用 Agent 处理用户消息（SSE 流式）。

    流程：
      1. 从 Redis 加载对话历史
      2. Agent 流式处理（Tool Calling + LLM 生成）
      3. 完成后异步写回消息到 Redis

    Yields:
        {"type": "token" | "tool_start" | "tool_end" | "done" | "error", "content": str}
    """
    set_auth_context(auth_header, user_id)

    # 1. 从 Redis 加载历史
    history_mgr = await get_session_history(session_id)
    chat_history = await history_mgr.aget_messages()

    # 2. Agent 流式处理
    agent = _get_agent()
    full_reply = []

    try:
        async for event in agent.astream_events(
            {"input": message, "chat_history": chat_history},
            config={"configurable": {"session_id": session_id}},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    full_reply.append(content)
                    yield {"type": "token", "content": content}

            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                yield {"type": "tool_start", "content": tool_name}

            elif kind == "on_tool_end":
                tool_name = event.get("name", "unknown")
                yield {"type": "tool_end", "content": tool_name}

    except Exception as e:
        logger.exception("Agent 流式处理异常")
        yield {"type": "error", "content": str(e)}
        return

    # 3. 异步写回消息到 Redis
    reply_text = "".join(full_reply).strip()
    if reply_text:
        try:
            await history_mgr.aadd_messages([
                HumanMessage(content=message),
                AIMessage(content=reply_text),
            ])
        except Exception as e:
            logger.warning("保存流式对话历史失败: %s", e)

    yield {"type": "done", "content": ""}
