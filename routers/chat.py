"""AI 聊天路由 —— POST /chat, POST /chat/stream, GET /history, DELETE /history/clear"""

import json
import logging
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from models.request import ChatRequest
from models.response import success, fail, not_found
from services.ai_service import handle_chat, handle_history, handle_clear
from services.langchain.chain import chat_with_agent_stream
from services.langchain.tools import set_auth_context

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    """发送消息 — Agent 自主决策 + Tool Calling"""
    try:
        auth = request.headers.get("Authorization")
        uid = request.headers.get("user-id")

        result = await handle_chat(
            message=body.message,
            session_id=body.session_id,
            auth_header=auth,
            user_id=uid,
        )
        return success(result)
    except Exception as e:
        logger.exception("chat 接口异常")
        return fail("AI 服务暂时不可用，请稍后重试")


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """
    发送消息 — SSE 流式输出。

    前端使用 EventSource 或 fetch + ReadableStream 接收：
      event: token   → data: {"content": "您"}
      event: tool    → data: {"tool": "query_user_orders", "status": "start"}
      event: done    → data: {}
    """
    auth = request.headers.get("Authorization")
    uid = request.headers.get("user-id")
    session_id = body.session_id or ""

    async def event_stream():
        try:
            async for chunk in chat_with_agent_stream(
                message=body.message,
                session_id=session_id,
                auth_header=auth,
                user_id=uid,
            ):
                event_type = chunk["type"]
                content = chunk["content"]

                if event_type == "token":
                    yield f"event: token\ndata: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                elif event_type == "tool_start":
                    yield f"event: tool\ndata: {json.dumps({'tool': content, 'status': 'start'}, ensure_ascii=False)}\n\n"
                elif event_type == "tool_end":
                    yield f"event: tool\ndata: {json.dumps({'tool': content, 'status': 'end'}, ensure_ascii=False)}\n\n"
                elif event_type == "done":
                    yield f"event: done\ndata: {{}}\n\n"
        except Exception as e:
            logger.exception("stream 异常")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history")
async def history(sessionId: str):
    """获取对话历史"""
    try:
        data = await handle_history(sessionId)
        if data is None:
            return not_found()
        return success(data)
    except Exception as e:
        logger.exception("history 接口异常")
        return fail("获取历史记录失败")


@router.delete("/history/clear")
async def clear(sessionId: str):
    """清除对话历史"""
    try:
        await handle_clear(sessionId)
        return success()
    except Exception as e:
        logger.exception("clear 接口异常")
        return fail("清除历史记录失败")


@router.delete("/history")
async def clear_alt(sessionId: str):
    """清除对话历史（兼容路径）"""
    try:
        await handle_clear(sessionId)
        return success()
    except Exception as e:
        logger.exception("clear 接口异常")
        return fail("清除历史记录失败")
