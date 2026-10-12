"""Pydantic 响应模型 + 统一响应封装"""

from typing import Any, Optional, List
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse


# ---------- 业务数据模型 ----------

class ChatResponseData(BaseModel):
    """POST /chat 的 data 字段"""
    reply: str
    session_id: str = Field(alias="sessionId")

    class Config:
        populate_by_name = True


class ChatMessageItem(BaseModel):
    """历史消息条目"""
    role: str       # "user" / "ai"
    text: str
    time: str       # "yyyy-MM-dd HH:mm:ss"


class ChatHistoryData(BaseModel):
    """GET /history 的 data 字段"""
    session_id: str = Field(alias="sessionId")
    messages: List[ChatMessageItem]

    class Config:
        populate_by_name = True


class ItemReviewData(BaseModel):
    """POST /review/item 的 data 字段"""
    suggestion: str   # approve / reject / review
    reason: str


# ---------- 统一响应 ----------

def success(data: Any = None) -> JSONResponse:
    """统一成功响应"""
    body: dict = {"code": 200}
    if data is not None:
        body["data"] = data
    return JSONResponse(content=body)


def fail(message: str, code: int = 500) -> JSONResponse:
    """统一失败响应"""
    return JSONResponse(content={"code": code, "message": message})


def not_found(message: str = "会话不存在或已过期") -> JSONResponse:
    """404"""
    return JSONResponse(content={"code": 404, "message": message}, status_code=404)
