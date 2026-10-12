"""Pydantic 请求模型 —— 与 Java 端 ChatRequest / ItemReviewRequest 一一对应"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat 请求体"""
    message: str = Field(..., description="用户自然语言输入")
    session_id: Optional[str] = Field(
        default=None,
        alias="sessionId",
        description="会话ID，首次为空，后端创建并返回"
    )

    class Config:
        populate_by_name = True


class ItemReviewRequest(BaseModel):
    """POST /review/item 请求体"""
    name: Optional[str] = Field(default=None, description="商品名称")
    price: Optional[int] = Field(default=None, description="价格（分）")
    stock: Optional[int] = Field(default=None, description="库存")
    image: Optional[str] = Field(default=None, description="商品图片")
    category_id: Optional[int] = Field(default=None, alias="categoryId")
    brand_id: Optional[int] = Field(default=None, alias="brandId")

    class Config:
        populate_by_name = True
