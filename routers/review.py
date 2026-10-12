"""AI 商品审核路由 —— POST /review/item"""

import logging
from fastapi import APIRouter

from models.request import ItemReviewRequest
from models.response import success, fail
from services.review_service import ai_review

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/review/item")
async def review_item(body: ItemReviewRequest):
    """AI 审核商品（管理端）"""
    try:
        result = await ai_review(
            name=body.name,
            price=body.price,
            stock=body.stock,
        )
        return success(result)
    except Exception as e:
        logger.exception("review 接口异常")
        return fail("AI 审核服务暂不可用")
