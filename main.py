"""
云梦 AI 助手 — FastAPI 入口
启动命令: uvicorn main:app --host 0.0.0.0 --port 8095 --reload
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from config import settings
from routers import chat, review, vector
from utils.redis_client import close_redis

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动与关闭"""
    logger.info("🚀 云梦 AI 助手启动中...")
    logger.info("   DeepSeek model: %s", settings.DEEPSEEK_MODEL)
    logger.info("   Redis: %s:%s", settings.REDIS_HOST, settings.REDIS_PORT)
    logger.info("   网关: %s", settings.GATEWAY_URL)
    yield
    logger.info("👋 正在关闭...")
    await close_redis()


app = FastAPI(
    title="云梦 AI 助手",
    description="云梦商城智能客服 Python 服务",
    version="2.0.0",
    lifespan=lifespan,
)

# 注册路由（注意网关注入时 StripPrefix=1，路径不带 /ai 前缀）
app.include_router(chat.router, tags=["AI 聊天"])
app.include_router(review.router, tags=["AI 商品审核"])
app.include_router(vector.router, tags=["向量检索"])


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "yunmeng-ai-python"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8095, reload=True)
