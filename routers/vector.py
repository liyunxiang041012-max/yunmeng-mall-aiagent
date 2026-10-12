"""向量索引管理路由 — POST /vector/index, DELETE /vector/index"""

import logging
from fastapi import APIRouter

from models.response import success, fail
from utils.http_client import fetch_all_items
from utils.vector_store import index_items, clear_index, is_indexed, index_knowledge_base, is_knowledge_indexed
from data.knowledge import KNOWLEDGE_ENTRIES

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/vector/index")
async def build_index():
    """
    构建商品向量索引。
    从 Java 服务拉取全量商品 → 向量化 → 存入 Redis + 内存缓存。
    建议：商品数据变更后手动调用，或配合定时任务使用。
    """
    try:
        # 先清旧索引
        await clear_index()

        # 拉取全量商品
        logger.info("正在从 Java 服务拉取全量商品...")
        items = await fetch_all_items(page_size=100)
        if not items:
            return fail("未能获取到商品数据，请检查 Java 服务是否正常", code=503)

        logger.info("拉取到 %d 件商品，开始构建向量索引...", len(items))
        count = await index_items(items)

        return success({
            "total": len(items),
            "indexed": count,
            "message": f"成功索引 {count}/{len(items)} 件商品" if count > 0 else "索引构建失败，请检查日志",
        })
    except Exception as e:
        logger.exception("构建向量索引失败")
        return fail(f"索引构建失败: {str(e)}")


@router.delete("/vector/index")
async def delete_index():
    """清空向量索引"""
    try:
        await clear_index()
        return success({"message": "向量索引已清空"})
    except Exception as e:
        logger.exception("清空向量索引失败")
        return fail(f"清空索引失败: {str(e)}")


@router.get("/vector/status")
async def index_status():
    """查询向量索引状态"""
    try:
        product_indexed = await is_indexed()
        kb_indexed = await is_knowledge_indexed()
        return success({
            "product_indexed": product_indexed,
            "knowledge_indexed": kb_indexed,
        })
    except Exception as e:
        logger.exception("查询索引状态失败")
        return fail("查询失败")


@router.post("/vector/index/knowledge")
async def build_knowledge_index():
    """
    构建知识库向量索引。
    将标准问答转为向量，存入 Redis。
    无需外部数据源，直接使用 data/knowledge.py 中的数据。
    """
    try:
        logger.info("正在构建知识库向量索引，共 %d 条...", len(KNOWLEDGE_ENTRIES))
        count = await index_knowledge_base(KNOWLEDGE_ENTRIES)

        return success({
            "total": len(KNOWLEDGE_ENTRIES),
            "indexed": count,
            "message": f"成功索引 {count}/{len(KNOWLEDGE_ENTRIES)} 条知识库条目",
        })
    except Exception as e:
        logger.exception("构建知识库索引失败")
        return fail(f"知识库索引构建失败: {str(e)}")


@router.post("/vector/index/all")
async def build_all_index():
    """
    一键构建全部索引：商品 + 知识库。
    建议首次部署时调用此接口。
    """
    try:
        # 先清旧索引
        await clear_index()

        # 构建知识库索引（不依赖外部服务，总是成功）
        kb_count = await index_knowledge_base(KNOWLEDGE_ENTRIES)

        # 构建商品索引
        logger.info("正在从 Java 服务拉取全量商品...")
        items = await fetch_all_items(page_size=100)
        product_count = 0
        if items:
            product_count = await index_items(items)
        else:
            logger.warning("未能获取商品数据，跳过商品索引")

        return success({
            "knowledge": {"total": len(KNOWLEDGE_ENTRIES), "indexed": kb_count},
            "product": {"total": len(items) if items else 0, "indexed": product_count},
            "message": f"知识库 {kb_count} 条 + 商品 {product_count} 件 索引完成",
        })
    except Exception as e:
        logger.exception("构建全部索引失败")
        return fail(f"索引构建失败: {str(e)}")
