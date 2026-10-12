"""向量存储 —— Redis 持久化 + 内存缓存，余弦相似度检索"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from services.embedding_service import embed, embed_batch, cosine_similarity
from utils.redis_client import get_redis

logger = logging.getLogger(__name__)

# Redis Key 前缀
VECTOR_KEY_PREFIX = "item:vector:"
VECTOR_META_KEY = "item:vector:meta:"
VECTOR_INDEX_TTL = 3600 * 24  # 向量缓存 24 小时

# 内存缓存：item_id -> numpy vector
_cache: Dict[str, np.ndarray] = {}

# ==================== 知识库相关 ====================

KB_KEY_PREFIX = "kb:vector:"
KB_META_KEY = "kb:vector:meta:"
KB_TTL = 3600 * 24 * 7  # 知识库向量缓存 7 天

_kb_cache: Dict[str, np.ndarray] = {}  # 知识库内存缓存


async def index_items(items: List[dict]) -> int:
    """
    将商品列表向量化并存入 Redis + 内存缓存。

    Args:
        items: 商品列表，每个元素至少含 id, name 字段

    Returns:
        成功索引的数量
    """
    if not items:
        return 0

    # 构建待向量化的文本（name + 简短描述）
    texts = []
    valid_items = []
    for item in items:
        item_id = str(item.get("id", ""))
        if not item_id:
            continue
        name = item.get("name", "") or item.get("skuName", "") or ""
        desc = item.get("description", "") or item.get("subTitle", "") or ""
        text = f"{name} {desc}".strip()
        if not text:
            continue
        texts.append(text)
        valid_items.append(item)

    if not texts:
        return 0

    logger.info("正在向量化 %d 件商品...", len(texts))
    vectors = embed_batch(texts)

    r = await get_redis()
    count = 0

    for item, vec in zip(valid_items, vectors):
        if vec is None:
            continue
        item_id = str(item["id"])
        vec_bytes = vec.astype(np.float32).tobytes()

        # 存入 Redis
        await r.set(VECTOR_KEY_PREFIX + item_id, vec_bytes, ex=VECTOR_INDEX_TTL)

        # 存入元信息（用于返回结果时拼接数据）
        meta = {
            "id": item_id,
            "name": item.get("name", ""),
            "price": item.get("price", 0),
            "image": item.get("image", "") or item.get("defaultImage", ""),
        }
        await r.set(VECTOR_META_KEY + item_id, json.dumps(meta, ensure_ascii=False), ex=VECTOR_INDEX_TTL)

        # 内存缓存
        _cache[item_id] = vec
        count += 1

    logger.info("成功索引 %d 件商品", count)
    return count


async def search_similar(
    query_text: str,
    top_k: int = 5,
    min_score: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    向量相似度检索：输入查询文本，返回最相似的 top_k 件商品。

    Args:
        query_text: 查询文本
        top_k: 返回结果数量
        min_score: 最低相似度阈值（0~1），低于此值的结果会被过滤

    Returns:
        商品列表，按相似度降序排列，每个元素含 id, name, price, image, score
    """
    if not query_text or not query_text.strip():
        return []

    # 1. 查询向量化
    query_vec = embed(query_text.strip())
    if query_vec is None:
        logger.warning("查询向量化失败: %s", query_text)
        return []

    # 2. 加载索引中的向量并计算相似度
    results = await _compute_scores(query_vec, top_k, min_score)

    # 3. 拼接元信息
    return await _attach_meta(results)


async def _compute_scores(
    query_vec: np.ndarray,
    top_k: int,
    min_score: float,
) -> List[Tuple[str, float]]:
    """计算所有缓存向量与查询向量的相似度，返回 top_k"""

    # 先尝试内存缓存
    if _cache:
        scores = []
        for item_id, vec in _cache.items():
            score = cosine_similarity(query_vec, vec)
            if score >= min_score:
                scores.append((item_id, score))
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

    # 内存缓存为空，从 Redis 加载
    r = await get_redis()
    keys = []
    cursor = 0
    while True:
        cursor, batch = await r.scan(cursor, match=VECTOR_KEY_PREFIX + "*", count=200)
        keys.extend(batch)
        if cursor == 0:
            break

    if not keys:
        return []

    scores = []
    for key in keys:
        item_id = key[len(VECTOR_KEY_PREFIX):]
        vec_bytes = await r.get(key)
        if not vec_bytes:
            continue
        try:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            # 缓存到内存
            _cache[item_id] = vec
            score = cosine_similarity(query_vec, vec)
            if score >= min_score:
                scores.append((item_id, score))
        except Exception as e:
            logger.warning("解析向量失败 key=%s: %s", key, e)

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


async def _attach_meta(scores: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """给相似度结果拼接商品元信息"""
    if not scores:
        return []

    r = await get_redis()
    results = []
    for item_id, score in scores:
        # 优先从内存缓存读；这里需要额外存储元信息，从 Redis 取
        meta_raw = await r.get(VECTOR_META_KEY + item_id)
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
                meta["score"] = round(score, 4)
                results.append(meta)
            except json.JSONDecodeError:
                continue

    return results


async def is_indexed() -> bool:
    """检查是否有已索引的商品向量"""
    if _cache:
        return True
    r = await get_redis()
    cursor, keys = await r.scan(0, match=VECTOR_KEY_PREFIX + "*", count=1)
    return len(keys) > 0


# ==================== 知识库操作 ====================


async def index_knowledge_base(entries: List[dict]) -> int:
    """
    将标准问答知识库向量化并存入 Redis + 内存缓存。

    Args:
        entries: 知识库条目列表，每个含 question, answer, intent, category, keywords

    Returns:
        成功索引的数量
    """
    if not entries:
        return 0

    # 构建待向量化的文本：question + keywords（增强匹配）
    texts = []
    for entry in entries:
        question = entry.get("question", "")
        keywords = " ".join(entry.get("keywords", []))
        text = f"{question} {keywords}".strip()
        if not text:
            continue
        texts.append(text)

    if not texts:
        return 0

    logger.info("正在向量化 %d 条知识库条目...", len(texts))
    vectors = embed_batch(texts)

    r = await get_redis()
    count = 0

    for entry, vec in zip(entries, vectors):
        if vec is None:
            continue
        # 用 question 的 hash 作为唯一 key
        entry_id = str(hash(entry.get("question", "")))
        vec_bytes = vec.astype(np.float32).tobytes()

        # 存入 Redis
        await r.set(KB_KEY_PREFIX + entry_id, vec_bytes, ex=KB_TTL)

        # 存入元信息
        meta = {
            "question": entry.get("question", ""),
            "answer": entry.get("answer", ""),
            "intent": entry.get("intent", ""),
            "category": entry.get("category", ""),
            "keywords": entry.get("keywords", []),
        }
        await r.set(KB_META_KEY + entry_id, json.dumps(meta, ensure_ascii=False), ex=KB_TTL)

        # 内存缓存
        _kb_cache[entry_id] = vec
        count += 1

    logger.info("成功索引 %d 条知识库条目", count)
    return count


async def search_knowledge(
    query_text: str,
    top_k: int = 3,
    min_score: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    向量检索知识库：输入用户问题，返回最相似的标准问答。

    Args:
        query_text: 用户输入的问题
        top_k: 返回结果数量
        min_score: 最低相似度阈值

    Returns:
        知识库条目列表，每个含 question, answer, intent, category, score
    """
    if not query_text or not query_text.strip():
        return []

    # 1. 查询向量化
    query_vec = embed(query_text.strip())
    if query_vec is None:
        return []

    # 2. 计算相似度
    scores = await _compute_kb_scores(query_vec, top_k, min_score)

    # 3. 拼接元信息
    return await _attach_kb_meta(scores)


async def _compute_kb_scores(
    query_vec: np.ndarray,
    top_k: int,
    min_score: float,
) -> List[Tuple[str, float]]:
    """计算知识库向量的相似度"""

    # 先尝试内存缓存
    if _kb_cache:
        scores = []
        for kb_id, vec in _kb_cache.items():
            score = cosine_similarity(query_vec, vec)
            if score >= min_score:
                scores.append((kb_id, score))
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

    # 从 Redis 加载
    r = await get_redis()
    keys = []
    cursor = 0
    while True:
        cursor, batch = await r.scan(cursor, match=KB_KEY_PREFIX + "*", count=200)
        keys.extend(batch)
        if cursor == 0:
            break

    if not keys:
        return []

    scores = []
    for key in keys:
        kb_id = key[len(KB_KEY_PREFIX):]
        vec_bytes = await r.get(key)
        if not vec_bytes:
            continue
        try:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            _kb_cache[kb_id] = vec
            score = cosine_similarity(query_vec, vec)
            if score >= min_score:
                scores.append((kb_id, score))
        except Exception as e:
            logger.warning("解析知识库向量失败 key=%s: %s", key, e)

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


async def _attach_kb_meta(scores: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """给知识库相似度结果拼接元信息"""
    if not scores:
        return []

    r = await get_redis()
    results = []
    for kb_id, score in scores:
        meta_raw = await r.get(KB_META_KEY + kb_id)
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
                meta["score"] = round(score, 4)
                results.append(meta)
            except json.JSONDecodeError:
                continue

    return results


async def is_knowledge_indexed() -> bool:
    """检查知识库是否已索引"""
    if _kb_cache:
        return True
    r = await get_redis()
    cursor, keys = await r.scan(0, match=KB_KEY_PREFIX + "*", count=1)
    return len(keys) > 0


async def clear_index():
    """清空全部向量索引（商品 + 知识库，内存 + Redis）"""
    global _cache, _kb_cache
    _cache.clear()
    _kb_cache.clear()

    r = await get_redis()

    # 清除商品向量
    await _delete_by_pattern(r, VECTOR_KEY_PREFIX)
    await _delete_by_pattern(r, VECTOR_META_KEY)

    # 清除知识库向量
    await _delete_by_pattern(r, KB_KEY_PREFIX)
    await _delete_by_pattern(r, KB_META_KEY)

    logger.info("向量索引已清空")


async def _delete_by_pattern(r, pattern: str):
    """按 pattern 扫描并批量删除 Redis key"""
    cursor = 0
    while True:
        cursor, keys = await r.scan(cursor, match=pattern + "*", count=200)
        if keys:
            await r.delete(*keys)
        if cursor == 0:
            break
