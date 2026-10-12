"""向量嵌入服务 —— 本地 sentence-transformers 模型，无需额外 API 费用"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 全局模型单例
_model = None
_model_name: Optional[str] = None


def get_model(model_name: str = "BAAI/bge-small-zh-v1.5"):
    """
    获取/加载 embedding 模型（单例延迟加载）。
    首次调用会自动下载模型（约 100MB），后续使用缓存。
    """
    global _model, _model_name
    if _model is None or _model_name != model_name:
        logger.info("正在加载 embedding 模型: %s ...", model_name)
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model_name)
            _model_name = model_name
            logger.info("模型加载完成，向量维度: %d", _model.get_sentence_embedding_dimension())
        except Exception as e:
            logger.error("加载 embedding 模型失败: %s", e)
            raise RuntimeError(f"无法加载 embedding 模型 {model_name}: {e}")
    return _model


def embed(text: str, model_name: str = "BAAI/bge-small-zh-v1.5") -> Optional[np.ndarray]:
    """
    将单条文本转为向量。

    Args:
        text: 输入文本
        model_name: 模型名称

    Returns:
        numpy 向量 (ndarray)，失败返回 None
    """
    if not text or not text.strip():
        return None
    try:
        model = get_model(model_name)
        vec = model.encode(text.strip(), normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        logger.error("embed 失败: %s", e)
        return None


def embed_batch(texts: List[str], model_name: str = "BAAI/bge-small-zh-v1.5") -> List[Optional[np.ndarray]]:
    """
    批量将文本转为向量。

    Args:
        texts: 输入文本列表
        model_name: 模型名称

    Returns:
        向量列表，与输入一一对应（失败项为 None）
    """
    if not texts:
        return []
    valid_texts = [t.strip() for t in texts if t and t.strip()]
    if not valid_texts:
        return [None] * len(texts)

    try:
        model = get_model(model_name)
        vectors = model.encode(valid_texts, normalize_embeddings=True, show_progress_bar=False)
        result = []
        idx = 0
        for t in texts:
            if t and t.strip():
                result.append(np.array(vectors[idx], dtype=np.float32))
                idx += 1
            else:
                result.append(None)
        return result
    except Exception as e:
        logger.error("embed_batch 失败: %s", e)
        return [None] * len(texts)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """计算两个已归一化向量的余弦相似度（等价于点积）"""
    return float(np.dot(vec_a, vec_b))
