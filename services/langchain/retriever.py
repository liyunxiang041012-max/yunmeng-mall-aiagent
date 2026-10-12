"""LangChain Retriever — 包装向量存储为 LangChain Retriever 接口"""

import logging
from typing import List, Any, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from utils.vector_store import search_similar

logger = logging.getLogger(__name__)


class ProductVectorRetriever(BaseRetriever):
    """
    商品向量检索器 —— LangChain BaseRetriever 实现。
    用于 RAG Chain 中，将用户查询转为语义匹配的商品列表。
    """

    top_k: int = 5
    min_score: float = 0.3

    def __init__(self, top_k: int = 5, min_score: float = 0.3, **kwargs: Any):
        super().__init__(top_k=top_k, min_score=min_score, **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """同步检索（不推荐，本项目用异步版本）"""
        return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """异步检索商品，返回 LangChain Document 列表"""
        try:
            results = await search_similar(query, top_k=self.top_k, min_score=self.min_score)
            if not results:
                return []

            docs = []
            for r in results:
                name = r.get("name", "")
                price = r.get("price", 0)
                score = r.get("score", 0)
                content = f"商品：{name}，价格：{int(price) / 100:.2f}元" if price else f"商品：{name}"

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "id": r.get("id", ""),
                        "name": name,
                        "price": price,
                        "image": r.get("image", ""),
                        "score": score,
                    },
                ))

            logger.info("Retriever 检索到 %d 件商品，query=%s", len(docs), query)
            return docs
        except Exception as e:
            logger.error("Retriever 检索失败: %s", e)
            return []


def build_retriever(top_k: int = 5, min_score: float = 0.3) -> ProductVectorRetriever:
    """构建商品向量检索器"""
    return ProductVectorRetriever(top_k=top_k, min_score=min_score)
