"""LangChain Embeddings — 包装 sentence-transformers 为 LangChain 接口"""

from typing import List, Optional, Any
from langchain_core.embeddings import Embeddings
from services.embedding_service import get_model
from config import settings


class HuggingFaceEmbeddings(Embeddings):
    """
    LangChain Embeddings 接口实现。
    内部复用现有的 sentence-transformers 模型单例，不需要重新加载。
    """

    model_name: str = ""

    def __init__(self, model_name: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_name = model_name or settings.EMBEDDING_MODEL
        # 私有属性：懒加载的模型实例
        object.__setattr__(self, "_model", None)

    def _get_model(self):
        if self._model is None:
            self._model = get_model(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档"""
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化"""
        model = self._get_model()
        vec = model.encode(text.strip(), normalize_embeddings=True)
        return vec.tolist()


def build_embeddings() -> HuggingFaceEmbeddings:
    """构建 LangChain Embeddings 实例"""
    return HuggingFaceEmbeddings()
