"""LangChain 模块 —— 统一导出"""

from services.langchain.chat_model import build_chat_model
from services.langchain.embeddings import build_embeddings, HuggingFaceEmbeddings
from services.langchain.prompts import build_agent_prompt, build_chat_prompt
from services.langchain.tools import ALL_TOOLS, set_auth_context
from services.langchain.memory import get_session_history, RedisChatMessageHistory
from services.langchain.retriever import build_retriever, ProductVectorRetriever
from services.langchain.chain import (
    build_chain_with_memory,
    chat_with_agent,
    chat_with_agent_stream,
)

__all__ = [
    "build_chat_model",
    "build_embeddings",
    "HuggingFaceEmbeddings",
    "build_agent_prompt",
    "build_chat_prompt",
    "ALL_TOOLS",
    "set_auth_context",
    "get_session_history",
    "RedisChatMessageHistory",
    "build_retriever",
    "ProductVectorRetriever",
    "build_chain_with_memory",
    "chat_with_agent",
    "chat_with_agent_stream",
]
