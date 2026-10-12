"""应用配置 —— 从环境变量 / .env 加载"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ========== DeepSeek ==========
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_TEMPERATURE: float = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7"))
    DEEPSEEK_MAX_TOKENS: int = int(os.getenv("DEEPSEEK_MAX_TOKENS", "1000"))

    # ========== Redis ==========
    REDIS_HOST: str = os.getenv("REDIS_HOST", "192.168.150.101")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    CHAT_HISTORY_TTL_HOURS: int = int(os.getenv("CHAT_HISTORY_TTL_HOURS", "24"))

    # ========== 网关 ==========
    GATEWAY_URL: str = os.getenv("GATEWAY_URL", "http://localhost:8080")

    # ========== 向量检索 ==========
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "5"))
    VECTOR_MIN_SCORE: float = float(os.getenv("VECTOR_MIN_SCORE", "0.3"))

    # ========== LangChain ==========
    LANGCHAIN_VERBOSE: bool = os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true"
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "yunmeng-ai")

    # ========== JWT ==========
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")


settings = Settings()
