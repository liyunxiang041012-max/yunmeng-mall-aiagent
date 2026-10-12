"""LangChain ChatModel — DeepSeek via OpenAI-compatible API"""

from langchain_openai import ChatOpenAI
from config import settings


def build_chat_model(streaming: bool = False) -> ChatOpenAI:
    """
    构建 ChatOpenAI 实例，对接 DeepSeek API。

    DeepSeek 的 API 完全兼容 OpenAI 格式，所以用 ChatOpenAI 即可。
    支持 function calling (tool calling)，用于 Agent 自主决策。
    """
    return ChatOpenAI(
        model=settings.DEEPSEEK_MODEL,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=settings.DEEPSEEK_TEMPERATURE,
        max_tokens=settings.DEEPSEEK_MAX_TOKENS,
        streaming=streaming,
        model_kwargs={
            # DeepSeek 特有的参数（非标准 OpenAI 参数放这里）
        },
    )
