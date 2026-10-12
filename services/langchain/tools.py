"""LangChain Tools — Agent 可调用的工具，替代关键词意图识别

每个 @tool 函数都有清晰的 docstring，LLM 根据 docstring 自主判断何时调用。
工具通过 contextvars 获取当前请求的 auth 信息，避免显式传参。
"""

import logging
import contextvars
from typing import Optional, List

from langchain_core.tools import tool

from utils.http_client import get_user_orders, get_user_cart, get_user_coupons, search_items
from utils.vector_store import search_similar, is_indexed, search_knowledge, is_knowledge_indexed
from config import settings

logger = logging.getLogger(__name__)

# ========== 请求上下文（通过 contextvars 在异步环境中安全传递） ==========

_auth_ctx = contextvars.ContextVar("auth_context", default={})


def set_auth_context(auth_header: Optional[str], user_id: Optional[str]):
    """设置当前请求的认证上下文"""
    _auth_ctx.set({
        "auth_header": auth_header,
        "user_id": user_id,
    })


def _get_auth() -> dict:
    """获取当前请求的认证上下文"""
    return _auth_ctx.get()


def _is_logged_in() -> bool:
    """检查当前用户是否已登录"""
    ctx = _get_auth()
    return bool(ctx.get("user_id") and ctx.get("auth_header"))


# ========== Tool 定义 ==========


@tool
async def query_user_orders() -> str:
    """
    查询当前登录用户的订单列表。

    当用户询问以下问题时调用此工具：
    - "我的订单"、"查订单"、"最近订单"
    - "订单到哪了"、"物流信息"、"快递"
    - "有什么订单"、"买了什么"

    返回：用户的订单列表（含商品名、金额、状态），或"暂无订单"的提示。
    需要用户已登录。
    """
    ctx = _get_auth()
    if not _is_logged_in():
        return "⚠️ 需要先登录才能查看订单。请登录后再说「我的订单」试试。"

    try:
        result = await get_user_orders(ctx["auth_header"], ctx["user_id"])
        if not result or not result.get("data"):
            return "您暂时没有订单记录。快去商城逛逛吧！"

        orders = result["data"]
        if not isinstance(orders, list) or not orders:
            return "您暂时没有订单记录。快去商城逛逛吧！"

        lines = [f"您共有 {len(orders)} 笔订单："]
        for i, o in enumerate(orders[:10], 1):
            name = o.get("productName", o.get("skuName", "商品"))
            status = o.get("status", "")
            amount = o.get("payAmount")
            price = f"{int(amount) / 100:.2f}元" if amount else "未知"
            lines.append(f"{i}. {name} | {_order_status_text(status)} | {price}")

        if len(orders) > 10:
            lines.append(f"... 还有 {len(orders) - 10} 笔订单")
        return "\n".join(lines)
    except Exception as e:
        logger.error("查询订单失败: %s", e)
        return "暂时无法查询订单，请稍后重试。"


@tool
async def query_user_coupons() -> str:
    """
    查询当前登录用户的优惠券。

    当用户询问以下问题时调用此工具：
    - "我有什么优惠券"、"优惠券"、"券"
    - "有什么折扣"、"满减券"
    - "领券"、"红包"

    返回：用户的优惠券列表（含券名、有效期），或"暂无优惠券"的提示。
    需要用户已登录。
    """
    ctx = _get_auth()
    if not _is_logged_in():
        return "⚠️ 需要先登录才能查看优惠券。登录后说「我有什么券」就可以查看哦。"

    try:
        result = await get_user_coupons(ctx["auth_header"], ctx["user_id"])
        if not result:
            return "暂时无法查询优惠券，请稍后重试。"

        data = result.get("data")
        coupons = data.get("list") if isinstance(data, dict) else data
        if not isinstance(coupons, list) or not coupons:
            return "您当前没有可用的优惠券。去「领券中心」看看吧，每日都有新券！"

        lines = [f"您有 {len(coupons)} 张可用优惠券："]
        for i, c in enumerate(coupons[:10], 1):
            end = c.get("termEndTime", "")
            line = f"{i}. {c.get('name', '优惠券')}"
            if end:
                line += f"（有效期至 {end}）"
            lines.append(line)

        if len(coupons) > 10:
            lines.append(f"... 还有 {len(coupons) - 10} 张券")
        return "\n".join(lines)
    except Exception as e:
        logger.error("查询优惠券失败: %s", e)
        return "暂时无法查询优惠券，请稍后重试。"


@tool
async def query_user_cart() -> str:
    """
    查询当前登录用户的购物车。

    当用户询问以下问题时调用此工具：
    - "我的购物车"、"购物车里有什么"
    - "购物车"、"加了什么"
    - "结算"、"去结算"

    返回：购物车中的商品列表及合计金额，或"购物车为空"的提示。
    需要用户已登录。
    """
    ctx = _get_auth()
    if not _is_logged_in():
        return "⚠️ 需要先登录才能查看购物车。登录后说「我的购物车」试试。"

    try:
        result = await get_user_cart(ctx["auth_header"], ctx["user_id"])
        if not result or not result.get("data"):
            return "您的购物车是空的，快去挑选心仪的商品吧！"

        items = result["data"]
        if not isinstance(items, list) or not items:
            return "您的购物车是空的，快去挑选心仪的商品吧！"

        total = 0
        lines = [f"您的购物车有 {len(items)} 件商品："]
        for i, it in enumerate(items[:10], 1):
            price = it.get("price", 0)
            total += int(price) if price else 0
            lines.append(f"{i}. {it.get('skuName', '商品')}")

        if len(items) > 10:
            lines.append(f"... 还有 {len(items) - 10} 件")
        lines.append(f"\n合计约 {total / 100:.2f} 元")
        return "\n".join(lines)
    except Exception as e:
        logger.error("查询购物车失败: %s", e)
        return "暂时无法查询购物车，请稍后重试。"


@tool
async def search_mall_products(query: str) -> str:
    """
    在商城中搜索或推荐商品。支持语义搜索。

    当用户询问以下问题时调用此工具：
    - "推荐一款耳机"、"有什么好的T恤"
    - "帮我找"、"想买"、"搜一下"
    - "有什么划算的"、"推荐"

    Args:
        query: 用户的搜索意图或关键词。例如用户说"推荐一款蓝牙耳机"，query 就传"蓝牙耳机"。

    返回：匹配的商品列表（含名称、价格），或"未找到"的提示。
    """
    if not query or not query.strip():
        return "请告诉我您想找什么商品，比如「推荐一款耳机」。"

    query = query.strip()

    # 1. 优先向量语义检索
    try:
        indexed = await is_indexed()
        if indexed:
            results = await search_similar(query, top_k=5, min_score=0.3)
            if results:
                lines = [f"为您找到以下相关商品："]
                for i, r in enumerate(results, 1):
                    name = r.get("name", "商品")
                    price = r.get("price", 0)
                    price_str = f"{int(price) / 100:.2f}元" if price else "未知"
                    score = r.get("score", 0)
                    lines.append(f"{i}. {name} — {price_str}（匹配度 {score:.0%}）")
                return "\n".join(lines)
    except Exception as e:
        logger.warning("向量检索失败，降级关键词搜索: %s", e)

    # 2. 降级：关键词搜索
    try:
        result = await search_items(query)
        if result and result.get("data"):
            items = result["data"]
            if isinstance(items, dict):
                items = items.get("list", [])
            if isinstance(items, list) and items:
                lines = [f"为您找到以下相关商品："]
                for i, it in enumerate(items[:5], 1):
                    name = str(it.get("name", it.get("skuName", "商品")))
                    price = it.get("price", 0)
                    price_str = f"{int(price) / 100:.2f}元" if price else "未知"
                    lines.append(f"{i}. {name} — {price_str}")
                return "\n".join(lines)
    except Exception as e:
        logger.error("商品搜索失败: %s", e)

    return f"没有找到与「{query}」相关的商品，换个关键词试试吧。"


@tool
async def query_knowledge_base(question: str) -> str:
    """
    查询商城常见问题知识库。

    当用户询问商城使用规则、政策、流程时调用此工具，例如：
    - "怎么退货"、"退款流程"
    - "优惠券怎么用"、"怎么领券"
    - "如何联系客服"、"电话多少"
    - "支持什么支付方式"、"多久发货"
    - "怎么修改地址"、"忘记密码怎么办"

    Args:
        question: 用户的问题原文或关键词。

    返回：知识库中的标准答案，或"未找到"的提示。
    """
    if not question or not question.strip():
        return ""

    try:
        kb_indexed = await is_knowledge_indexed()
        if not kb_indexed:
            return ""

        results = await search_knowledge(question.strip(), top_k=2, min_score=0.35)
        if not results:
            return ""

        lines = []
        for r in results:
            lines.append(r.get("answer", ""))

        return "\n\n".join(lines)
    except Exception as e:
        logger.warning("知识库检索失败: %s", e)
        return ""


# ========== 内部辅助 ==========

def _order_status_text(status: str) -> str:
    mapping = {
        "PENDING_PAYMENT": "待付款", "0": "待付款",
        "PAID": "已付款", "1": "已付款",
        "SHIPPED": "已发货", "2": "已发货",
        "COMPLETED": "已完成", "3": "已完成",
        "CANCELLED": "已取消", "4": "已取消",
        "REFUNDED": "已退款", "5": "已退款",
    }
    return mapping.get(str(status), str(status))


# ========== 工具集合 ==========

ALL_TOOLS = [
    query_user_orders,
    query_user_coupons,
    query_user_cart,
    search_mall_products,
    query_knowledge_base,
]
