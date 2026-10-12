"""HTTP 客户端 —— 通过网关调用 Java 微服务"""

import httpx
from typing import Optional, Dict, Any
from config import settings

GATEWAY = settings.GATEWAY_URL


async def call_microservice(
    method: str,
    path: str,
    auth_header: Optional[str] = None,
    user_id: Optional[str] = None,
    params: Optional[dict] = None,
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """
    通过网关调用 Java 微服务，自动透传认证信息。

    Args:
        method: GET / POST / PUT / DELETE
        path: 网关路径，如 /py/order/list
        auth_header: Authorization header 值
        user_id: user-id header 值
        params: URL 查询参数
        timeout: 超时秒数

    Returns:
        JSON 响应字典，失败返回 None
    """
    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header
    if user_id:
        headers["user-id"] = user_id

    url = f"{GATEWAY}{path}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method.upper() == "GET":
                resp = await client.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                resp = await client.post(url, headers=headers, json=params)
            elif method.upper() == "PUT":
                resp = await client.put(url, headers=headers, json=params)
            else:
                resp = await client.delete(url, headers=headers, params=params)

            if resp.status_code == 200:
                return resp.json()
            return None
    except Exception:
        return None


# ==================== 便捷调用 ====================

async def get_user_orders(auth: str, uid: str) -> Optional[dict]:
    """查询用户订单"""
    return await call_microservice("GET", "/py/order/list", auth_header=auth, user_id=uid)


async def get_user_cart(auth: str, uid: str) -> Optional[dict]:
    """查询用户购物车"""
    return await call_microservice("GET", "/ca/cart/list", auth_header=auth, user_id=uid)


async def get_user_coupons(auth: str, uid: str, status: int = 1) -> Optional[dict]:
    """查询用户优惠券"""
    return await call_microservice("GET", "/pm/user-coupons/page", auth_header=auth, user_id=uid, params={"status": status})


async def search_items(keyword: str, page_size: int = 20) -> Optional[dict]:
    """搜索商品（无需登录）"""
    return await call_microservice("GET", "/it/items/page", params={"keyword": keyword, "pageSize": page_size})


async def fetch_all_items(page_size: int = 100) -> Optional[list]:
    """
    拉取全量商品列表（用于向量索引构建）。
    分页拉取，合并后返回商品列表。
    """
    all_items = []
    page = 1
    while True:
        result = await call_microservice("GET", "/it/items/page", params={"page": page, "pageSize": page_size})
        if not result:
            break
        data = result.get("data")
        items = []
        if isinstance(data, dict):
            items = data.get("list", [])
        elif isinstance(data, list):
            items = data
        if not items:
            break
        all_items.extend(items)
        if len(items) < page_size:
            break
        page += 1
    return all_items
