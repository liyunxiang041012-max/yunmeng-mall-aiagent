"""AI 商品审核服务 —— 替代 Java AiReviewService"""

import json
import logging
import re
from typing import Optional
from services.deepseek_service import chat as deepseek_chat

logger = logging.getLogger(__name__)

REVIEW_PROMPT = """你是云梦商城的内容审核助手。只审核文字信息，图片由人工审核。

严格按以下规则判断，返回 JSON（不要加 ``` 标记）：
{"suggestion":"approve|reject","reason":"具体原因，一句话"}

【必须拒绝 reject】
- 含脏话、人身攻击、政治敏感、色情暗示
- 含联系方式（手机号、微信号、QQ号、网址）
- 含违法信息（枪支、毒品、赌博、假币、盗版）
- 商品名与商城品类完全无关（如"今天天气真好"）
- 明显刷单/测试数据（如"测试商品123"、"111"）

【建议通过 approve】
- 名称清晰描述商品，中英文均可，如"iPhone 15 Pro Max"
- 名称包含品牌+品类+型号，如"华为Mate 60手机"
- 名称虽简短但明确，如"男士T恤"、"有机大米"

【不确定 review】
- 名称模糊但有可能是真实商品，如"新款上市"、"特价清仓"
- 无法判断是否存在违规但感觉不太对劲"""


def local_check(name: Optional[str], price: Optional[int], stock: Optional[int]) -> Optional[dict]:
    """本地规则快速检查（不调 AI，省 Token）。返回 dict 表示命中规则，None 表示通过"""
    # 规则1：名称为空
    if not name or not name.strip():
        return {"suggestion": "reject", "reason": "【违规】商品名称为空"}

    name = name.strip()

    # 规则2：名称过短（单字无意义）
    if len(name) < 2:
        return {"suggestion": "reject", "reason": "【违规】商品名称过短（少于2个字符），无法识别商品"}

    # 规则3：纯数字名称
    if re.match(r"^\d+$", name):
        return {"suggestion": "reject", "reason": f"【违规】商品名称为纯数字'{name}'，不是有效商品名"}

    # 规则4：纯符号名称
    if re.match(r"^[^a-zA-Z\u4e00-\u9fa5\d]+$", name):
        return {"suggestion": "reject", "reason": "【违规】商品名称仅含符号，不是有效商品名"}

    # 规则5：名称超过100字符
    if len(name) > 100:
        return {"suggestion": "reject", "reason": "【违规】商品名称过长（超过100字符），请精简到100字符以内"}

    # 规则6：价格异常
    if price is not None and price <= 0:
        return {"suggestion": "reject", "reason": "【违规】商品价格必须大于0"}

    # 规则7：库存为负
    if stock is not None and stock < 0:
        return {"suggestion": "reject", "reason": "【违规】库存不能为负数"}

    return None  # 本地规则通过


async def ai_review(name: Optional[str], price: Optional[int], stock: Optional[int]) -> dict:
    """
    审核商品：先本地规则，再 AI 深度检查。
    返回 {"suggestion": "approve|reject|review", "reason": "..."}
    """
    # 1. 本地规则快速拦截
    local = local_check(name, price, stock)
    if local:
        return local

    # 2. 构建提示
    price_yuan = f"{price / 100:.2f}元" if price is not None else "未知"
    user_prompt = f"审核商品：名称='{name or ''}'，价格={price_yuan}，库存={stock or 0}"

    # 3. AI 深度检查
    try:
        result_text = await deepseek_chat(REVIEW_PROMPT, user_prompt)
        if result_text:
            return _parse_result(result_text)
    except Exception as e:
        logger.error("AI审核调用失败: %s", e)

    return {"suggestion": "review", "reason": "AI审核服务暂不可用，请人工审核"}


def _parse_result(text: str) -> dict:
    """解析 DeepSeek 返回的审核结果"""
    text = text.strip().replace("```json", "").replace("```", "").strip()

    suggestion = "review"
    reason = text

    if '"suggestion"' in text:
        suggestion = _extract_json_val(text, "suggestion") or "review"
        r = _extract_json_val(text, "reason")
        if r:
            reason = r

    # 合法值校验
    if suggestion not in ("approve", "reject", "review"):
        suggestion = "review"

    return {"suggestion": suggestion, "reason": reason}


def _extract_json_val(text: str, key: str) -> Optional[str]:
    m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
    return m.group(1) if m else None
