# 云梦 AI 助手 — Python 服务开发文档

> **v3.0** — LangChain Agent + Tool Calling 架构，替代原 Java `yunmeng-ai` 模块。
> 前端零改动，网关一层路由切换即可完成迁移。

---

## 一、项目概览

### 1.1 目录结构

```
python-ai/
├── main.py                      # FastAPI 入口 (uvicorn :8095)
├── config.py                    # 配置中心 (.env → Settings)
├── requirements.txt             # Python 依赖
├── .env                         # 密钥、密码 (不提交 Git)
├── models/
│   ├── request.py               # Pydantic 请求体
│   └── response.py              # 统一响应封装
├── routers/
│   ├── chat.py                  # /chat, /chat/stream, /history
│   ├── review.py                # /review/item 商品审核
│   └── vector.py                # /vector/index 向量索引管理
├── services/
│   ├── ai_service.py            # 对话服务 (委托 LangChain Agent)
│   ├── review_service.py        # 商品AI审核 (本地规则 + DeepSeek)
│   ├── deepseek_service.py      # DeepSeek API 底层调用 (仅审核用)
│   ├── embedding_service.py     # sentence-transformers 向量模型管理
│   └── langchain/               # 🔥 LangChain Agent 核心模块
│       ├── __init__.py          # 统一导出
│       ├── chain.py             # Agent + Tool + Memory 组装
│       ├── tools.py             # 5 个 @tool 智能工具
│       ├── prompts.py           # ChatPromptTemplate 模板
│       ├── memory.py            # RedisChatMessageHistory 实现
│       ├── chat_model.py        # ChatOpenAI(DeepSeek) 封装
│       ├── embeddings.py        # LangChain Embeddings 接口适配
│       └── retriever.py         # LangChain Retriever 接口适配
├── utils/
│   ├── http_client.py           # Java 网关 HTTP 调用
│   ├── vector_store.py          # Redis 向量存储 + 余弦检索
│   └── redis_client.py          # Redis 连接池 + 历史 CRUD
└── data/
    └── knowledge.py             # 16 条标准问答知识库
```

### 1.2 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 异步高性能 |
| AI 编排 | LangChain 1.3 + langchain-classic | Agent + Tool Calling + Memory |
| LLM | DeepSeek (deepseek-chat) | 通过 OpenAI 兼容 API 调用 |
| 向量化 | BAAI/bge-small-zh-v1.5 | 512 维中文语义向量 |
| 缓存 | Redis 6.0+ | 对话历史 + 向量索引 |
| 业务接口 | Java 网关 (HTTP) | 透传 JWT 调用订单/优惠券等微服务 |

---

## 二、核心架构

### 2.1 请求全链路

```
前端 (Vue)
  │  Authorization: Bearer <JWT>
  ▼
┌─────────────────────────────────────────┐
│           Gateway :8080                  │
│  AuthGlobalFilter: 解析JWT → 注入 header │
│  user-id, user-role                     │
│  StripPrefix=1:  /ai/chat → /chat       │
└─────────────────────────┬───────────────┘
                          ▼
┌──────────────────────────────────────────┐
│        Python FastAPI :8095              │
│                                           │
│  POST /chat  →  ai_service.handle_chat() │
│                   ↓                      │
│            LangChain AgentExecutor       │
│            ┌──────────────────┐          │
│            │  ChatOpenAI       │          │
│            │  (DeepSeek)       │          │
│            │                   │          │
│            │  Tool Calling:    │          │
│            │  ├ query_user_orders        │
│            │  ├ query_user_coupons       │
│            │  ├ query_user_cart          │
│            │  ├ search_mall_products     │
│            │  └ query_knowledge_base     │
│            │                   │          │
│            │  RunnableWithMessageHistory │
│            │  → RedisChatMessageHistory  │
│            └──────────────────┘          │
│                                           │
│  POST /chat/stream → SSE 逐 token 推送   │
│  POST /review/item → 本地规则 + AI审核   │
│  POST /vector/index → 构建向量索引       │
└──────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
   ┌──────────┐  ┌──────────┐  ┌─────────────┐
   │ DeepSeek │  │  Redis   │  │ Java 网关    │
   │   API    │  │ :6379    │  │ :8080        │
   │          │  │          │  │ /py/order/*  │
   │ 闲聊+推理│  │ 历史+向量│  │ /ca/cart/*   │
   └──────────┘  └──────────┘  │ /pm/coupon/* │
                               └─────────────┘
```

### 2.2 Agent 决策流程

```
用户: "帮我查订单，再推荐一款耳机"
        │
        ▼
  AgentExecutor.ainvoke()
        │
        ▼
  DeepSeek 分析输入 + 读取 Tool docstrings
        │
        ├─→ 判断需要 query_user_orders()
        │      → 执行 Tool → Java 网关 → 订单数据
        │
        ├─→ 判断需要 search_mall_products("耳机")
        │      → 执行 Tool → Redis 向量检索 → 商品列表
        │
        ▼
  DeepSeek 综合 2 个 Tool 结果 + 对话历史 → 生成回复
        │
        ▼
  保存 HumanMessage + AIMessage 到 Redis
        │
        ▼
  返回: "您有2笔订单：1. iPhone 15... 为您推荐以下耳机：..."
```

> **关键点**：Agent 自主决定调用哪个 Tool、调用顺序、如何组装结果。不需要人工编写 if-else 分支。

### 2.3 对话记忆机制

```
第1轮: 用户"你好"          → Redis: []
        Agent处理           → Redis: [Human("你好"), AI("你好呀！...")]
        
第2轮: 用户"我的订单"      → Redis: [Human("你好"), AI("你好呀！...")]
        Agent处理           → Redis: [... Human("我的订单"), AI("您有2笔订单...")]
        
第N轮: Agent 始终看到完整历史上下文
```

- 实现 `BaseChatMessageHistory` 接口
- 数据格式与 Java 端完全兼容：`chat:history:{sessionId}` → List<{role, text, time}>
- TTL 默认 24 小时，可通过 `CHAT_HISTORY_TTL_HOURS` 配置

---

## 三、五大智能工具 (Tools)

每个工具就是一个 `@tool` 装饰的异步函数，**docstring 是 LLM 决策的依据**。

### 3.1 query_user_orders
- **触发词**：我的订单、物流、快递、发货
- **数据源**：Java 网关 `GET /py/order/list`
- **需登录**：是
- **未登录返回**：`⚠️ 需要先登录才能查看订单...`

### 3.2 query_user_coupons
- **触发词**：优惠券、领券、满减、红包
- **数据源**：Java 网关 `GET /pm/user-coupons/page`
- **需登录**：是

### 3.3 query_user_cart
- **触发词**：购物车、结算
- **数据源**：Java 网关 `GET /ca/cart/list`
- **需登录**：是

### 3.4 search_mall_products
- **触发词**：推荐、搜一下、想买、有什么好
- **数据源**：Redis 向量检索 → 降级 Java 关键词搜索
- **需登录**：否
- **参数**：`query: str` — 搜索关键词

### 3.5 query_knowledge_base
- **触发词**：怎么退货、如何支付、多久发货、忘记密码
- **数据源**：Redis 知识库向量检索 (16 条标准问答)
- **需登录**：否
- **参数**：`question: str` — 用户问题原文

### 如何新增 Tool？

只需 3 步：

```python
# 1. 在 services/langchain/tools.py 中添加
@tool
async def query_new_feature(param: str) -> str:
    """工具描述：何时触发、参数含义、返回值格式"""
    # 业务逻辑
    return result

# 2. 加入工具列表
ALL_TOOLS = [..., query_new_feature]

# 3. 重启服务 — Agent 自动识别新工具
```

---

## 四、接口文档

> 所有路径不带 `/ai` 前缀，由网关 `StripPrefix=1` 去掉。

### 4.1 POST /chat — 发送消息 (非流式)

```json
// 请求
{
  "message": "帮我查最近订单",
  "sessionId": "ai_1786243524013_e52721"
}

// 响应
{
  "code": 200,
  "data": {
    "reply": "您最近有 3 笔订单：\n1. iPhone 15 Pro Max — 已发货...",
    "sessionId": "ai_1786243524013_e52721"
  }
}
```

### 4.2 POST /chat/stream — SSE 流式输出

```json
// 请求体同上，前端使用 EventSource 接收

// SSE 事件流:
event: token
data: {"content": "您"}

event: token
data: {"content": "好"}

event: tool
data: {"tool": "query_user_orders", "status": "start"}

event: tool
data: {"tool": "query_user_orders", "status": "end"}

event: token
data: {"content": "！"}

event: done
data: {}
```

前端接收示例：
```javascript
const eventSource = new EventSource('/chat/stream');
eventSource.addEventListener('token', (e) => {
  const { content } = JSON.parse(e.data);
  // 逐字追加到界面
});
eventSource.addEventListener('tool', (e) => {
  const { tool, status } = JSON.parse(e.data);
  // 显示 "正在查询订单..." / "查询完成"
});
```

### 4.3 GET /history — 对话历史

```
GET /history?sessionId=ai_xxx
```

```json
{
  "code": 200,
  "data": {
    "sessionId": "ai_xxx",
    "messages": [
      {"role": "ai",   "text": "你好！我是云梦AI助手...", "time": "2026-08-09 14:30:00"},
      {"role": "user", "text": "帮我查订单",              "time": "2026-08-09 14:31:00"}
    ]
  }
}
```

### 4.4 DELETE /history/clear — 清除历史

```
DELETE /history/clear?sessionId=ai_xxx
→ {"code": 200}
```

### 4.5 POST /review/item — AI 商品审核

```json
// 请求
{"name": "蓝牙耳机", "price": 29900, "stock": 100}

// 响应
{"code": 200, "data": {"suggestion": "approve", "reason": "商品名称清晰，符合规范"}}
```

suggestion 可选值：`approve` / `reject` / `review`

### 4.6 POST /vector/index/all — 一键构建全部向量索引

首次部署后调用，构建商品 + 知识库向量索引。

### 4.7 GET /vector/status — 索引状态

```json
{"code": 200, "data": {"product_indexed": true, "knowledge_indexed": true}}
```

---

## 五、环境要求与启动

### 5.1 依赖服务

| 服务 | 地址 | 说明 |
|------|------|------|
| DeepSeek API | api.deepseek.com | LLM 推理 |
| Redis | 192.168.150.101:6379 | 对话历史 + 向量存储 |
| Java 网关 | localhost:8080 | 业务数据 (订单/优惠券等) |

### 5.2 安装启动

```bash
cd python-ai

# 1. 虚拟环境
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 .env
cp .env.example .env
# 编辑 DEEPSEEK_API_KEY, REDIS_HOST 等

# 4. 启动
uvicorn main:app --host 0.0.0.0 --port 8095 --reload

# 5. 首次部署：构建向量索引
curl -X POST http://localhost:8095/vector/index/all
```

### 5.3 验证

```bash
# 健康检查
curl http://localhost:8095/health
# → {"status":"ok","service":"yunmeng-ai-python"}

# 发送消息
curl -X POST http://localhost:8095/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好，介绍一下你自己","sessionId":""}'
```

---

## 六、降级与容错

| 场景 | 策略 |
|------|------|
| DeepSeek API 不可用 | `ai_service.handle_chat()` 捕获异常，返回 "AI 服务暂时不可用" |
| Redis 不可用 | `RedisChatMessageHistory` 返回空历史，对话仍可进行（无记忆） |
| Java 网关不可用 | Tool 返回 "暂时无法查询"，Agent 告知用户稍后重试 |
| 向量检索失败 | `search_mall_products` 自动降级到 Java 关键词搜索 |
| 用户未登录查业务数据 | Tool 内部检测 auth 上下文，返回友好登录引导 |

---

## 七、Redis 数据结构

| Key | 类型 | 内容 | TTL |
|-----|------|------|-----|
| `chat:history:{sessionId}` | List | `{"role":"user","text":"...","time":"2026-..."}` | 24h |
| `item:vector:{itemId}` | String (bytes) | float32 向量二进制 | 24h |
| `item:vector:meta:{itemId}` | String (JSON) | `{"id":"","name":"","price":0,"image":""}` | 24h |
| `kb:vector:{hash}` | String (bytes) | 知识库条目向量 | 7d |
| `kb:vector:meta:{hash}` | String (JSON) | `{"question":"","answer":"","keywords":[]}` | 7d |

---

## 八、与 Java 端关系

### 8.1 Python 替代的 Java 组件

| Java 组件 | Python 替代 |
|-----------|------------|
| AiChatService (对话) | LangChain Agent + Tool Calling |
| AiReviewService (审核) | review_service.py |
| DeepSeekService (API调用) | ChatOpenAI / deepseek_service.py |
| IntentParser (意图识别) | Agent 自主 Tool Calling |

### 8.2 Python 不需要的

| 组件 | 原因 |
|------|------|
| Nacos 注册 | 网关直连 `http://localhost:8095` |
| JWT 解析 | 网关已解析，注入 `user-id` header |
| Feign | httpx 走 HTTP 网关 |
| MySQL | 对话历史全存 Redis |

### 8.3 网关配置

```yaml
# ym-gateway/application.yaml
- id: yunmeng-ai
  uri: http://localhost:8095   # 直连 Python
```

---

## 九、调试指南

### 9.1 直接测试 Agent

```bash
python _test_agent.py
# 1. Agent 初始化 → AgentExecutor with 5 tools
# 2. Agent 调用   → DeepSeek 返回中文回复
# 3. 带历史调用   → 多轮对话 + Redis 记忆
```

### 9.2 模块导入测试

```bash
python _test_imports.py
# 验证全部 21 个模块正常加载
```

### 9.3 API 接口测试

```bash
python _test_api.py
# 1. Health        → 健康检查
# 2. Vector status → 索引状态
# 3. Chat (闲聊)   → Agent 自我介绍
# 4. Chat (知识库) → Agent 触发 query_knowledge_base
# 5. Chat (订单)   → Agent 触发 query_user_orders
```

### 9.4 curl 直接调试

```bash
# 绕过网关
curl -X POST http://localhost:8095/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"怎么退货","sessionId":""}'

# 模拟登录用户
curl -X POST http://localhost:8095/chat \
  -H "Content-Type: application/json" \
  -H "user-id: 2" \
  -H "Authorization: Bearer <JWT>" \
  -d '{"message":"我的订单","sessionId":""}'
```

### 9.5 开启 LangChain 详细日志

```bash
# .env 中设置
LANGCHAIN_VERBOSE=true
```

---

## 十、常见问题

**Q: Agent 没有调用我预期的 Tool？**

检查 Tool 的 docstring 是否准确描述了触发场景。docstring 是 LLM 决策的唯一依据。

**Q: 如何让 Agent 支持新的业务功能？**

只需在 `tools.py` 中添加一个 `@tool` 函数并加入 `ALL_TOOLS` 列表即可。Agent 会自动识别。

**Q: 对话历史为什么不正确？**

确认 Redis 连接正常。`RedisChatMessageHistory` 的 key 格式是 `chat:history:{sessionId}`，与 Java 端一致。

**Q: DeepSeek 返回空或超时？**

检查 `DEEPSEEK_API_KEY` 有效性。Agent 异常时 `ai_service.handle_chat()` 会返回降级提示。

**Q: 向量检索不生效？**

先调用 `POST /vector/index/all` 构建索引。确认 Redis 中有 `item:vector:*` 和 `kb:vector:*` 的 key。
