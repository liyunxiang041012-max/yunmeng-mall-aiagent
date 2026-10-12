# 云梦 AI 助手 · yunmeng-mall-aiagent

> 云梦商城的智能客服与 AI 能力中台，基于 **FastAPI + LangChain + DeepSeek** 构建，为商城提供对话式导购、订单/优惠券/购物车查询、语义商品检索、常见问题问答与 AI 商品审核等能力。

---

## ✨ 核心特性

- **🤖 Agent 自主决策**：基于 LangChain `create_tool_calling_agent`，由大模型根据用户意图自主决定是否调用工具、调用哪个工具，替代传统关键词硬编码路由。
- **🛠️ 工具调用（Tool Calling）**：内置 5 个业务工具，覆盖订单、优惠券、购物车、商品搜索、知识库问答。
- **🔍 向量语义检索（RAG）**：使用 `BAAI/bge-small-zh-v1.5` 中文向量模型，对商品与知识库进行向量化，支持语义搜索并按匹配度阈值召回。
- **💬 SSE 流式输出**：`/chat/stream` 通过 Server-Sent Events 逐字返回，并实时推送工具调用状态（`tool_start` / `tool_end`）。
- **🧠 会话记忆**：基于 Redis 的多轮对话记忆，按 `session_id` 隔离，支持 TTL 过期与历史清理。
- **🛡️ AI 商品审核**：采用「本地规则预筛 + 大模型语义审核」两段式策略，兼顾成本与准确率。
- **🔌 网关集成**：作为微服务挂载于 Spring Cloud Gateway 之后（`StripPrefix=1`，路径不带 `/ai` 前缀）。

---

## 🏗️ 技术栈

| 分类 | 技术选型 |
| --- | --- |
| Web 框架 | FastAPI + Uvicorn |
| AI 编排 | LangChain / LangChain-OpenAI / LangChain-Community |
| 大模型 | DeepSeek（`deepseek-chat`，OpenAI 兼容协议） |
| 向量模型 | sentence-transformers（`BAAI/bge-small-zh-v1.5`） |
| 缓存/存储 | Redis（会话记忆 + 向量索引持久化） |
| HTTP 客户端 | httpx（异步调用 Java 业务服务） |
| 配置管理 | python-dotenv（`.env`） |

---

## 📁 项目结构

```
python-ai/
├── main.py                 # FastAPI 应用入口，注册路由与生命周期
├── config.py               # 配置加载（环境变量 / .env）
├── requirements.txt        # Python 依赖
├── routers/                # 路由层
│   ├── chat.py             #   聊天：/chat, /chat/stream, /history
│   ├── review.py           #   商品审核：/review/item
│   └── vector.py           #   向量索引管理：/vector/*
├── services/               # 业务服务层
│   ├── ai_service.py       #   聊天业务编排
│   ├── review_service.py   #   AI 商品审核（两段式）
│   ├── deepseek_service.py #   DeepSeek 调用封装
│   ├── embedding_service.py#   向量化服务
│   └── langchain/          #   LangChain 相关
│       ├── chain.py        #     Agent + Memory 流水线组装
│       ├── tools.py        #     工具定义（Tool Calling）
│       ├── chat_model.py   #     LLM 构建
│       ├── prompts.py      #     提示词模板
│       ├── memory.py       #     Redis 会话记忆
│       ├── retriever.py    #     检索器
│       └── embeddings.py   #     向量模型封装
├── models/                 # 请求 / 响应数据模型
├── data/knowledge.py       # 商城常见问题知识库
└── utils/                  # 工具类
    ├── redis_client.py     #   Redis 连接
    ├── http_client.py      #   调用 Java 业务服务
    └── vector_store.py     #   向量索引与检索
```

---

## 🚀 快速开始

### 1. 环境要求

- Python **3.10+**
- Redis 服务（用于会话记忆与向量索引）
- DeepSeek API Key
- （可选）云梦商城 Java 后端服务，用于拉取真实商品数据

### 2. 安装依赖

```bash
# 创建并激活虚拟环境
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

复制示例配置并填入真实值：

```bash
cp .env.example .env
```

`.env` 关键项说明：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek 密钥（必填） | - |
| `DEEPSEEK_BASE_URL` | 接口地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 模型名 | `deepseek-chat` |
| `REDIS_HOST` / `REDIS_PORT` | Redis 地址 | `192.168.150.101` / `6379` |
| `REDIS_PASSWORD` / `REDIS_DB` | Redis 认证与库号 | - / `0` |
| `GATEWAY_URL` | Java 业务服务网关地址 | `http://localhost:8080` |
| `EMBEDDING_MODEL` | 向量模型 | `BAAI/bge-small-zh-v1.5` |
| `VECTOR_TOP_K` / `VECTOR_MIN_SCORE` | 召回数量 / 最低匹配度 | `5` / `0.3` |
| `CHAT_HISTORY_TTL_HOURS` | 会话记忆过期时间（小时） | `24` |
| `JWT_SECRET` | JWT 校验密钥 | - |

> ⚠️ **安全提示**：`.env` 含敏感密钥，已被 `.gitignore` 排除，**切勿提交到仓库**。

### 4. 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8095 --reload
```

启动后：

- 健康检查：<http://localhost:8095/health>
- 交互式 API 文档（Swagger）：<http://localhost:8095/docs>

### 5. 构建向量索引（首次部署必做）

```bash
# 一键构建：商品 + 知识库
curl -X POST http://localhost:8095/vector/index/all
```

---

## 📡 API 接口

> 说明：以下路径为服务内部路径。经网关访问时会带 `/ai` 前缀（网关 `StripPrefix=1` 会剥离该前缀）。

### AI 聊天

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/chat` | 发送消息（Agent 自主决策 + Tool Calling，非流式） |
| `POST` | `/chat/stream` | 发送消息（SSE 流式输出，含工具调用状态） |
| `GET` | `/history?sessionId=xxx` | 获取对话历史 |
| `DELETE` | `/history/clear?sessionId=xxx` | 清除对话历史 |

**请求头**：`Authorization`（登录态）、`user-id`（用户 ID）。

**流式事件类型**：`token`（正文分片）、`tool`（工具开始/结束）、`done`（结束）、`error`（异常）。

### AI 商品审核

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/review/item` | 审核商品（名称/价格/库存），返回是否合规及原因 |

### 向量检索

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/vector/index` | 构建商品向量索引（从 Java 服务拉取全量商品） |
| `POST` | `/vector/index/knowledge` | 构建知识库向量索引 |
| `POST` | `/vector/index/all` | 一键构建全部索引（商品 + 知识库） |
| `GET` | `/vector/status` | 查询索引状态 |
| `DELETE` | `/vector/index` | 清空向量索引 |

---

## 🧩 Agent 工具集

大模型根据每个工具的 `docstring` 自主判断何时调用：

| 工具 | 能力 | 是否需登录 |
| --- | --- | --- |
| `query_user_orders` | 查询用户订单列表与物流状态 | 是 |
| `query_user_coupons` | 查询用户可用优惠券 | 是 |
| `query_user_cart` | 查询购物车商品与合计金额 | 是 |
| `search_mall_products` | 语义搜索 / 推荐商品（向量检索优先，关键词降级） | 否 |
| `query_knowledge_base` | 查询商城常见问题（退货、支付、领券等） | 否 |

---

## 🔒 安全说明

- `.env`、`.venv/`、`__pycache__/` 等敏感与临时文件已通过 `.gitignore` 排除。
- 请勿将 `DEEPSEEK_API_KEY`、`REDIS_PASSWORD`、`JWT_SECRET` 等真实密钥提交到版本库。
- 如需分享配置，请仅提交 `.env.example` 占位模板。

---

## 📄 License

本项目为云梦商城课程/学习项目的一部分，仅供学习交流使用。
