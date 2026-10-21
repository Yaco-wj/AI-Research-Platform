# AI Research Platform

AI Research Platform 是一个带 Web 界面的深度研究应用。用户提交研究问题后，平台会调用基于 LangGraph 的 Deep Research Agent，完成需求澄清、问题拆解、并发子课题研究、资料压缩和最终报告生成，并通过 SSE 将研究进度实时推送到前端。

本项目适合以下场景：

- 对需要多渠道检索和结构化整理的主题进行深度调研；
- 演示 FastAPI、React、LangGraph 与 PostgreSQL 的完整协作；
- 作为 Agent 工作流、任务持久化、SSE 流式输出和 interrupt/resume 的学习项目；
- 在单机或单副本环境中搭建可交互的研究原型。

## 功能特性

- **研究需求澄清**：Agent 先判断问题是否足够明确；需要补充信息时，通过 LangGraph interrupt 暂停，并在前端显示澄清问题。
- **问题拆解与研究简报**：将用户需求拆成可独立执行的子问题，同时生成 `research_brief` 和 `sub_questions`。
- **Supervisor / Researcher 协作**：Supervisor 负责派发、去重和判断研究是否结束；Researcher 围绕单个子课题循环调用工具、反思并压缩结果。
- **并发研究**：一组子问题可并发启动 Researcher；批次数量和单批并发数分别由配置限制。
- **搜索与 MCP**：实际支持 Tavily、OpenAI/Anthropic 原生网页搜索，并可加载配置中指定的 MCP 工具。
- **研究报告生成**：根据研究笔记和研究简报生成 Markdown 报告，要求保留来源引用。
- **后台任务管理**：FastAPI 创建任务后立即返回，研究在进程内 `asyncio` 后台任务中继续运行。
- **SSE 实时进度**：推送节点状态、子课题开始/完成、澄清问题、报告和任务终态；支持 `since_id` 断点续传。
- **事件持久化**：研究过程写入 `research_task_events`，重新进入详情页或服务重启后仍可回放已有事件。
- **LangGraph Checkpoint**：使用 PostgreSQL 保存图状态，支持同一 `thread_id` 上的 interrupt/resume。
- **前端交互**：支持新建研究、查看历史、观察执行阶段和子课题、回答澄清问题、查看 Markdown 报告。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | React 19、TypeScript、Vite 7、React Router、EventSource、react-markdown |
| 后端 | FastAPI、Uvicorn、Pydantic 2、SQLAlchemy 2 异步 ORM、Alembic、sse-starlette |
| Agent 工作流 | LangGraph、LangChain、Supervisor/Researcher 子图、interrupt/resume |
| 数据库 | PostgreSQL；业务数据与 LangGraph checkpoint 分表/分连接管理 |
| 搜索与模型 | Tavily、OpenAI、Anthropic、Google、DeepSeek、Groq、AWS 的 LangChain 集成；可选 MCP |
| 部署与工程化 | uv、npm、Docker Compose（仅 PostgreSQL）、pytest、TypeScript 类型检查、Vite 构建 |

当前代码没有使用 Redis、pgvector、Celery 或消息队列。后台任务由 API 进程内的 `asyncio.Task` 管理。

## 系统架构

```mermaid
flowchart LR
    U[浏览器用户] --> F[React + Vite 前端]
    F -->|REST /api/v1| B[FastAPI 后端]
    F <-->|SSE /research/{id}/stream| B
    B --> TS[TaskService 后台任务]
    TS --> A[ResearchAgentAdapter]
    A --> G[LangGraph Deep Researcher]

    G --> C[Clarify 与问题拆解]
    C --> S[Supervisor]
    S --> R1[Researcher 子图 1]
    S --> R2[Researcher 子图 N]
    R1 --> T[Tavily / 原生搜索 / MCP]
    R2 --> T
    S --> W[最终报告生成]

    B --> DB[(PostgreSQL 业务表)]
    A --> CP[(PostgreSQL Checkpoint)]
    B --> BUS[进程内事件总线]
    BUS -->|实时 SSE| F
    DB -->|事件回放| BUS
```

一次研究任务的主要调用顺序如下：

1. 前端调用 `POST /api/v1/research` 创建任务，后端立即返回 `task_id`。
2. `TaskService` 在后台驱动 `ResearchAgentAdapter`，Agent 的 Graph 状态由 PostgreSQL checkpoint 保存。
3. Agent 执行澄清、问题拆解、Supervisor 调度、Researcher 工具调用和报告生成。
4. 后端把 Agent 事件转换为平台事件，先写入 `research_task_events`，再通过进程内事件总线广播。
5. 前端通过 EventSource 消费 SSE；事件流断开不会取消后台研究。

## 项目目录

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI 入口与应用生命周期
│   │   ├── api/                       # 健康检查、研究任务 API、SSE
│   │   ├── agent_adapter/             # Backend 与 LangGraph Agent 的唯一适配层
│   │   ├── core/                      # 配置、业务数据库、PostgresSaver
│   │   ├── models/                    # 任务与事件 SQLAlchemy 模型
│   │   ├── repository/                # 任务/事件持久化
│   │   ├── schemas/                   # Pydantic API 模型
│   │   └── service/                   # 应用服务、后台任务、事件总线
│   ├── alembic/versions/              # 业务数据库迁移
│   ├── tests/                         # 后端 API、任务生命周期、SSE 测试
│   ├── alembic.ini
│   ├── pyproject.toml
│   └── run.py                         # Windows 友好的后端启动入口
├── frontend/
│   ├── src/
│   │   ├── api/client.ts              # REST 与 SSE 地址封装
│   │   ├── hooks/useResearchStream.ts # SSE 事件到 UI 状态的转换
│   │   ├── pages/                     # 首页与研究详情页
│   │   ├── components/                # 输入、历史、时间线、澄清、报告
│   │   └── types/research.ts          # 与后端 Schema 对应的类型
│   ├── package.json
│   └── vite.config.ts
├── src/
│   ├── open_deep_research/            # Agent 主图、状态、提示词、配置与工具
│   └── security/auth.py               # LangGraph 部署的可选 Supabase 鉴权
├── tests/
│   ├── test_deep_researcher_graph.py  # 使用假模型/假工具的 Agent 测试
│   └── evaluators.py 等               # 可选的 LangSmith 评估工具
├── compose.yaml                       # 本地 PostgreSQL
├── main.py                            # Agent 命令行演示
├── langgraph.json                     # LangGraph 部署/Studio 配置
├── pyproject.toml
├── uv.lock
├── .env.example
└── LICENSE
```

## 环境要求

- Python `>=3.10`；仓库的 LangGraph 部署配置使用 Python 3.11，建议安装 Python 3.11 或更高版本。
- Node.js `^20.19.0 || >=22.12.0`，由 Vite 7 的版本要求决定。
- PostgreSQL。`compose.yaml` 使用 `postgres:16-alpine`。
- 可用性的模型 API Key 和搜索 API Key。

本项目不需要单独安装 pgvector。LangGraph checkpoint 的表由 `AsyncPostgresSaver.setup()` 在后端启动时创建；业务表由 Alembic 创建。

## 环境变量

配置分为两个本地文件，二者都已被 `.gitignore` 排除：

- 根目录 `.env`：模型、搜索、MCP、LangSmith 与 Agent 行为配置；
- `backend/.env`：数据库、CORS、日志和 checkpoint 连接池配置。

先从模板复制，再填写自己的值：

```powershell
Copy-Item .env.example .env
Copy-Item backend/.env.example backend/.env
```

常用变量：

| 变量 | 用途 |
| --- | --- |
| `DATABASE_URL` | 业务库异步连接串，必须使用 `postgresql+asyncpg://...` |
| `LANGGRAPH_DATABASE_URL` | LangGraph checkpoint 连接串，必须使用 `postgresql://...`，不能带 `+asyncpg` |
| `OPENAI_API_KEY` | OpenAI 模型或 OpenAI 兼容服务密钥 |
| `OPENAI_BASE_URL` | 使用 OpenAI 兼容接口时填写服务地址；标准 OpenAI 服务可留空 |
| `DEEPSEEK_API_KEY` | 使用 DeepSeek 模型时的密钥 |
| `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | 使用对应模型提供方时的密钥 |
| `TAVILY_API_KEY` | 默认搜索工具 Tavily 的密钥 |
| `SEARCH_API` | `tavily`、`openai`、`anthropic` 或 `none` |
| `RESEARCH_MODEL` | 澄清、拆解、Supervisor 与 Researcher 使用的模型 |
| `COMPRESSION_MODEL` | 子研究结果压缩模型 |
| `FINAL_REPORT_MODEL` | 最终报告模型 |
| `SUMMARIZATION_MODEL` | Tavily 网页正文摘要模型 |
| `ALLOW_CLARIFICATION` | 是否允许 Agent 发起澄清问题 |
| `MAX_CONCURRENT_RESEARCH_UNITS` | 单批最大并发研究数 |
| `MAX_RESEARCHER_ITERATIONS` | Supervisor 最大研究批次数 |
| `MAX_REACT_TOOL_CALLS` | 单个 Researcher 的最大工具调用轮次 |
| `MCP_CONFIG` | MCP 连接 JSON，例如 `{"url":"https://example.com","tools":["search"],"auth_required":false}` |
| `MCP_PROMPT` | 追加到 Researcher 提示词中的 MCP 工具说明 |
| `CORS_ORIGINS` | 后端允许的前端来源，JSON 数组格式 |

模型名可使用 `provider:model` 形式。代码默认配置为 OpenAI `gpt-4.1` 系列；也可使用项目已安装的其他 LangChain provider，例如：

```dotenv
DEEPSEEK_API_KEY=your-key
RESEARCH_MODEL=deepseek-chat
COMPRESSION_MODEL=deepseek-chat
FINAL_REPORT_MODEL=deepseek-chat
```

不要把真实 API Key、数据库密码、Supabase Token 或私钥写入 `.env.example`、README 或 Git 提交。

## 启动方式

以下命令以已经安装 `uv`、Node.js 和 Docker Desktop 为前提。

### 1. 安装 Python 依赖

在仓库根目录执行：

```powershell
uv sync --extra dev --extra backend
```

`uv sync` 使用根目录 `uv.lock` 统一安装 Agent、前端开发工具与 FastAPI 服务层依赖。后端源码通过仓库根目录导入，不需要把 `backend/` 作为独立 Python 包安装。

### 2. 启动 PostgreSQL

```powershell
docker compose up -d postgres
docker compose ps
```

默认数据库、用户名和密码分别为 `deep_research`、`postgres`、`postgres`，对应连接串已经写入 `backend/.env.example`。如果修改了 `POSTGRES_*`，必须同步修改 `DATABASE_URL` 和 `LANGGRAPH_DATABASE_URL`。

### 3. 执行数据库迁移

```powershell
Set-Location backend
..\.venv\Scripts\Activate.ps1
alembic upgrade head
alembic current
```

`alembic upgrade head` 创建 `research_tasks` 与 `research_task_events`。LangGraph checkpoint 相关表在首次启动 FastAPI 时由应用自动创建。

### 4. 启动后端

```powershell
Set-Location backend
..\.venv\Scripts\Activate.ps1
python run.py
```

默认监听 `127.0.0.1:8000`：

- Swagger：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/api/v1/health`

Linux/macOS 可执行 `source .venv/bin/activate` 后运行相同的 `python run.py`。

Windows 上不要直接使用会落到 ProactorEventLoop 的无 reload Uvicorn 启动方式；`python run.py` 会预先切换到 `SelectorEventLoop`。开发时也可以从仓库根目录使用 `uvicorn backend.app.main:app --reload --port 8000`。

### 5. 启动前端

打开另一个终端：

```powershell
Set-Location frontend
npm ci
npm run dev
```

浏览器访问 `http://localhost:5173`。Vite 开发服务器已将 `/api` 代理到 `http://localhost:8000`；如需连接其他后端地址，可设置 `VITE_API_BASE_URL`。

### 6. 不依赖 Docker 启动 PostgreSQL

也可以使用已有 PostgreSQL 实例，但需要：

1. 创建数据库；
2. 在 `backend/.env` 中设置两种连接串；
3. 执行 `alembic upgrade head`。

## 使用说明

1. 在首页输入研究问题，点击“开始研究”或按 `Ctrl/Cmd + Enter`。
2. 页面进入 `/research/{task_id}`，展示任务状态、执行阶段和研究子课题。
3. 如果 Agent 认为需求不够明确，任务进入“等待澄清”；在左侧面板补充范围、对象或重点后提交。
4. 后端使用相同 `thread_id` 和 `Command(resume=answer)` 恢复同一个 LangGraph 任务。
5. 研究完成后，右侧“最终报告”区域渲染 Markdown 报告；首页可查看最近的研究记录。

任务状态流转：

```text
PENDING -> RUNNING -> COMPLETED
                |
                +-> WAITING_USER --回答澄清--> RUNNING
                |
                +-> FAILED
```

### API 速览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/research` | 创建研究任务，请求体为 `{"query":"..."}` |
| `GET` | `/api/v1/research` | 查询研究历史，支持 `limit`、`offset` |
| `GET` | `/api/v1/research/{task_id}` | 查询任务详情与状态 |
| `GET` | `/api/v1/research/{task_id}/report` | 获取最终 Markdown 报告 |
| `POST` | `/api/v1/research/{task_id}/resume` | 提交澄清回答，请求体为 `{"answer":"..."}` |
| `GET` | `/api/v1/research/{task_id}/stream` | 订阅 SSE，可选 `since_id` |

SSE 事件包括 `status`、`node_started`、`node_completed`、`research_started`、`research_completed`、`clarification_required`、`report_chunk`、`completed`、`error` 和 `stream_unavailable`。当前最终报告由一次模型调用生成，因此 `report_chunk` 通常只有一个完整分片。

## 测试

Agent 单元测试全部使用假模型和假工具，不会调用真实模型或搜索 API：

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_deep_researcher_graph.py
```

后端测试使用 SQLite 和假 Agent 适配器：

```powershell
.venv\Scripts\python.exe -m pytest -q backend\tests
```

前端检查与生产构建：

```powershell
Set-Location frontend
npm run typecheck
npm run build
```

检查依赖锁文件：

```powershell
uv lock --check
```

`tests/` 中的 LangSmith 评估脚本是可选手工工具，需要单独配置数据集和外部 API，不属于默认离线测试。

## 常见问题

### 数据库连接失败

- 确认 PostgreSQL 容器已经启动：`docker compose ps`。
- 业务库必须使用 `postgresql+asyncpg://`，checkpoint 必须使用 `postgresql://`。
- 检查主机、端口、数据库名、用户名和密码是否与 Compose 配置一致。
- 业务表不存在时，在 `backend/` 目录执行 `alembic upgrade head`。

### Windows 报 ProactorEventLoop 或 psycopg 错误

使用 `python run.py`，不要在 Windows 上以不支持的事件循环启动后端。后端会在启动时显式切换到 `SelectorEventLoop`。

### 模型或搜索 API 报错

- 检查模型名与已安装的 provider 依赖是否匹配；
- 检查对应的 API Key 是否非空、额度是否可用；
- 检查 `SEARCH_API` 与实际配置的搜索服务是否一致；
- 将 `LANGSMITH_TRACING=false` 保持关闭可避免未配置 LangSmith 时产生额外请求。

### pytest 出现临时目录 PermissionError

某些 Windows 账户无法访问系统临时目录。可把测试临时目录切到项目内：

```powershell
$env:TMP = Join-Path $PWD ".tmp\pytest"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force $env:TMP
.venv\Scripts\python.exe -m pytest -q backend\tests
```

### 刷新详情页后没有实时进度

SSE 会先回放数据库中的事件日志。若服务进程已经重启，当前进程不再持有原来的实时 EventSource；前端会通过任务详情接口读取权威状态。已有事件仍可回放，但进程内任务不会自动续跑。

## 当前边界

- 后台任务使用单进程 `asyncio`，不可跨进程自动恢复；当前实现适合单副本部署。
- FastAPI 平台接口当前没有用户体系或鉴权；`src/security/auth.py` 仅用于 LangGraph 部署的可选 Supabase 鉴权。
- 没有 Redis/Celery 任务队列，也没有 pgvector 向量检索。
- 研究报告不是逐 token 流式生成，而是在报告节点结束后作为一个完整 SSE 分片返回。

## 许可证与项目来源

本项目保留了原项目的 MIT `LICENSE`，版权声明为 `Copyright (c) 2025 LangChain`。

Deep Research Agent 核心及部分提示词、配置和测试基于 LangChain 的开源项目 [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)。本仓库在此基础上增加了 FastAPI 平台层、PostgreSQL 任务与事件持久化、SSE 订阅/回放、React 前端、中文提示词与相关测试，并清理了未使用的旧实现。原始开源成果仍归原作者及贡献者所有；二次使用和分发时请遵守 [MIT License](LICENSE)。
