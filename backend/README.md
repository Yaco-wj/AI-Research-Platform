# 平台后端（第一阶段）

Backend 是 Deep Research 平台的**服务层**，只负责任务管理、SSE 推送与数据库持久化，
研究逻辑全部由 Agent（`src/open_deep_research/`）完成。

```
Frontend ──HTTP/SSE──▶ Backend ──Agent Adapter──▶ Deep Research Agent
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
      PostgreSQL 业务表        LangGraph Checkpoint
        research_tasks            PostgresSaver
```

## 目录结构

```
backend/
├── app/
│   ├── main.py                     # FastAPI 入口 + lifespan（初始化 DB / checkpoint / adapter / 孤儿任务恢复）
│   ├── dependencies.py             # 依赖注入：会话、事件总线、适配器、服务
│   ├── api/
│   │   ├── health.py               # GET /api/v1/health
│   │   └── research.py             # 研究任务相关接口（含只读 SSE 订阅）
│   ├── service/
│   │   ├── research_service.py     # 应用服务：创建 / 查询 / 恢复 / 取报告 / 事件回放
│   │   ├── task_service.py         # 后台执行器：任务生命周期唯一 owner（落库 + 事件发布 + 恢复）
│   │   └── event_bus.py            # 进程内实时事件总线（广播 + 快照回放）
│   ├── repository/
│   │   └── research_task_repository.py
│   ├── models/
│   │   ├── base.py
│   │   └── research_task.py        # research_tasks 表 + research_task_events 事件日志表
│   ├── schemas/
│   │   └── research.py             # Pydantic 请求 / 响应模型
│   ├── agent_adapter/
│   │   └── research_agent.py       # Backend ↔ Agent 的唯一适配层
│   └── core/
│       ├── config.py               # 环境变量配置
│       ├── database.py             # SQLAlchemy 2.x + asyncpg（业务库）
│       └── checkpoint.py           # AsyncPostgresSaver（LangGraph checkpoint）
├── alembic/                        # 数据库迁移
├── tests/                          # 全部使用假 Agent / mock adapter
├── pyproject.toml
└── .env.example
```

## 环境准备

平台后端与 Agent 共用仓库根目录的虚拟环境（该环境已包含 Agent 依赖）：

```bash
# 仓库根目录
uv venv                       # 若尚未创建
uv sync --extra dev --extra backend          # 一次安装 Agent 与平台层依赖
```

主要新增依赖：`fastapi`、`uvicorn`、`sqlalchemy[asyncio]`、`asyncpg`、`alembic`、
`langgraph-checkpoint-postgres`、`psycopg[binary,pool]`、`pydantic-settings`。

## 环境变量

```bash
cd backend
cp .env.example .env       # 然后填入真实的数据库密码与模型 key
```

| 变量 | 说明 |
| --- | --- |
| `DATABASE_URL` | 业务数据库（SQLAlchemy + asyncpg），如 `postgresql+asyncpg://postgres:密码@localhost:5432/deep_research` |
| `LANGGRAPH_DATABASE_URL` | LangGraph checkpoint（psycopg 连接串，**不能带 `+asyncpg`**）|
| `API_PREFIX` | 接口前缀，默认 `/api/v1` |
| `CORS_ORIGINS` | 允许的前端来源，默认 `["http://localhost:5173"]` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `DEEPSEEK_API_KEY` / `TAVILY_API_KEY` | 透传给 Agent 的模型与搜索密钥 |

> `backend/.env` 中的模型配置会透传给 Agent；未设置时会回退读取仓库根目录的 `.env`。

## 数据库初始化

PostgreSQL 同时承载两类数据：**业务表**（`research_tasks`）和 **LangGraph checkpoint 表**。
两者互相独立：

- 业务表由 Alembic 管理；
- checkpoint 表由 `AsyncPostgresSaver.setup()` 在应用启动时自动创建。

官方推荐：**Alembic 命令一律在 `backend/` 目录下执行**。

```bash
cd backend

# 1. 配置连接串（只需一次）
cp .env.example .env          # 填写 DATABASE_URL / LANGGRAPH_DATABASE_URL

# 2. 业务表迁移（幂等，可反复执行）
alembic upgrade head

# 3. 查看当前版本
alembic current

# 4. 回滚 / 重建（需要时）
alembic downgrade base
alembic upgrade head

# 5. 只查看将要执行的 SQL（不连库）
alembic upgrade head --sql
```

关于路径解析（无需手工设置 `PYTHONPATH`）：

- `backend/alembic.ini` 使用 `%(here)s`（即 `alembic.ini` 所在目录）声明
  `script_location = %(here)s/alembic` 与 `prepend_sys_path = %(here)s`，
  所以迁移脚本目录与导入路径都不再依赖当前工作目录；
- `backend/alembic/env.py` 会按自身文件位置把仓库根目录和 `backend/` 加入
  `sys.path`，后端代码统一以 `backend.app.xxx` 导入；
- 因此以下两种方式都可用，前者是官方推荐方式：

```bash
cd backend && alembic upgrade head              # 官方推荐
alembic -c backend/alembic.ini upgrade head     # 从仓库根目录执行也可以
```

## 启动

```bash
cd backend

# 方式一（推荐，Windows / Linux 通用）
python run.py

# 方式二（开发热重载）
uvicorn app.main:app --reload --port 8000
```

- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/api/v1/health

> **Windows 注意**：Windows 默认的 `ProactorEventLoop` 不被 psycopg（PostgresSaver 的驱动）的
> 异步模式支持。`python run.py` 会先切换到 `SelectorEventLoop`；
> 直接使用 `uvicorn app.main:app --reload` 也可以（uvicorn 在 `--reload` / 多 worker 模式下会切换）。
> 若在 Windows 上使用 `uvicorn app.main:app`（不带 `--reload`），应用会以明确的报错提示启动失败。

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/health` | 健康检查，返回 `{"status": "ok"}` |
| POST | `/api/v1/research` | 创建研究任务，立即返回 `PENDING`（后台异步执行）|
| GET | `/api/v1/research` | 研究历史列表（`limit` / `offset`）|
| GET | `/api/v1/research/{task_id}` | 任务详情与状态 |
| GET | `/api/v1/research/{task_id}/report` | 最终报告（未完成返回 409）|
| GET | `/api/v1/research/{task_id}/stream` | SSE 流式研究过程（支持 `since_id` 断点续传）|
| POST | `/api/v1/research/{task_id}/resume` | 回答澄清问题并恢复研究 |

### 任务状态机

```
PENDING ──▶ RUNNING ──▶ COMPLETED
                │
                ├──▶ WAITING_USER ──(POST /resume)──▶ RUNNING
                └──▶ FAILED
```

状态流转只有 `TaskService` 一个 owner（API 层只读），因此不会出现并发写状态。

**不会出现“永远 RUNNING”**：

- 客户端断开 SSE 只影响订阅，不影响后台研究；
- 进程正常关闭时 `TaskService.shutdown()` 取消后台任务并写入 `FAILED`；
- 进程崩溃 / 被 kill 时，下次启动由 `recover_orphaned_tasks()` 依据 LangGraph checkpoint
  把残留的 `PENDING` / `RUNNING` 修正为 `COMPLETED`（已有报告）、`WAITING_USER`（停在 interrupt）
  或 `FAILED`（其余情况）。

### SSE 事件

连接地址：`GET /api/v1/research/{task_id}/stream`（`text/event-stream`）。
事件来自 Agent 的真实执行过程，不做任何伪造：

| event | 说明 |
| --- | --- |
| `status` | 任务状态变化（来自业务库）|
| `node_started` / `node_completed` | LangGraph 节点开始 / 结束（含 `scope`：main / supervisor / researcher）|
| `research_started` | 主管派发了一个研究子课题（`topic` 来自 researcher 任务输入）|
| `research_completed` | 子课题研究结果压缩完成（`topic` + 摘要）|
| `clarification_required` | Agent interrupt，前端展示澄清问题 |
| `report_chunk` | 最终报告内容（当前 Agent 一次性生成，因此只有一个完整分片）|
| `completed` | 任务完成 |
| `error` | 任务失败（只含友好文案，不含 traceback）|
| `stream_unavailable` | 当前进程没有该任务的实时事件流，且数据库事件日志也没有可回放事件 |

> SSE 只是订阅者：回放来源是 `research_task_events` 事件日志（先回放历史，再推送实时事件，
> 通过同一个自增 id 去重），因此重新进入详情页 / 服务重启后依然能恢复当前状态、
> 已完成阶段、clarification 信息与 research timeline。
> `since_id` 用于断点续传：重连不会重复推送已处理事件，也**不会重新启动 Agent**。

### interrupt / resume

1. Agent 在 `wait_for_user_clarification` 节点调用 `interrupt({...})`；
2. Backend 捕获后把任务置为 `WAITING_USER` 并推送 `clarification_required`，同时结束当前 SSE 连接；
3. 前端提交答案 → `POST /api/v1/research/{task_id}/resume`；
4. Backend 使用**同一个 `thread_id`** 调用 `Command(resume=answer)` 继续执行，任务回到 `RUNNING`，前端重新连接 SSE；
5. 完成后写入 `final_report`，状态变为 `COMPLETED`。

## 测试

```bash
cd backend
pytest -q
```

测试全部使用 `FakeResearchAgentAdapter` + SQLite，不会调用真实 DeepSeek / Tavily API。

## 第一阶段边界

- 后台任务使用进程内 `asyncio` 任务（未引入 Redis / Celery），进程重启后**不会自动续跑**；
  但 LangGraph checkpoint 与业务数据库都保留了完整状态，重启时会把残留任务修正为
  `FAILED` / `WAITING_USER` / `COMPLETED`（不会留下假的 `RUNNING`），
  后续接入外部队列后可以在此基础上实现“自动续跑”；
- 研究过程事件已持久化到 `research_task_events`，重启后仍可完整回放研究时间线
  （进程内事件总线只负责实时广播，不再是回放的唯一来源）；
- 上述「孤儿任务恢复」假设**单副本部署**（进程内任务无法跨进程存活）；
  若要多副本/多 worker，需要给任务增加 owner / lease（心跳时间戳）字段，
  或直接换成 Redis / Celery 等外部队列，避免一个副本误判另一个副本正在跑的任务；
- 无用户体系与权限控制，接口不鉴权；
- 报告当前由 Agent 一次性生成，因此 `report_chunk` 只有一个分片。
