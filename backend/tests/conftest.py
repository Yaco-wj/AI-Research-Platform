"""测试公共设施：假 Agent 适配器、SQLite 业务库、可注入的 FastAPI 应用。

所有测试都不调用真实的 DeepSeek / Tavily API：
Agent 一律使用 FakeResearchAgentAdapter 替代，
checkpoint 使用内存 checkpointer 或完全不使用。
"""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import asyncio
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# 平台代码统一以 `backend.app.xxx` 导入，因此仓库根目录必须在 sys.path 上。
# 这里按 conftest 自身位置补齐，保证 `cd backend && pytest -q`
# 与 `pytest backend/tests` 两种方式都能正确导入。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.agent_adapter.research_agent import AgentEvent, AgentStateView
from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.models import Base

DEFAULT_REPORT = "# 2026 年 AI Agent 技术发展趋势研究报告\n\n假报告内容：企业级落地是重点。"


@pytest.fixture(autouse=True)
def reset_sse_app_status():
    """sse-starlette 的 AppStatus 是模块级单例，跨事件循环测试前需要重置。"""
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit = False
    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None


def make_event(event_type: str, **kwargs) -> AgentEvent:
    """构造一条 AgentEvent。"""
    return AgentEvent(type=event_type, **kwargs)


@dataclass
class FakeScript:
    """一次 Agent 调用的脚本：要么产出事件，要么抛出异常。"""

    events: list[AgentEvent] = field(default_factory=list)
    error: Exception | None = None


def default_initial_script() -> FakeScript:
    """默认的首次研究脚本：走完节点、派发一个研究员并产出报告。"""
    return FakeScript(
        events=[
            make_event("node_started", node="clarify_with_user"),
            make_event("node_completed", node="clarify_with_user"),
            make_event("node_started", node="decompose_question"),
            make_event("node_completed", node="decompose_question"),
            make_event("node_started", node="research_supervisor", scope="supervisor"),
            make_event("research_started", node="researcher", scope="researcher", topic="研究 LangGraph"),
            make_event(
                "research_completed",
                node="compress_research",
                scope="researcher",
                topic="研究 LangGraph",
                summary="LangGraph 的研究结论摘要",
            ),
            make_event("node_completed", node="research_supervisor", scope="supervisor"),
            make_event("node_started", node="final_report_generation"),
            make_event("report", node="final_report_generation", content=DEFAULT_REPORT),
        ]
    )


class FakeResearchAgentAdapter:
    """测试用假 Agent 适配器：按脚本产出事件，绝不调用真实模型。"""

    def __init__(
        self,
        *,
        initial: FakeScript | None = None,
        resume: FakeScript | None = None,
        state: AgentStateView | None = None,
        step_delay: float = 0.0,
    ) -> None:
        self.initial = initial or default_initial_script()
        self.resume = resume or FakeScript(events=[])
        self.state = state
        # 每条事件之间的延迟：用于模拟“研究仍在进行时客户端断开 SSE”的场景
        self.step_delay = step_delay
        # 记录每次调用的参数，用于断言 query / resume 的传递方式
        self.calls: list[dict] = []
        # 记录读取过 checkpoint 的 thread_id，用于断言“没有触碰某个任务”
        self.state_calls: list[str] = []

    async def stream(
        self,
        thread_id: str,
        *,
        query: str | None = None,
        resume: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """产出脚本中的事件（或抛出脚本中的异常）。"""
        self.calls.append({"thread_id": thread_id, "query": query, "resume": resume})
        script = self.resume if resume is not None else self.initial
        if script.error is not None:
            raise script.error
        for event in script.events:
            if self.step_delay:
                await asyncio.sleep(self.step_delay)
            yield event

    async def get_state(self, thread_id: str) -> AgentStateView:
        """返回预设的 Agent 状态。"""
        self.state_calls.append(thread_id)
        if self.state is not None:
            return self.state
        return AgentStateView(
            thread_id=thread_id,
            has_checkpoint=True,
            next_nodes=(),
        )


def make_test_settings() -> Settings:
    """测试配置：不依赖真实 PostgreSQL。"""
    return Settings(
        database_url="sqlite+aiosqlite://",
        langgraph_database_url="postgresql://unused:unused@localhost:5432/unused",
        cors_origins=["http://localhost:5173"],
    )


@pytest.fixture
async def session_factory(tmp_path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """基于文件型 SQLite 的业务库会话工厂（测试专用，不走 Alembic）。"""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'research.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.fixture
async def app_factory(session_factory):
    """构造注入了假适配器与 SQLite 业务库的 FastAPI 应用与测试客户端。"""
    clients: list[AsyncClient] = []

    async def _factory(adapter: FakeResearchAgentAdapter | None = None):
        adapter = adapter or FakeResearchAgentAdapter()
        app = create_app(make_test_settings(), adapter=adapter, session_factory=session_factory)
        client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        )
        clients.append(client)
        return app, adapter, client

    try:
        yield _factory
    finally:
        for client in clients:
            await client.aclose()


async def create_task(client: AsyncClient, query: str = "研究 2026 年 AI Agent 技术发展趋势") -> str:
    """通过 API 创建研究任务并返回 task_id。"""
    response = await client.post("/api/v1/research", json={"query": query})
    assert response.status_code == 201, response.text
    return response.json()["task_id"]


async def wait_for_status(
    app,
    client: AsyncClient,
    task_id: str,
    expected: str,
    *,
    timeout: float = 5.0,
) -> dict:
    """等待后台任务把状态推进到期望值（测试辅助，非产品轮询实现）。"""
    task_service = app.state.task_service
    await task_service.wait_for(uuid.UUID(task_id), timeout=timeout)
    response = await client.get(f"/api/v1/research/{task_id}")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == expected, payload
    return payload
