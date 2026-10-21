"""SSE 接口测试。

说明：httpx 的 ASGITransport 会等待响应体全部结束，
因此这里使用 Starlette TestClient（支持真正的流式读取）。
"""

import asyncio
import time

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.testclient import TestClient

from backend.app.main import create_app
from backend.app.models import Base
from backend.tests.conftest import (
    FakeResearchAgentAdapter,
    FakeScript,
    make_test_settings,
    make_event,
)

API = "/api/v1/research"


def _build_app(tmp_path, adapter: FakeResearchAgentAdapter):
    """构造使用文件型 SQLite 的应用（TestClient 运行在独立事件循环中）。"""
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'sse.db'}"

    async def _prepare() -> None:
        engine = create_async_engine(database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_prepare())
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return create_app(make_test_settings(), adapter=adapter, session_factory=session_factory)


def _wait_status(client: TestClient, task_id: str, expected: str, timeout: float = 10.0):
    """等待任务状态推进（测试辅助）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"{API}/{task_id}").json()
        if payload["status"] == expected:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"任务状态未在 {timeout}s 内变为 {expected}")


def test_sse_stream_replays_history_and_reports_completion(tmp_path):
    """任务完成后连接 SSE：建立 text/event-stream 连接并回放完整历史事件。"""
    adapter = FakeResearchAgentAdapter()
    app = _build_app(tmp_path, adapter)

    with TestClient(app) as client:
        response = client.post(f"{API}", json={"query": "研究 2026 年 AI Agent 技术发展趋势"})
        task_id = response.json()["task_id"]
        _wait_status(client, task_id, "COMPLETED")

        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith("text/event-stream")
            body = "".join(stream.iter_text())

    assert "event: status" in body
    assert "event: node_started" in body
    assert "event: research_started" in body
    assert "研究 LangGraph" in body
    assert "event: report_chunk" in body
    assert "event: completed" in body
    assert '"status": "COMPLETED"' in body


def test_sse_stream_pushes_clarification_and_stays_open(tmp_path):
    """interrupt 场景：SSE 推送 clarification_required 后结束当前连接。"""
    adapter = FakeResearchAgentAdapter(
        initial=FakeScript(
            events=[
                make_event("node_started", node="clarify_with_user"),
                make_event("clarification_required", question="你希望重点研究哪些方面？"),
            ]
        )
    )
    app = _build_app(tmp_path, adapter)

    with TestClient(app) as client:
        response = client.post(f"{API}", json={"query": "帮我研究一下 AI Agent"})
        task_id = response.json()["task_id"]
        _wait_status(client, task_id, "WAITING_USER")

        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith("text/event-stream")
            text = "".join(stream.iter_text())

    assert "event: status" in text
    assert "event: clarification_required" in text
    assert "你希望重点研究哪些方面？" in text
    assert '"status": "WAITING_USER"' in text
