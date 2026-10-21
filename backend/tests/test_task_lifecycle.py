"""任务生命周期测试。

覆盖用户报告的线上问题：
1. 研究任务执行不能和 SSE 连接绑定 —— 客户端断开 SSE 后研究必须继续；
2. 重新进入详情页必须能恢复：状态、已完成阶段、clarification 信息、研究时间线；
3. SSE 断线重连不能重新启动 Agent（只做订阅 + 事件回放）；
4. 历史任务状态必须正确 —— 进程重启留下的 PENDING / RUNNING 不能永远是 RUNNING。
"""

import asyncio
import time
import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.testclient import TestClient

from backend.app.agent_adapter.research_agent import AgentStateView
from backend.app.main import create_app
from backend.app.models import Base, ResearchTaskStatus
from backend.app.repository.research_task_repository import ResearchTaskRepository
from backend.app.service.task_service import (
    RESTART_INTERRUPT_MESSAGE,
    SHUTDOWN_INTERRUPT_MESSAGE,
)
from backend.tests.conftest import (
    DEFAULT_REPORT,
    FakeResearchAgentAdapter,
    FakeScript,
    create_task,
    make_event,
    make_test_settings,
    wait_for_status,
)

API = "/api/v1/research"
CLARIFY_QUESTION = "你希望重点研究哪些方面？"


# ---------------------------------------------------------------------- #
# 测试设施：文件型 SQLite + Starlette TestClient（支持真正的流式读取）
# ---------------------------------------------------------------------- #
def _prepare_database(database_url: str) -> None:
    """建表（测试环境不走 Alembic）。"""

    async def _run() -> None:
        engine = create_async_engine(database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_run())


def _open_app(tmp_path, adapter: FakeResearchAgentAdapter, name: str = "lifecycle.db"):
    """构造一个指向指定 SQLite 文件的应用（可重复构造以模拟进程重启）。"""
    database_url = f"sqlite+aiosqlite:///{tmp_path / name}"
    _prepare_database(database_url)
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(make_test_settings(), adapter=adapter, session_factory=session_factory)
    return app, database_url


def _wait_status(client: TestClient, task_id: str, expected: str, timeout: float = 15.0):
    """轮询等待任务进入期望状态（测试辅助，非产品实现）。"""
    deadline = time.time() + timeout
    payload = {}
    while time.time() < deadline:
        payload = client.get(f"{API}/{task_id}").json()
        if payload["status"] == expected:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"任务状态未在 {timeout}s 内变为 {expected}，当前：{payload}")


def _create_task(client: TestClient, query: str = "研究 LangGraph 的生产级架构") -> str:
    """创建任务并返回 task_id。"""
    response = client.post(API, json={"query": query})
    assert response.status_code == 201, response.text
    return response.json()["task_id"]


def _seed_task(
    database_url: str,
    status: str,
    *,
    task_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """直接向业务库写入一条任务记录（用于模拟“上次进程崩溃留下的状态”）。"""

    async def _run() -> uuid.UUID:
        engine = create_async_engine(database_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            repository = ResearchTaskRepository(session)
            created_id = task_id or uuid.uuid4()
            await repository.create(
                task_id=created_id,
                thread_id=str(created_id),
                user_query="被中断的研究任务",
                title="被中断的研究任务",
            )
            task = await repository.get_by_task_id(created_id)
            assert task is not None
            if status == "RUNNING":
                await repository.mark_running(task)
            elif status == "COMPLETED":
                await repository.mark_completed(task, DEFAULT_REPORT)
            await session.commit()
        await engine.dispose()
        return created_id

    return asyncio.run(_run())


def _read_task_state(database_url: str, task_id: uuid.UUID) -> dict:
    """直接读业务库快照：状态 / 报告 / 错误 / 事件数量。"""

    async def _run() -> dict:
        engine = create_async_engine(database_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            repository = ResearchTaskRepository(session)
            task = await repository.get_by_task_id(task_id)
            assert task is not None
            events = await repository.list_events(task_id)
            snapshot = {
                "status": task.status.value,
                "final_report": task.final_report,
                "error_message": task.error_message,
                "event_names": [row.event for row in events],
                "event_count": len(events),
            }
        await engine.dispose()
        return snapshot

    return asyncio.run(_run())


def _initial_script() -> FakeScript:
    """一份包含完整主流程事件的脚本（与默认脚本一致，便于断言时间线）。"""
    return FakeScript(
        events=[
            make_event("node_started", node="clarify_with_user"),
            make_event("node_completed", node="clarify_with_user"),
            make_event("node_started", node="decompose_question"),
            make_event("node_completed", node="decompose_question"),
            make_event("node_started", node="research_supervisor", scope="supervisor"),
            make_event(
                "research_started",
                node="researcher",
                scope="researcher",
                topic="研究 LangGraph",
            ),
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


def _interrupt_script() -> FakeScript:
    """首次执行到 clarify 节点触发 interrupt。"""
    return FakeScript(
        events=[
            make_event("node_started", node="clarify_with_user"),
            make_event("node_completed", node="clarify_with_user"),
            make_event("node_started", node="wait_for_user_clarification"),
            make_event("clarification_required", question=CLARIFY_QUESTION),
        ]
    )


def _resume_script() -> FakeScript:
    """恢复执行：继续研究并产出报告。"""
    return FakeScript(
        events=[
            make_event("node_started", node="decompose_question"),
            make_event("research_started", node="researcher", scope="researcher", topic="企业级落地"),
            make_event("node_started", node="final_report_generation"),
            make_event("report", node="final_report_generation", content=DEFAULT_REPORT),
        ]
    )


# ---------------------------------------------------------------------- #
# 1. SSE 断开不影响 Agent 生命周期
# ---------------------------------------------------------------------- #
def test_agent_keeps_running_after_sse_disconnect(tmp_path):
    """客户端断开 SSE 后，后台研究继续执行并最终 COMPLETED。"""
    adapter = FakeResearchAgentAdapter(initial=_initial_script(), step_delay=0.05)
    app, _ = _open_app(tmp_path, adapter)

    with TestClient(app) as client:
        task_id = _create_task(client)

        # 建立 SSE 连接，收到开头几条事件后主动断开（模拟用户返回首页）
        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            assert stream.status_code == 200
            read = 0
            for _line in stream.iter_lines():
                read += 1
                if read >= 4:
                    break
        assert read >= 4, "应当已经从 SSE 读到事件"

        # 断开之后研究必须继续推进（不受 SSE 影响）
        payload = _wait_status(client, task_id, "COMPLETED")

        report = client.get(f"{API}/{task_id}/report")
        assert report.status_code == 200
        assert report.json()["report"] == DEFAULT_REPORT
        assert payload["completed_at"] is not None


def test_reconnect_after_disconnect_replays_whole_timeline(tmp_path):
    """断线重连：不重启 Agent，且能回放到断开之前已经产生的过程事件。"""
    adapter = FakeResearchAgentAdapter(initial=_initial_script(), step_delay=0.02)
    app, _ = _open_app(tmp_path, adapter)

    with TestClient(app) as client:
        task_id = _create_task(client)
        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            read = 0
            for _line in stream.iter_lines():
                read += 1
                if read >= 4:
                    break

        _wait_status(client, task_id, "COMPLETED")
        # Agent 只被调用了一次：重连只是订阅，不会重新启动研究
        assert len(adapter.calls) == 1

        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            body = "".join(stream.iter_text())

    assert "event: node_started" in body
    assert "clarify_with_user" in body
    assert "event: research_started" in body
    assert "研究 LangGraph" in body
    assert "event: report_chunk" in body
    assert "event: completed" in body
    assert '"status": "COMPLETED"' in body


# ---------------------------------------------------------------------- #
# 2. 重新进入任务详情：状态 / 阶段 / 澄清信息恢复
# ---------------------------------------------------------------------- #
def test_reenter_waiting_task_restores_clarification_then_resume_completes(tmp_path):
    """WAITING_USER 任务重新进入详情页仍显示等待澄清，回答后可以继续到完成。"""
    adapter = FakeResearchAgentAdapter(
        initial=_interrupt_script(),
        resume=_resume_script(),
        step_delay=0.02,
    )
    app, _ = _open_app(tmp_path, adapter)

    with TestClient(app) as client:
        task_id = _create_task(client, "帮我研究一下 AI Agent")
        waiting = _wait_status(client, task_id, "WAITING_USER")
        assert waiting["clarification_question"] == CLARIFY_QUESTION

        # “重新进入详情页”：新建 SSE 连接，只订阅历史，不触发任何执行
        calls_before_reenter = len(adapter.calls)
        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            body = "".join(stream.iter_text())

        assert len(adapter.calls) == calls_before_reenter, "重新进入详情页不能重启 Agent"
        assert "event: clarification_required" in body
        assert CLARIFY_QUESTION in body
        assert '"status": "WAITING_USER"' in body
        assert "event: node_started" in body

        # 回答澄清问题 -> 后台恢复 -> 完成
        response = client.post(f"{API}/{task_id}/resume", json={"answer": "重点关注企业级落地"})
        assert response.status_code == 202

        completed = _wait_status(client, task_id, "COMPLETED")
        assert completed["clarification_question"] == CLARIFY_QUESTION
        assert client.get(f"{API}/{task_id}/report").json()["report"] == DEFAULT_REPORT

        # 恢复使用的是同一个 LangGraph thread
        assert adapter.calls[-1]["thread_id"] == task_id
        assert adapter.calls[-1]["resume"] == "重点关注企业级落地"

        # 恢复之后重新进入详情页，能同时看到澄清阶段与恢复后的研究过程
        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            final_body = "".join(stream.iter_text())

    assert "event: clarification_required" in final_body
    assert "企业级落地" in final_body
    assert "event: completed" in final_body


# ---------------------------------------------------------------------- #
# 3. 进程重启：事件日志回放 + 孤儿任务状态修正
# ---------------------------------------------------------------------- #
def test_sse_replays_persisted_timeline_after_process_restart(tmp_path):
    """进程重启后（内存事件总线为空），SSE 仍能从数据库事件日志回放完整时间线。"""
    # 第一个“进程”：跑完一个任务
    app_a, database_url = _open_app(tmp_path, FakeResearchAgentAdapter(), name="restart.db")
    with TestClient(app_a) as client:
        task_id = _create_task(client, "研究进程重启后的时间线恢复")
        _wait_status(client, task_id, "COMPLETED")

    # 第二个“进程”：全新的 broker，内存里没有任何该任务的事件
    app_b = create_app(
        make_test_settings(),
        adapter=FakeResearchAgentAdapter(),
        session_factory=async_sessionmaker(
            create_async_engine(database_url), expire_on_commit=False
        ),
    )
    assert not app_b.state.broker.has_stream(task_id)

    with TestClient(app_b) as client:
        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            body = "".join(stream.iter_text())

        assert client.get(f"{API}/{task_id}").json()["status"] == "COMPLETED"
        report = client.get(f"{API}/{task_id}/report")
        assert report.status_code == 200
        assert report.json()["report"] == DEFAULT_REPORT

    # 回放来自数据库事件日志：包含完整阶段与最终报告
    assert "event: node_started" in body
    assert "clarify_with_user" in body
    assert "event: research_started" in body
    assert "event: report_chunk" in body
    assert "event: completed" in body
    assert '"status": "COMPLETED"' in body


@pytest.mark.parametrize(
    ("checkpoint_state", "expected_status", "expected_question"),
    [
        (None, "FAILED", None),
        ("interrupt", "WAITING_USER", CLARIFY_QUESTION),
        ("completed", "COMPLETED", None),
    ],
)
def test_orphaned_running_task_is_reconciled_on_startup(
    tmp_path, checkpoint_state, expected_status, expected_question
):
    """上次进程留下的 RUNNING 任务：启动时按 checkpoint 修正为正确状态。"""
    database_url = f"sqlite+aiosqlite:///{tmp_path / f'orphan-{checkpoint_state}.db'}"
    _prepare_database(database_url)
    task_id = _seed_task(database_url, "RUNNING")

    adapter = FakeResearchAgentAdapter()
    if checkpoint_state == "interrupt":
        adapter.state = AgentStateView(
            thread_id=str(task_id),
            has_checkpoint=True,
            next_nodes=("wait_for_user_clarification",),
            interrupt_question=CLARIFY_QUESTION,
        )
    elif checkpoint_state == "completed":
        adapter.state = AgentStateView(
            thread_id=str(task_id),
            has_checkpoint=True,
            next_nodes=(),
            final_report=DEFAULT_REPORT,
        )

    app = create_app(
        make_test_settings(),
        adapter=adapter,
        session_factory=async_sessionmaker(
            create_async_engine(database_url), expire_on_commit=False
        ),
    )

    with TestClient(app) as client:
        payload = client.get(f"{API}/{task_id}").json()

    assert payload["status"] == expected_status
    assert payload["clarification_question"] == expected_question
    if expected_status == "FAILED":
        assert payload["error_message"] == RESTART_INTERRUPT_MESSAGE
    if expected_status == "COMPLETED":
        assert _read_report(database_url, task_id) == DEFAULT_REPORT


def _read_report(database_url: str, task_id: uuid.UUID) -> str:
    """在应用关闭后直接从数据库读取最终报告（避免依赖已关闭的 TestClient）。"""

    async def _run() -> str:
        engine = create_async_engine(database_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            repository = ResearchTaskRepository(session)
            task = await repository.get_by_task_id(task_id)
            assert task is not None and task.final_report is not None
            report = task.final_report
        await engine.dispose()
        return report

    return asyncio.run(_run())


# ---------------------------------------------------------------------- #
# 4. 状态机正确性：不会留下假的 RUNNING
# ---------------------------------------------------------------------- #
def test_shutdown_marks_inflight_task_failed(tmp_path):
    """服务关闭时正在执行的任务被取消并写成 FAILED，而不是永远 RUNNING。"""
    adapter = FakeResearchAgentAdapter(initial=_initial_script(), step_delay=1.0)
    app, database_url = _open_app(tmp_path, adapter, name="shutdown.db")

    with TestClient(app) as client:
        task_id = _create_task(client, "会被服务关闭打断的研究")
        _wait_status(client, task_id, "RUNNING")
    # 退出 TestClient -> lifespan 关闭 -> TaskService.shutdown() 取消后台任务

    async def _read() -> dict:
        engine = create_async_engine(database_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            repository = ResearchTaskRepository(session)
            task = await repository.get_by_task_id(uuid.UUID(task_id))
            assert task is not None
            result = {"status": task.status.value, "error_message": task.error_message}
        await engine.dispose()
        return result

    snapshot = asyncio.run(_read())
    assert snapshot["status"] == "FAILED"
    assert snapshot["error_message"] == SHUTDOWN_INTERRUPT_MESSAGE


def test_duplicate_resume_is_rejected(tmp_path):
    """重复 resume 不能并发驱动同一个 LangGraph thread。"""
    adapter = FakeResearchAgentAdapter(
        initial=_interrupt_script(),
        resume=_resume_script(),
        step_delay=0.05,
    )
    app, _ = _open_app(tmp_path, adapter, name="duplicate-resume.db")

    with TestClient(app) as client:
        task_id = _create_task(client, "帮我研究一下 AI Agent")
        _wait_status(client, task_id, "WAITING_USER")

        first = client.post(f"{API}/{task_id}/resume", json={"answer": "重点 A"})
        assert first.status_code == 202
        second = client.post(f"{API}/{task_id}/resume", json={"answer": "重点 B"})
        assert second.status_code == 409

        _wait_status(client, task_id, "COMPLETED")

    # 恢复脚本只执行了一次
    resume_calls = [call for call in adapter.calls if call["resume"] is not None]
    assert len(resume_calls) == 1


def test_sse_since_id_resumes_without_replaying_delivered_events(tmp_path):
    """断点续传：since_id 之后的事件才回放，已处理事件不会重复推送。"""
    app, _ = _open_app(tmp_path, FakeResearchAgentAdapter(), name="since-id.db")

    with TestClient(app) as client:
        task_id = _create_task(client, "研究断点续传")
        _wait_status(client, task_id, "COMPLETED")

        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            first_body = "".join(stream.iter_text())

        first_ids = _extract_event_ids(first_body)
        assert len(first_ids) >= 4
        cursor = first_ids[len(first_ids) // 2]

        with client.stream("GET", f"{API}/{task_id}/stream?since_id={cursor}") as stream:
            resumed_body = "".join(stream.iter_text())

    resumed_ids = _extract_event_ids(resumed_body)
    assert resumed_ids, "断点续传仍应推送游标之后的剩余事件"
    assert all(event_id > cursor for event_id in resumed_ids)
    assert cursor not in resumed_ids
    # 已处理过的早期阶段不会再次出现
    assert "event: completed" in resumed_body


def _extract_event_ids(body: str) -> list[int]:
    """从 SSE 文本中提取 id 字段（按出现顺序）。"""
    ids: list[int] = []
    for line in body.splitlines():
        if line.startswith("id:"):
            raw = line.split(":", 1)[1].strip()
            if raw.isdigit():
                ids.append(int(raw))
    return ids


# ---------------------------------------------------------------------- #
# 5. 已完成任务重新进入详情页（用户报告：详情页显示 FAILED 但库里 COMPLETED）
#
# 线上场景还原：一个已经 COMPLETED 的任务（final_report 存在、过程事件存在），
# 用户刷新首页 -> 再次进入详情页，此时：
#   - GET /research        （首页历史列表）          -> COMPLETED
#   - GET /research/{id}   （详情页任务信息）        -> COMPLETED
#   - GET /research/{id}/stream（详情页时间线）      -> 只回放历史，不改状态
#   - GET /research/{id}/report（详情页报告）        -> final_report 内容
#   并且绝不能再次触发 Agent 执行。
# ---------------------------------------------------------------------- #
def test_completed_task_detail_page_keeps_status_report_and_events(tmp_path):
    """完成的任务刷新首页后重进详情页：COMPLETED + 报告 + 过程事件，且不重跑 Agent。"""
    adapter = FakeResearchAgentAdapter()
    app, database_url = _open_app(tmp_path, adapter, name="detail-completed.db")

    with TestClient(app) as client:
        task_id = _create_task(client, "研究详情页状态一致性")
        _wait_status(client, task_id, "COMPLETED")
        task_uuid = uuid.UUID(task_id)
        calls_after_run = len(adapter.calls)

        snapshot = _read_task_state(database_url, task_uuid)
        assert snapshot["status"] == "COMPLETED"
        assert snapshot["final_report"] == DEFAULT_REPORT
        assert snapshot["event_count"] > 0, "完成任务必须有持久化的过程事件"

        # 1) 刷新首页：历史列表里的状态必须来自业务库
        listing = client.get(f"{API}?limit=8&offset=0").json()
        listed = next(item for item in listing["items"] if item["task_id"] == task_id)
        assert listed["status"] == "COMPLETED"

        # 2) 进入详情页：任务信息
        detail = client.get(f"{API}/{task_id}")
        assert detail.status_code == 200
        detail_payload = detail.json()
        assert detail_payload["status"] == "COMPLETED"
        assert detail_payload["error_message"] is None

        # 3) 详情页时间线：SSE 只回放事件，不触发执行
        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            body = "".join(stream.iter_text())

        assert "event: node_started" in body
        assert "event: research_started" in body
        assert "event: report_chunk" in body
        assert "event: completed" in body
        assert '"status": "COMPLETED"' in body
        assert "event: error" not in body

        # 4) 详情页报告
        report = client.get(f"{API}/{task_id}/report")
        assert report.status_code == 200
        assert report.json()["report"] == DEFAULT_REPORT

        # 5) 不能再次触发 Agent 执行 / 不能改状态
        assert len(adapter.calls) == calls_after_run
        after = _read_task_state(database_url, task_uuid)
        assert after == snapshot
        assert client.get(f"{API}/{task_id}").json()["status"] == "COMPLETED"


def test_completed_task_after_process_restart_keeps_status_report_and_events(tmp_path):
    """进程重启（内存事件总线为空）后重进详情页，仍然是 COMPLETED + 报告 + 历史事件。"""
    adapter_a = FakeResearchAgentAdapter()
    app_a, database_url = _open_app(tmp_path, adapter_a, name="detail-restart.db")
    with TestClient(app_a) as client:
        task_id = _create_task(client, "进程重启后的详情页")
        _wait_status(client, task_id, "COMPLETED")

    # 第二个“进程”：全新 broker，内存里没有该任务的事件流
    adapter_b = FakeResearchAgentAdapter()
    app_b = create_app(
        make_test_settings(),
        adapter=adapter_b,
        session_factory=async_sessionmaker(
            create_async_engine(database_url), expire_on_commit=False
        ),
    )
    assert not app_b.state.broker.has_stream(task_id)

    with TestClient(app_b) as client:
        assert client.get(f"{API}/{task_id}").json()["status"] == "COMPLETED"
        with client.stream("GET", f"{API}/{task_id}/stream") as stream:
            body = "".join(stream.iter_text())
        assert client.get(f"{API}/{task_id}/report").json()["report"] == DEFAULT_REPORT
        # 对账不会动 COMPLETED 任务，也不会读它的 checkpoint
        assert task_id not in adapter_b.state_calls

    assert "event: completed" in body
    assert "event: report_chunk" in body
    # 新进程没有重新执行 Agent
    assert adapter_b.calls == []


def test_stream_endpoint_never_writes_task_state(tmp_path):
    """GET /stream 是纯订阅：既不修改状态，也不新增事件。"""
    app, database_url = _open_app(tmp_path, FakeResearchAgentAdapter(), name="stream-readonly.db")

    with TestClient(app) as client:
        task_id = _create_task(client, "SSE 只读订阅")
        _wait_status(client, task_id, "COMPLETED")
        task_uuid = uuid.UUID(task_id)
        before = _read_task_state(database_url, task_uuid)

        for _ in range(3):
            with client.stream("GET", f"{API}/{task_id}/stream") as stream:
                "".join(stream.iter_text())

        assert _read_task_state(database_url, task_uuid) == before


def test_orphan_recovery_never_overwrites_completed_task(tmp_path):
    """启动对账只处理 PENDING / RUNNING，COMPLETED 的状态与报告绝不被覆盖。"""
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'orphan-completed.db'}"
    _prepare_database(database_url)
    task_id = _seed_task(database_url, "COMPLETED")

    adapter = FakeResearchAgentAdapter()
    app = create_app(
        make_test_settings(),
        adapter=adapter,
        session_factory=async_sessionmaker(
            create_async_engine(database_url), expire_on_commit=False
        ),
    )

    with TestClient(app) as client:
        payload = client.get(f"{API}/{task_id}").json()
        assert client.get(f"{API}/{task_id}/report").json()["report"] == DEFAULT_REPORT

    assert payload["status"] == "COMPLETED"
    assert payload["error_message"] is None
    # COMPLETED 任务不在对账范围内：不读 checkpoint、不执行 Agent
    assert adapter.state_calls == []
    assert adapter.calls == []
    assert _read_task_state(database_url, task_id)["final_report"] == DEFAULT_REPORT


async def test_terminal_status_is_never_downgraded(app_factory, session_factory):
    """终态不可倒退：COMPLETED 任务被迟到的取消 / 失败写入时保持不变。"""
    adapter = FakeResearchAgentAdapter()
    app, _, client = await app_factory(adapter)
    task_id = await create_task(client, "终态不可倒退")
    await wait_for_status(app, client, task_id, "COMPLETED")
    task_uuid = uuid.UUID(task_id)

    task_service = app.state.task_service
    # 模拟“服务关闭 / 迟到异常”在任务完成之后才落库
    await task_service._mark_failed(task_uuid, "迟到的失败写入")
    await task_service._mark_waiting_user(task_uuid, "迟到的澄清写入")

    async with session_factory() as session:
        repository = ResearchTaskRepository(session)
        task = await repository.get_by_task_id(task_uuid)
        assert task is not None
        assert task.status is ResearchTaskStatus.COMPLETED
        assert task.final_report == DEFAULT_REPORT
        assert task.error_message is None
        events = [row.event for row in await repository.list_events(task_uuid)]

    # 也不能把“迟到的失败”写进事件日志（否则详情页回放会显示 FAILED）
    assert "error" not in events
    assert "completed" in events
