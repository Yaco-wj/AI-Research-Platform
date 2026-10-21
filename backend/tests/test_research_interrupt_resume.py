"""interrupt / resume 闭环测试。"""

from tests.conftest import (
    DEFAULT_REPORT,
    FakeResearchAgentAdapter,
    FakeScript,
    create_task,
    make_event,
    wait_for_status,
)

CLARIFY_QUESTION = "你希望重点研究哪些方面？"


def interrupt_script() -> FakeScript:
    """首次执行：澄清判断后触发 interrupt。"""
    return FakeScript(
        events=[
            make_event("node_started", node="clarify_with_user"),
            make_event("node_completed", node="clarify_with_user"),
            make_event("node_started", node="wait_for_user_clarification"),
            make_event("clarification_required", question=CLARIFY_QUESTION),
        ]
    )


def resume_script() -> FakeScript:
    """恢复执行：继续研究并产出报告。"""
    return FakeScript(
        events=[
            make_event("node_started", node="decompose_question"),
            make_event("node_completed", node="decompose_question"),
            make_event("research_started", node="researcher", scope="researcher", topic="企业级落地"),
            make_event("node_started", node="final_report_generation"),
            make_event("report", node="final_report_generation", content=DEFAULT_REPORT),
        ]
    )


async def test_interrupt_sets_waiting_user_status(app_factory):
    """Agent interrupt 后任务状态变为 WAITING_USER，并保存澄清问题。"""
    adapter = FakeResearchAgentAdapter(initial=interrupt_script())
    app, _, client = await app_factory(adapter)
    task_id = await create_task(client, "帮我研究一下 AI Agent")

    payload = await wait_for_status(app, client, task_id, "WAITING_USER")

    assert payload["clarification_question"] == CLARIFY_QUESTION
    assert payload["completed_at"] is None
    # SSE 事件历史中包含状态与澄清事件（前端据此弹出澄清输入框）
    history = app.state.broker.history(task_id)
    event_names = [item.event for item in history]
    assert "status" in event_names
    assert "clarification_required" in event_names
    clarification = next(item for item in history if item.event == "clarification_required")
    assert clarification.data["question"] == CLARIFY_QUESTION


async def test_resume_continues_research_until_completed(app_factory):
    """POST /resume 通过 Command(resume=...) 恢复同一个 thread，并最终完成研究。"""
    adapter = FakeResearchAgentAdapter(initial=interrupt_script(), resume=resume_script())
    app, _, client = await app_factory(adapter)
    task_id = await create_task(client, "帮我研究一下 AI Agent")
    await wait_for_status(app, client, task_id, "WAITING_USER")

    response = await client.post(
        f"/api/v1/research/{task_id}/resume",
        json={"answer": "重点关注企业级落地"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "RUNNING"

    payload = await wait_for_status(app, client, task_id, "COMPLETED")
    # 澄清问题作为历史保留
    assert payload["clarification_question"] == CLARIFY_QUESTION
    # 恢复执行必须使用同一个 thread_id，且以 resume 方式调用 Agent
    assert adapter.calls[-1]["thread_id"] == task_id
    assert adapter.calls[-1]["resume"] == "重点关注企业级落地"
    assert adapter.calls[-1]["query"] is None

    report_response = await client.get(f"/api/v1/research/{task_id}/report")
    assert report_response.status_code == 200
    assert report_response.json()["report"] == DEFAULT_REPORT


async def test_resume_rejected_when_task_is_not_waiting(app_factory):
    """非 WAITING_USER 状态的任务不允许 resume。"""
    adapter = FakeResearchAgentAdapter()
    app, _, client = await app_factory(adapter)
    task_id = await create_task(client)
    await wait_for_status(app, client, task_id, "COMPLETED")

    response = await client.post(
        f"/api/v1/research/{task_id}/resume",
        json={"answer": "补充说明"},
    )

    assert response.status_code == 409
