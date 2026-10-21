"""Agent 执行异常时的失败处理测试。"""

from backend.app.service.task_service import FALLBACK_ERROR_MESSAGE
from tests.conftest import FakeResearchAgentAdapter, FakeScript, create_task, wait_for_status


async def test_agent_exception_marks_task_failed(app_factory):
    """Agent 抛异常时：状态置为 FAILED，只保存友好文案，不泄露 traceback。"""
    adapter = FakeResearchAgentAdapter(
        initial=FakeScript(error=RuntimeError("DeepSeek 调用失败：invalid api key sk-secret"))
    )
    app, _, client = await app_factory(adapter)
    task_id = await create_task(client, "研究一个会失败的问题")

    payload = await wait_for_status(app, client, task_id, "FAILED")

    assert payload["error_message"] == FALLBACK_ERROR_MESSAGE
    assert "invalid api key" not in payload["error_message"]
    assert "Traceback" not in payload["error_message"]
    assert payload["completed_at"] is not None

    # 通过 SSE 事件推送的是同一条友好文案
    history = app.state.broker.history(task_id)
    error_events = [item for item in history if item.event == "error"]
    assert error_events, "失败时必须推送 error 事件"
    assert error_events[-1].data["message"] == FALLBACK_ERROR_MESSAGE
    assert "sk-secret" not in str(error_events[-1].data)

    # 失败任务无法获取报告
    report_response = await client.get(f"/api/v1/research/{task_id}/report")
    assert report_response.status_code == 409
