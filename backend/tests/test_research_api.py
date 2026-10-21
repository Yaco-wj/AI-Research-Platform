"""研究任务基础接口测试：创建 / 查询 / 历史 / 报告。"""

from tests.conftest import FakeResearchAgentAdapter, create_task, wait_for_status


async def test_create_research_task_returns_pending(app_factory):
    """POST /api/v1/research 立即返回 PENDING，不同步等待研究完成。"""
    app, adapter, client = await app_factory(FakeResearchAgentAdapter())

    response = await client.post(
        "/api/v1/research", json={"query": "研究 2026 年 AI Agent 技术发展趋势"}
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "PENDING"
    # thread_id 与 task_id 一一对应
    assert payload["thread_id"] == payload["task_id"]

    # 后台确实用 Agent 适配器启动了研究（首次调用传入 query，而不是 resume）
    await wait_for_status(app, client, payload["task_id"], "COMPLETED")
    assert adapter.calls[0]["query"] == "研究 2026 年 AI Agent 技术发展趋势"
    assert adapter.calls[0]["resume"] is None


async def test_get_research_task_returns_detail(app_factory):
    """GET /api/v1/research/{task_id} 返回任务详情。"""
    app, _, client = await app_factory(FakeResearchAgentAdapter())
    task_id = await create_task(client, "调研企业级 AI Agent 落地难点")
    await wait_for_status(app, client, task_id, "COMPLETED")

    response = await client.get(f"/api/v1/research/{task_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == task_id
    assert payload["query"] == "调研企业级 AI Agent 落地难点"
    assert payload["status"] == "COMPLETED"
    assert payload["title"]
    assert payload["started_at"] is not None
    assert payload["completed_at"] is not None


async def test_get_unknown_task_returns_404(app_factory):
    """查询不存在的任务返回 404。"""
    _, _, client = await app_factory(FakeResearchAgentAdapter())

    response = await client.get("/api/v1/research/2b7a1f4c-0000-4000-8000-000000000000")

    assert response.status_code == 404


async def test_list_research_tasks(app_factory):
    """GET /api/v1/research 返回基础研究历史。"""
    app, _, client = await app_factory(FakeResearchAgentAdapter())
    first_id = await create_task(client, "第一个研究问题")
    second_id = await create_task(client, "第二个研究问题")
    await wait_for_status(app, client, first_id, "COMPLETED")
    await wait_for_status(app, client, second_id, "COMPLETED")

    response = await client.get("/api/v1/research", params={"limit": 10, "offset": 0})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    returned_ids = {item["task_id"] for item in payload["items"]}
    assert {first_id, second_id} <= returned_ids


async def test_report_before_completion_returns_409(app_factory):
    """任务未完成时获取报告返回 409。"""
    from tests.conftest import FakeScript, make_event

    adapter = FakeResearchAgentAdapter(
        initial=FakeScript(
            events=[
                make_event("node_started", node="clarify_with_user"),
                make_event("clarification_required", question="你想重点研究哪些方面？"),
            ]
        )
    )
    app, _, client = await app_factory(adapter)
    task_id = await create_task(client, "帮我研究一下 AI Agent")
    await wait_for_status(app, client, task_id, "WAITING_USER")

    response = await client.get(f"/api/v1/research/{task_id}/report")

    assert response.status_code == 409
    assert "尚未完成" in response.json()["detail"]


async def test_report_returns_markdown_after_completion(app_factory):
    """任务完成后返回 Markdown 报告。"""
    from tests.conftest import DEFAULT_REPORT

    app, _, client = await app_factory(FakeResearchAgentAdapter())
    task_id = await create_task(client)
    await wait_for_status(app, client, task_id, "COMPLETED")

    response = await client.get(f"/api/v1/research/{task_id}/report")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == task_id
    assert payload["report"] == DEFAULT_REPORT


async def test_create_request_validates_empty_query(app_factory):
    """空的 query 会被 Pydantic 校验拒绝。"""
    _, _, client = await app_factory(FakeResearchAgentAdapter())

    response = await client.post("/api/v1/research", json={"query": ""})

    assert response.status_code == 422
