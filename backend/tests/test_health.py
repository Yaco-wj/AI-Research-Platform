"""健康检查接口测试。"""

from tests.conftest import FakeResearchAgentAdapter


async def test_health_returns_ok(app_factory):
    """GET /api/v1/health 应返回 status=ok。"""
    _, _, client = await app_factory(FakeResearchAgentAdapter())
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
