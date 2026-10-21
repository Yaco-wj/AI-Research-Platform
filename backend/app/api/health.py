"""健康检查接口。"""

from fastapi import APIRouter

from backend.app.schemas.research import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    summary="健康检查",
    description="用于存活探测：服务可正常响应时返回 status=ok。",
)
async def health() -> HealthResponse:
    """返回服务健康状态。"""
    return HealthResponse(status="ok")
