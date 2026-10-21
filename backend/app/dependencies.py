"""FastAPI 依赖注入：统一从 app.state 获取平台组件。"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.agent_adapter.research_agent import ResearchAgentAdapter
from backend.app.service.event_bus import TaskEventBroker
from backend.app.service.research_service import ResearchService
from backend.app.service.task_service import TaskService


def get_app_settings(request: Request):
    """获取应用级配置。"""
    return request.app.state.settings


def get_broker(request: Request) -> TaskEventBroker:
    """获取事件总线。"""
    return request.app.state.broker


def get_adapter(request: Request) -> ResearchAgentAdapter:
    """获取 Agent 适配器。"""
    return request.app.state.adapter


def get_task_service(request: Request) -> TaskService:
    """获取后台任务执行器。"""
    return request.app.state.task_service


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """为一次请求提供业务数据库会话。"""
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with session_factory() as session:
        yield session


async def get_research_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    task_service: Annotated[TaskService, Depends(get_task_service)],
    broker: Annotated[TaskEventBroker, Depends(get_broker)],
    adapter: Annotated[ResearchAgentAdapter, Depends(get_adapter)],
) -> ResearchService:
    """构造研究任务服务。"""
    return ResearchService(
        session=session,
        task_service=task_service,
        broker=broker,
        adapter=adapter,
    )
