"""FastAPI 应用入口。

启动顺序（lifespan）：
1. 初始化业务数据库会话工厂（SQLAlchemy + asyncpg）；
2. 初始化 LangGraph PostgresSaver（独立连接池 + checkpoint 建表）；
3. 编译 Agent 主图并构造 Agent 适配器与任务执行器；
4. 应用运行；
5. 退出时关闭 checkpoint 连接池与业务数据库引擎。
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.agent_adapter.research_agent import ResearchAgentAdapter
from backend.app.api import health, research
from backend.app.core.checkpoint import CheckpointManager
from backend.app.core.config import Settings, get_settings
from backend.app.core.database import dispose_database, get_session_factory
from backend.app.service.event_bus import TaskEventBroker
from backend.app.service.task_service import TaskService

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    adapter: ResearchAgentAdapter | None = None,
    session_factory: Any | None = None,
    checkpointer: Any | None = None,
) -> FastAPI:
    """构造 FastAPI 应用。

    Args:
        settings: 平台配置，默认从环境变量读取
        adapter: Agent 适配器（测试时可注入假实现）
        session_factory: 业务数据库会话工厂（测试时可注入 SQLite）
        checkpointer: LangGraph checkpointer（测试时可注入 InMemorySaver）
    """
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期：初始化与释放平台资源。"""
        # 1. 业务数据库
        if app.state.session_factory is None:
            app.state.session_factory = get_session_factory(app.state.settings)

        # 2. LangGraph checkpoint（与业务数据库完全独立）
        if app.state.adapter is None:
            checkpointer_instance = app.state.checkpointer
            if checkpointer_instance is None:
                manager = CheckpointManager(
                    app.state.settings.langgraph_database_url,
                    min_size=app.state.settings.checkpoint_pool_min_size,
                    max_size=app.state.settings.checkpoint_pool_max_size,
                )
                checkpointer_instance = await manager.open()
                app.state.checkpoint_manager = manager
                app.state.checkpointer = checkpointer_instance
            app.state.adapter = ResearchAgentAdapter(checkpointer_instance)

        # 3. 后台任务执行器（单进程内运行 asyncio 任务）
        if app.state.task_service is None:
            app.state.task_service = TaskService(
                session_factory=app.state.session_factory,
                adapter=app.state.adapter,
                broker=app.state.broker,
            )

        # 4. 修正上一次进程崩溃 / 被 kill 留下的“孤儿任务”：
        #    进程内 asyncio 任务无法跨进程存活，若不做这一步，
        #    数据库里会永远留下 RUNNING / PENDING 的假状态。
        try:
            recovered = await app.state.task_service.recover_orphaned_tasks()
        except Exception:  # noqa: BLE001 - 恢复失败不能阻止服务启动
            logger.exception("孤儿研究任务恢复失败")
        else:
            if recovered:
                logger.info("已修正 %d 个孤儿研究任务", recovered)

        logger.info("%s 启动完成", app.state.settings.app_name)
        try:
            yield
        finally:
            # 先停掉后台研究任务：进程内任务随进程消失，
            # 这里显式取消并落库，避免留下永远 RUNNING 的记录。
            if app.state.task_service is not None:
                await app.state.task_service.shutdown()
            if app.state.checkpoint_manager is not None:
                await app.state.checkpoint_manager.close()
            await dispose_database()
            logger.info("%s 已关闭", app.state.settings.app_name)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Open Deep Research 平台后端：任务管理、SSE 流式研究过程、interrupt/resume。",
        lifespan=lifespan,
    )

    # 应用状态：允许测试直接注入替身，跳过真实数据库依赖
    app.state.settings = settings
    app.state.broker = TaskEventBroker()
    app.state.checkpointer = checkpointer
    app.state.checkpoint_manager = None
    app.state.adapter = adapter
    app.state.session_factory = session_factory
    app.state.task_service = (
        TaskService(session_factory, adapter, app.state.broker)
        if adapter is not None and session_factory is not None
        else None
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(research.router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """根路径：指向 API 文档。"""
        return {"name": settings.app_name, "docs": "/docs", "api": settings.api_prefix}

    return app


def _configure_logging(level: str) -> None:
    """配置基础日志（后端日志保留详细异常，对外只返回友好文案）。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


app = create_app()
