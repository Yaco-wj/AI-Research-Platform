"""业务数据库（SQLAlchemy 2.x 异步引擎 + asyncpg）连接管理。

这里只承载 ResearchTask 等业务表；
LangGraph checkpoint 由 app/core/checkpoint.py 独立管理，两者互不复用连接。
"""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import Settings, get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """获取（懒加载）业务数据库异步引擎。"""
    global _engine
    if _engine is None:
        settings = settings or get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.sql_echo,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """获取（懒加载）业务数据库会话工厂。"""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(settings),
            expire_on_commit=False,
        )
    return _session_factory


async def dispose_database() -> None:
    """关闭业务数据库连接池（应用退出时调用）。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
