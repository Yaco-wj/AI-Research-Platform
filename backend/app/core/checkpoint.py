"""LangGraph checkpoint（AsyncPostgresSaver）生命周期管理。

职责：
1. 创建 PostgresSaver 所需的 psycopg 连接池；
2. 首次启动时初始化 checkpoint 相关数据表；
3. 提供随应用生命周期存活的 saver 单例。

注意：checkpoint 表与业务表（research_tasks）是两套独立的数据结构，
不要用 PostgresSaver 充当 ResearchTask 的仓储。
"""

import asyncio
import logging
import sys

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


def _ensure_supported_event_loop() -> None:
    """在 Windows 上提前拦截 psycopg 不支持的事件循环。

    psycopg 异步模式无法运行在 Windows 默认的 ProactorEventLoop 上，
    若不提前检查，只会看到连接池超时的报错，难以定位。
    """
    if sys.platform != "win32":
        return
    loop = asyncio.get_running_loop()
    if isinstance(loop, asyncio.ProactorEventLoop):
        raise RuntimeError(
            "检测到 Windows 默认的 ProactorEventLoop，psycopg 异步模式无法使用。"
            "请改用 `python run.py` 启动，或使用 `uvicorn app.main:app --reload`，"
            "或在启动脚本中先执行 asyncio.set_event_loop_policy("
            "asyncio.WindowsSelectorEventLoopPolicy())。"
        )


class CheckpointManager:
    """管理 AsyncPostgresSaver 与连接池的生命周期。"""

    def __init__(
        self,
        conn_string: str,
        *,
        min_size: int = 1,
        max_size: int = 5,
    ) -> None:
        self._conn_string = conn_string
        self._min_size = min_size
        self._max_size = max_size
        self._pool: AsyncConnectionPool | None = None
        self._saver: AsyncPostgresSaver | None = None

    @property
    def saver(self) -> AsyncPostgresSaver:
        """获取已初始化的 checkpointer。"""
        if self._saver is None:
            raise RuntimeError("CheckpointManager 尚未初始化，请先调用 open()。")
        return self._saver

    async def open(self) -> AsyncPostgresSaver:
        """建立连接池并初始化 checkpoint 数据表。"""
        if self._saver is not None:
            return self._saver

        _ensure_supported_event_loop()

        # AsyncPostgresSaver 要求 autocommit + prepare_threshold=0 + dict_row
        self._pool = AsyncConnectionPool(
            conninfo=self._conn_string,
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        await self._pool.open(wait=True, timeout=15)
        self._saver = AsyncPostgresSaver(self._pool)
        # 首次启动需要建表：checkpoint_migrations / checkpoints / checkpoint_blobs / checkpoint_writes
        await self._saver.setup()
        logger.info("LangGraph PostgresSaver 初始化完成")
        return self._saver

    async def close(self) -> None:
        """关闭连接池。"""
        if self._pool is not None:
            await self._pool.close()
            logger.info("LangGraph PostgresSaver 连接池已关闭")
        self._pool = None
        self._saver = None
