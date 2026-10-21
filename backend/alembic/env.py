"""Alembic 迁移环境（异步引擎，连接串来自 DATABASE_URL）。"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# 平台代码以 `backend.app.xxx` 导入：这里根据 env.py 自身位置把
# 仓库根目录与 backend/ 都加入 sys.path，保证无论从哪个工作目录执行 alembic，
# 导入路径都正确（不依赖 PYTHONPATH）。
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
for _path in (BACKEND_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from backend.app.core.config import get_settings  # noqa: E402  (必须在 sys.path 调整之后导入)
from backend.app.models import Base  # noqa: E402

# Alembic 配置对象
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 目标元数据：所有业务表模型
target_metadata = Base.metadata


def _database_url() -> str:
    """读取业务数据库连接串（asyncpg 驱动）。"""
    return get_settings().database_url


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL，不连接数据库。"""
    context.configure(
        url=_database_url().replace("%", "%%"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """在给定连接上执行迁移。"""
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """在线模式：使用异步引擎连接数据库后执行迁移。"""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = async_engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """在线模式入口。"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
