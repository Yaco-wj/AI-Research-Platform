"""平台层配置。

配置来源优先级（从高到低）：
1. 进程环境变量；
2. backend/.env；
3. 仓库根目录 .env（方便开发时复用 Agent 的模型与搜索密钥）；
4. 字段默认值。
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> core / app / backend / 仓库根目录
CORE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CORE_DIR.parents[1]
REPO_ROOT = BACKEND_DIR.parent
BACKEND_ENV_FILE = BACKEND_DIR / ".env"
REPO_ENV_FILE = REPO_ROOT / ".env"


def load_environment() -> None:
    """加载 .env 文件，不覆盖已经存在的进程环境变量。"""
    load_dotenv(BACKEND_ENV_FILE, override=False)
    load_dotenv(REPO_ENV_FILE, override=False)


class Settings(BaseSettings):
    """平台层运行配置。"""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 服务基础配置
    app_name: str = "Open Deep Research Platform"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # 业务数据库：SQLAlchemy + asyncpg
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/deep_research"
    sql_echo: bool = False

    # LangGraph checkpoint：AsyncPostgresSaver（psycopg 连接串）
    langgraph_database_url: str = "postgresql://postgres:postgres@localhost:5432/deep_research"
    checkpoint_pool_min_size: int = 1
    checkpoint_pool_max_size: int = 5

    # 开发环境跨域：默认只允许 Vite 开发服务器
    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。"""
    load_environment()
    return Settings()
