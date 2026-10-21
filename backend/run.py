"""Windows 友好的启动入口。

Windows 上 asyncio 默认使用 ProactorEventLoop，而 psycopg（LangGraph PostgresSaver 的底层驱动）
的异步模式只在 SelectorEventLoop 上工作。直接运行 ``python run.py`` 会先切换事件循环策略，
再交给 uvicorn，从而避免 `Psycopg cannot use the 'ProactorEventLoop' to run in async mode`。

用法：
    cd backend
    python run.py
"""

import asyncio
import sys
from pathlib import Path

# 允许直接以脚本方式运行：平台代码以 `backend.app.xxx` 导入，
# 因此既需要仓库根目录（导入 backend.*），也需要 backend 目录（兼容 app.* 导入）。
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
for _path in (BACKEND_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn  # noqa: E402  (必须在设置事件循环策略之后导入)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main() -> None:
    """启动 FastAPI 服务（生产/本地均可，Windows 上同样适用）。"""
    uvicorn.run("backend.app.main:app", host=DEFAULT_HOST, port=DEFAULT_PORT, reload=False)


if __name__ == "__main__":
    main()
