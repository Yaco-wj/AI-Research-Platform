"""进程内事件总线。

职责：把后台研究任务发布的实时事件广播给所有 SSE 订阅者，并在新订阅者接入时
先回放内存中的历史事件（保证 SSE 晚于任务启动也不会丢事件）。

设计边界（很重要）：
- 事件总线只负责“实时”，它不驱动 Agent，也不感知 HTTP 请求的生命周期；
  SSE 断开只会移除一个订阅者，绝不会影响后台研究任务。
- 内存历史有长度上限，跨进程/重启后不可用，因此 **不承担可回放的持久化职责**：
  研究过程会由 TaskService 同步写入 research_task_events 表，
  SSE 回放以数据库为准，本总线的历史只作为“实时快照”。

`event_id` 允许由持久化层提供（数据库自增主键），这样 SSE 的 id 在
“实时流”与“数据库回放”两条路径上完全一致，前端 since_id 断点续传才可靠。
"""

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 每个任务最多保留的历史事件数量，避免长时间任务占用过多内存
DEFAULT_HISTORY_LIMIT = 500


@dataclass(slots=True)
class StreamEvent:
    """一条 SSE 事件。"""

    id: int
    event: str
    data: dict[str, Any]


@dataclass
class _TaskStream:
    """单个任务的事件流：历史事件 + 活跃订阅者。"""

    history: deque[StreamEvent] = field(default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_LIMIT))
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    finished: bool = False


@dataclass(slots=True)
class Subscription:
    """一次原子订阅的结果：订阅队列 + 订阅瞬间的历史快照。

    `queue` 与 `history` 在同一个同步调用中取得，两者之间没有任何 await，
    因此不会出现“事件既不在历史里、也没进队列”的丢事件窗口。
    """

    queue: asyncio.Queue
    history: list[StreamEvent]
    finished: bool


class TaskEventBroker:
    """任务事件总线：发布 / 订阅 / 历史回放。"""

    def __init__(self, history_limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        self._history_limit = history_limit
        self._streams: dict[str, _TaskStream] = {}
        self._counter = 1

    def _get_stream(self, task_id: str) -> _TaskStream:
        """获取（或创建）任务事件流。"""
        task_id = str(task_id)
        stream = self._streams.get(task_id)
        if stream is None:
            stream = _TaskStream(history=deque(maxlen=self._history_limit))
            self._streams[task_id] = stream
        return stream

    def publish(
        self,
        task_id: str,
        event: str,
        data: dict[str, Any] | None = None,
        *,
        event_id: int | None = None,
    ) -> StreamEvent:
        """发布一条事件：写入历史并广播给所有订阅者。

        Args:
            task_id: 任务 ID
            event: 事件名
            data: 事件负载
            event_id: 可选的稳定事件 ID（通常来自数据库自增主键）。
                传入时保证与持久化事件日志使用同一套 ID。
        """
        stream = self._get_stream(task_id)
        if event_id is not None:
            resolved_id = int(event_id)
            if resolved_id >= self._counter:
                self._counter = resolved_id + 1
        else:
            resolved_id = self._counter
            self._counter += 1
        item = StreamEvent(id=resolved_id, event=event, data=data or {})
        stream.history.append(item)
        for queue in list(stream.subscribers):
            queue.put_nowait(item)
        return item

    def begin(self, task_id: str) -> None:
        """开始（或恢复）一次执行：重新打开事件流，历史事件仍然保留用于回放。"""
        stream = self._get_stream(task_id)
        stream.finished = False

    def history(self, task_id: str) -> list[StreamEvent]:
        """返回任务的历史事件（用于测试与调试）。"""
        stream = self._streams.get(str(task_id))
        return list(stream.history) if stream else []

    def is_finished(self, task_id: str) -> bool:
        """任务事件流是否已结束。"""
        stream = self._streams.get(str(task_id))
        return bool(stream and stream.finished)

    def has_stream(self, task_id: str) -> bool:
        """当前进程是否持有该任务的事件流。"""
        return str(task_id) in self._streams

    def close(self, task_id: str) -> None:
        """标记任务事件流结束，通知所有订阅者退出。"""
        stream = self._get_stream(task_id)
        stream.finished = True
        for queue in list(stream.subscribers):
            queue.put_nowait(None)

    # ------------------------------------------------------------------ #
    # 订阅
    # ------------------------------------------------------------------ #
    def attach(self, task_id: str) -> Subscription | None:
        """原子地订阅实时事件，返回队列与订阅瞬间的历史快照。

        Returns:
            当前进程存在该任务的事件流时返回 Subscription；
            否则返回 None（说明实时流不在本进程，调用方应回退到数据库事件日志）。
        """
        stream = self._streams.get(str(task_id))
        if stream is None:
            return None
        queue: asyncio.Queue = asyncio.Queue()
        stream.subscribers.add(queue)
        return Subscription(
            queue=queue,
            history=list(stream.history),
            finished=stream.finished,
        )

    def detach(self, task_id: str, queue: asyncio.Queue) -> None:
        """取消一个订阅者（幂等）。"""
        stream = self._streams.get(str(task_id))
        if stream is not None:
            stream.subscribers.discard(queue)

    async def subscribe(self, task_id: str) -> AsyncIterator[StreamEvent]:
        """订阅任务事件：先回放历史事件，再持续推送新事件。

        保留该生成器接口以兼容既有调用方；内部复用 attach/detach，
        因此订阅处的“历史快照 + 队列注册”同样是原子的。
        """
        subscription = self.attach(task_id)
        if subscription is None:
            return
        queue = subscription.queue
        try:
            for item in subscription.history:
                yield item
            if subscription.finished:
                return
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item
                if self.is_finished(task_id) and queue.empty():
                    return
        finally:
            self.detach(task_id, queue)
