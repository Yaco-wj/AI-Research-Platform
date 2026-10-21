"""研究任务应用服务：面向 API 的用例编排（创建 / 查询 / 恢复 / 取报告 / 事件回放）。"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent_adapter.research_agent import ResearchAgentAdapter
from backend.app.models.research_task import ResearchTask, ResearchTaskStatus
from backend.app.repository.research_task_repository import ResearchTaskRepository
from backend.app.service.event_bus import TaskEventBroker
from backend.app.service.task_service import TaskAlreadyRunningError, TaskService

logger = logging.getLogger(__name__)

TITLE_MAX_LENGTH = 60


@dataclass(slots=True)
class StoredEvent:
    """可回放的研究过程事件（来自数据库事件日志）。"""

    id: int
    event: str
    data: dict[str, Any]


class TaskNotFoundError(Exception):
    """研究任务不存在。"""


class ResearchStateError(Exception):
    """当前任务状态不允许该操作（例如未完成就获取报告）。"""


class ResearchService:
    """研究任务服务。

    该类只做用例编排，不直接执行 Agent：
    - 创建任务后交给 `TaskService` 在后台执行；
    - 恢复任务同样交给 `TaskService`，API 请求不会阻塞在 Agent 上。
    """

    def __init__(
        self,
        session: AsyncSession,
        task_service: TaskService,
        broker: TaskEventBroker,
        adapter: ResearchAgentAdapter,
    ) -> None:
        self._session = session
        self._repository = ResearchTaskRepository(session)
        self._task_service = task_service
        self._broker = broker
        self._adapter = adapter

    async def create_task(self, query: str) -> ResearchTask:
        """创建研究任务并立即在后台启动研究（不阻塞 HTTP 请求）。"""
        task_id = uuid.uuid4()
        # thread_id 与 task_id 一一对应，便于通过 task_id 直接恢复 LangGraph 执行
        thread_id = str(task_id)
        task = await self._repository.create(
            task_id=task_id,
            thread_id=thread_id,
            user_query=query,
            title=self._build_title(query),
        )
        await self._session.commit()

        # 关键：研究在后台 asyncio 任务里执行，HTTP 请求立刻返回。
        # 之后即使浏览器关闭、SSE 断开，研究也不会被取消。
        self._task_service.start(task.task_id)
        return task

    async def get_task(self, task_id: uuid.UUID) -> ResearchTask:
        """查询任务，不存在时抛出 TaskNotFoundError。"""
        task = await self._repository.get_by_task_id(task_id)
        if task is None:
            raise TaskNotFoundError(f"研究任务不存在：{task_id}")
        return task

    async def list_tasks(
        self, *, limit: int = 20, offset: int = 0
    ) -> tuple[int, list[ResearchTask]]:
        """查询研究历史（按创建时间倒序）。"""
        total = await self._repository.count()
        tasks = await self._repository.list_tasks(limit=limit, offset=offset)
        return total, tasks

    async def get_report(self, task_id: uuid.UUID) -> str:
        """获取最终报告；任务未完成时抛出 ResearchStateError。"""
        task = await self.get_task(task_id)
        if task.status != ResearchTaskStatus.COMPLETED or not task.final_report:
            raise ResearchStateError(f"研究任务尚未完成，当前状态：{task.status.value}")
        return task.final_report

    async def resume_task(self, task_id: uuid.UUID, answer: str) -> ResearchTask:
        """用户回答澄清问题后恢复研究。"""
        task = await self.get_task(task_id)
        if task.status != ResearchTaskStatus.WAITING_USER:
            raise ResearchStateError("当前任务不处于等待用户澄清状态，无法恢复。")
        if self._task_service.is_running(task.task_id):
            # 重复提交（例如前端重复点击）不能再次驱动同一个 LangGraph thread
            raise ResearchStateError("任务已经在恢复执行中，请勿重复提交。")

        await self._repository.mark_running(task)
        await self._session.commit()
        # 状态事件同样走“先落库再广播”，保证事件日志完整
        await self._task_service.publish_status(task.task_id, ResearchTaskStatus.RUNNING)

        try:
            self._task_service.resume(task.task_id, answer)
        except TaskAlreadyRunningError as exc:
            raise ResearchStateError("任务已经在恢复执行中，请勿重复提交。") from exc
        return task

    async def get_pending_question(self, task_id: uuid.UUID) -> str | None:
        """获取待回答的澄清问题：优先读业务表，其次读 Agent 状态。"""
        task = await self.get_task(task_id)
        if task.status != ResearchTaskStatus.WAITING_USER:
            return None
        if task.clarification_question:
            return task.clarification_question
        state = await self._adapter.get_state(task.thread_id)
        return state.interrupt_question

    async def list_events(
        self, task_id: uuid.UUID, *, after_id: int = 0, limit: int = 2000
    ) -> list[StoredEvent]:
        """读取持久化的研究过程事件（SSE 回放来源，支持断点续传）。"""
        rows = await self._repository.list_events(task_id, after_id=after_id, limit=limit)
        return [
            StoredEvent(id=row.id, event=row.event, data=dict(row.data or {}))
            for row in rows
        ]

    @staticmethod
    def _build_title(query: str) -> str:
        """用问题前若干字符生成任务标题。"""
        normalized = " ".join(query.split())
        return normalized[:TITLE_MAX_LENGTH] if normalized else "未命名研究任务"
