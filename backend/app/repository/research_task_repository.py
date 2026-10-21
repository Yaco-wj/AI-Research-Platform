"""ResearchTask 仓储：所有针对 research_tasks / research_task_events 表的读写都集中在这里。"""

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.research_task import (
    ResearchTask,
    ResearchTaskEvent,
    ResearchTaskStatus,
    utc_now,
)


class ResearchTaskRepository:
    """研究任务仓储。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        task_id: uuid.UUID,
        thread_id: str,
        user_query: str,
        title: str | None = None,
    ) -> ResearchTask:
        """创建研究任务（初始状态 PENDING）。"""
        task = ResearchTask(
            task_id=task_id,
            thread_id=thread_id,
            user_query=user_query,
            title=title,
            status=ResearchTaskStatus.PENDING,
        )
        self._session.add(task)
        await self._session.flush()
        return task

    async def get_by_task_id(self, task_id: uuid.UUID) -> ResearchTask | None:
        """按 task_id 查询任务。"""
        result = await self._session.execute(
            select(ResearchTask).where(ResearchTask.task_id == task_id)
        )
        return result.scalar_one_or_none()

    async def list_by_statuses(
        self, statuses: list[ResearchTaskStatus]
    ) -> list[ResearchTask]:
        """按状态查询任务（用于进程启动时清理“孤儿任务”）。"""
        result = await self._session.execute(
            select(ResearchTask)
            .where(ResearchTask.status.in_(statuses))
            .order_by(ResearchTask.created_at.asc())
        )
        return list(result.scalars().all())

    async def list_tasks(self, *, limit: int = 20, offset: int = 0) -> list[ResearchTask]:
        """按创建时间倒序分页查询任务。"""
        result = await self._session.execute(
            select(ResearchTask)
            .order_by(ResearchTask.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        """统计任务总数。"""
        result = await self._session.execute(select(func.count()).select_from(ResearchTask))
        return int(result.scalar_one())

    async def mark_running(self, task: ResearchTask) -> ResearchTask:
        """把任务标记为执行中。"""
        task.status = ResearchTaskStatus.RUNNING
        task.error_message = None
        if task.started_at is None:
            task.started_at = utc_now()
        await self._session.flush()
        return task

    async def mark_waiting_user(self, task: ResearchTask, question: str) -> ResearchTask:
        """把任务标记为等待用户澄清，并保存澄清问题。"""
        task.status = ResearchTaskStatus.WAITING_USER
        task.clarification_question = question
        await self._session.flush()
        return task

    async def mark_completed(self, task: ResearchTask, report: str) -> ResearchTask:
        """把任务标记为已完成，并保存最终报告。"""
        task.status = ResearchTaskStatus.COMPLETED
        task.final_report = report
        task.completed_at = utc_now()
        task.error_message = None
        await self._session.flush()
        return task

    async def mark_failed(self, task: ResearchTask, error_message: str) -> ResearchTask:
        """把任务标记为失败，并保存错误信息（不含 traceback）。"""
        task.status = ResearchTaskStatus.FAILED
        task.error_message = error_message
        task.completed_at = utc_now()
        await self._session.flush()
        return task

    # ------------------------------------------------------------------ #
    # 事件日志（研究过程时间线）
    # ------------------------------------------------------------------ #
    async def append_event(
        self, task_id: uuid.UUID, event: str, data: dict[str, Any]
    ) -> ResearchTaskEvent:
        """追加一条研究过程事件（SSE 事件日志，可跨进程回放）。"""
        row = ResearchTaskEvent(task_id=task_id, event=event, data=data)
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_events(
        self, task_id: uuid.UUID, *, after_id: int = 0, limit: int = 2000
    ) -> list[ResearchTaskEvent]:
        """按 id 升序读取任务的历史事件（after_id 用于断点续传）。"""
        result = await self._session.execute(
            select(ResearchTaskEvent)
            .where(
                ResearchTaskEvent.task_id == task_id,
                ResearchTaskEvent.id > after_id,
            )
            .order_by(ResearchTaskEvent.id.asc())
            .limit(limit)
        )
        return list(result.scalars().all())
