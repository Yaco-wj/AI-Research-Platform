"""研究任务（ResearchTask）数据表模型。"""

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.models.base import Base


class ResearchTaskStatus(str, enum.Enum):
    """研究任务状态。"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_USER = "WAITING_USER"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def utc_now() -> datetime:
    """返回带时区的当前 UTC 时间。"""
    return datetime.now(timezone.utc)


class ResearchTask(Base):
    """研究任务：平台层对一次 Deep Research 执行的业务记录。

    task_id 与 LangGraph 的 thread_id 一一对应（thread_id 直接使用 task_id 的字符串形式），
    Agent 的研究状态保存在 LangGraph checkpoint 中，本表只保存平台层需要的业务字段。
    """

    __tablename__ = "research_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), unique=True, index=True, default=uuid.uuid4, nullable=False
    )
    thread_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[ResearchTaskStatus] = mapped_column(
        Enum(ResearchTaskStatus, native_enum=False, length=32, validate_strings=True),
        default=ResearchTaskStatus.PENDING,
        nullable=False,
        index=True,
    )
    final_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    clarification_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        """调试用字符串表示。"""
        return f"<ResearchTask task_id={self.task_id} status={self.status}>"


class ResearchTaskEvent(Base):
    """研究任务的事件流（研究过程的持久化时间线）。

    为什么需要这张表：
    - 进程内事件总线（TaskEventBroker）只服务于“当前进程仍然活着”的实时订阅，
      用户重新进入任务详情页、服务重启之后，内存里的历史事件就没了；
    - 把研究过程同时落库后，SSE 可以把数据库当作可回放的事件日志，
      因此“重新打开详情页”一定能够恢复：当前状态、已完成阶段、
      clarification 信息与 research timeline。

    `id` 是全局自增主键，同时作为 SSE 的 event id 使用，
    这样前端记录的 `lastEventId`（断点续传）在“内存实时流”和“数据库回放”
    两条路径上完全一致，不会重复推送已处理事件。
    """

    __tablename__ = "research_task_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), index=True, nullable=False
    )
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    # 事件负载（不含 event 字段），与 SSE data 一一对应
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )

    def __repr__(self) -> str:
        """调试用字符串表示。"""
        return f"<ResearchTaskEvent id={self.id} task_id={self.task_id} event={self.event}>"
