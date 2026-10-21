"""研究任务相关的 Pydantic 模型（与前端 TypeScript 类型保持一致）。"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.app.models.research_task import ResearchTaskStatus


class ResearchCreateRequest(BaseModel):
    """创建研究任务请求。"""

    query: str = Field(min_length=1, max_length=8000, description="用户想深入研究的问题")


class ResearchCreateResponse(BaseModel):
    """创建研究任务响应。"""

    task_id: uuid.UUID = Field(description="研究任务 ID")
    thread_id: str = Field(description="LangGraph thread_id（与 task_id 对应）")
    status: ResearchTaskStatus = Field(description="任务状态，创建后为 PENDING")


class ResearchTaskResponse(BaseModel):
    """研究任务详情响应。"""

    model_config = ConfigDict(from_attributes=True)

    task_id: uuid.UUID
    thread_id: str
    query: str = Field(validation_alias="user_query", description="用户研究问题")
    title: str | None = None
    status: ResearchTaskStatus
    clarification_question: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ResearchTaskListResponse(BaseModel):
    """研究历史列表响应。"""

    total: int = Field(description="任务总数")
    items: list[ResearchTaskResponse] = Field(description="按创建时间倒序的任务列表")


class ResearchResumeRequest(BaseModel):
    """回答澄清问题、恢复研究请求。"""

    answer: str = Field(min_length=1, max_length=8000, description="用户对澄清问题的回答")


class ResearchReportResponse(BaseModel):
    """研究报告响应。"""

    task_id: uuid.UUID
    report: str = Field(description="Markdown 格式的最终研究报告")


class ResearchResumeResponse(BaseModel):
    """恢复研究响应。"""

    task_id: uuid.UUID
    thread_id: str
    status: ResearchTaskStatus


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(default="ok", description="服务状态")
