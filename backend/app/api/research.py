"""研究任务接口：创建 / 查询 / SSE 流式 / 恢复澄清 / 获取报告。

SSE 的定位（务必保持）：
- `GET /research/{id}/stream` **只是订阅者**，它不会创建、恢复、取消任何研究任务；
- 客户端断开连接只会移除一个订阅者，后台研究任务继续执行并持续更新数据库；
- 事件回放以数据库事件日志（research_task_events）为准，因此重新进入详情页、
  甚至后端进程重启之后，依然可以恢复完整的研究过程时间线。
"""

import asyncio
import json
import uuid
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from backend.app.dependencies import get_broker, get_research_service
from backend.app.schemas.research import (
    ResearchCreateRequest,
    ResearchCreateResponse,
    ResearchReportResponse,
    ResearchResumeRequest,
    ResearchResumeResponse,
    ResearchTaskListResponse,
    ResearchTaskResponse,
)
from backend.app.service.event_bus import TaskEventBroker
from backend.app.service.research_service import ResearchService, ResearchStateError, TaskNotFoundError

router = APIRouter(prefix="/research", tags=["research"])

# SSE 心跳间隔（秒），用于保持长连接并尽早发现断开的客户端
SSE_PING_INTERVAL = 15
# 单次从数据库读取的回放页大小（分页读取，避免长任务事件过多时被截断）
SSE_REPLAY_PAGE_SIZE = 500
# 会让服务端主动结束当前 SSE 连接的事件（前端收到后按需重连）
TERMINAL_EVENTS = {"completed", "error", "clarification_required"}


def _sse(data: dict, *, event: str, event_id: int | None = None) -> ServerSentEvent:
    """构造一条 SSE 消息；data 中同时带 event 字段，兼容只用 onmessage 的客户端。"""
    return ServerSentEvent(
        id=None if event_id is None else str(event_id),
        event=event,
        data=json.dumps({"event": event, **data}, ensure_ascii=False),
    )


@router.post(
    "",
    response_model=ResearchCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建研究任务",
    description="创建研究任务并在后台启动研究，接口立即返回，不会同步等待研究完成。",
)
async def create_research(
    payload: ResearchCreateRequest,
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchCreateResponse:
    """创建研究任务。"""
    task = await service.create_task(payload.query)
    return ResearchCreateResponse(
        task_id=task.task_id,
        thread_id=task.thread_id,
        status=task.status,
    )


@router.get(
    "",
    response_model=ResearchTaskListResponse,
    status_code=status.HTTP_200_OK,
    summary="研究历史列表",
    description="按创建时间倒序返回研究任务，用于展示基础研究历史。",
)
async def list_research(
    service: Annotated[ResearchService, Depends(get_research_service)],
    limit: Annotated[int, Query(ge=1, le=100, description="每页数量")] = 20,
    offset: Annotated[int, Query(ge=0, description="偏移量")] = 0,
) -> ResearchTaskListResponse:
    """查询研究历史。"""
    total, tasks = await service.list_tasks(limit=limit, offset=offset)
    return ResearchTaskListResponse(
        total=total,
        items=[ResearchTaskResponse.model_validate(task) for task in tasks],
    )


@router.get(
    "/{task_id}",
    response_model=ResearchTaskResponse,
    status_code=status.HTTP_200_OK,
    summary="获取研究任务",
    description="返回任务的基本信息与当前状态。",
)
async def get_research(
    task_id: uuid.UUID,
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchTaskResponse:
    """查询单个研究任务。"""
    try:
        task = await service.get_task(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ResearchTaskResponse.model_validate(task)


@router.get(
    "/{task_id}/report",
    response_model=ResearchReportResponse,
    status_code=status.HTTP_200_OK,
    summary="获取最终报告",
    description="任务未完成时返回 409，完成后返回 Markdown 报告。",
)
async def get_research_report(
    task_id: uuid.UUID,
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchReportResponse:
    """获取最终研究报告。"""
    try:
        report = await service.get_report(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ResearchStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ResearchReportResponse(task_id=task_id, report=report)


@router.post(
    "/{task_id}/resume",
    response_model=ResearchResumeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="恢复研究（回答澄清问题）",
    description="把用户回答通过 Command(resume=...) 传回同一个 LangGraph thread，继续在后台执行研究。",
)
async def resume_research(
    task_id: uuid.UUID,
    payload: ResearchResumeRequest,
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchResumeResponse:
    """回答澄清问题并恢复研究。"""
    try:
        task = await service.resume_task(task_id, payload.answer)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ResearchStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ResearchResumeResponse(
        task_id=task.task_id,
        thread_id=task.thread_id,
        status=task.status,
    )


@router.get(
    "/{task_id}/stream",
    summary="研究过程 SSE 流",
    description=(
        "以 text/event-stream 推送 Agent 的真实执行过程：status、node_started、"
        "node_completed、research_started、research_completed、clarification_required、"
        "report_chunk、completed、error。"
        "该接口只订阅事件、不影响任务生命周期；支持 since_id 断点续传。"
    ),
    response_class=EventSourceResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_research(
    task_id: uuid.UUID,
    request: Request,
    service: Annotated[ResearchService, Depends(get_research_service)],
    broker: Annotated[TaskEventBroker, Depends(get_broker)],
    since_id: Annotated[
        int,
        Query(ge=0, description="只接收 id 大于该值的事件，用于断点续传（避免重连时重复回放）"),
    ] = 0,
) -> EventSourceResponse:
    """订阅研究任务事件流（数据库回放 + 实时推送，均为只读订阅）。"""
    try:
        task = await service.get_task(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    key = str(task.task_id)
    current_status = task.status

    async def event_generator() -> AsyncIterator[ServerSentEvent]:
        """先回放持久化的研究过程，再推送实时事件；全程不触发任何 Agent 执行。"""
        sent_id = since_id
        emitted_any = False

        # attach 是同步的原子操作：调用之后发布的事件一定在 queue 里，
        # 调用之前发布的事件一定已经落库（TaskService 先落库再 publish），不会丢事件。
        subscription = broker.attach(key)
        try:
            # 1) 数据库事件日志：跨进程、跨重启的完整时间线（分页，避免被上限截断）
            while True:
                page = await service.list_events(
                    task_id, after_id=sent_id, limit=SSE_REPLAY_PAGE_SIZE
                )
                if not page:
                    break
                for item in page:
                    if item.id <= sent_id:
                        continue
                    sent_id = item.id
                    emitted_any = True
                    yield _sse(item.data, event=item.event, event_id=item.id)
                if len(page) < SSE_REPLAY_PAGE_SIZE:
                    break

            if subscription is None:
                # 当前进程没有该任务的实时事件流（例如服务重启过）：
                # 数据库回放已经尽力还原了历史，这里如实告知前端并结束连接。
                if not emitted_any:
                    yield _sse(
                        {"task_id": key, "status": current_status.value},
                        event="status",
                    )
                    yield _sse(
                        {
                            "task_id": key,
                            "status": current_status.value,
                            "message": "当前进程没有该任务的实时事件流，请通过任务查询接口获取状态。",
                        },
                        event="stream_unavailable",
                    )
                return

            # 2) 订阅瞬间的内存历史快照（与数据库回放按 id 去重）
            for item in subscription.history:
                if item.id <= sent_id:
                    continue
                sent_id = item.id
                emitted_any = True
                yield _sse(item.data, event=item.event, event_id=item.id)

            if subscription.finished:
                # 本次事件流已结束（完成 / 失败 / 等待澄清），前端按需重连
                return

            # 3) 实时事件：只读地消费订阅队列
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(subscription.queue.get(), timeout=SSE_PING_INTERVAL)
                except asyncio.TimeoutError:
                    # 空闲时周期性检查客户端是否已断开（保持连接由 sse-starlette ping 负责）
                    continue
                if item is None:
                    break
                if item.id <= sent_id:
                    continue
                sent_id = item.id
                yield _sse(item.data, event=item.event, event_id=item.id)
                if item.event in TERMINAL_EVENTS:
                    break
        finally:
            if subscription is not None:
                broker.detach(key, subscription.queue)

    return EventSourceResponse(event_generator(), ping=SSE_PING_INTERVAL)
