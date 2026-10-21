"""后台研究任务执行器（任务生命周期的唯一 owner）。

职责边界（这是本模块最重要的设计约束）：
- 研究任务的执行由后台 asyncio 任务驱动，**与 HTTP 请求 / SSE 连接完全解耦**：
  SSE 只是订阅者，浏览器关闭、断线重连都不会取消这里的研究；
- 所有状态流转（PENDING/RUNNING/WAITING_USER/COMPLETED/FAILED）都在这里落库，
  API 层只读不写，避免“谁都能改状态”导致的竞态；
- 每条研究过程事件都会先写入 research_task_events 再广播到事件总线，
  因此重新进入详情页、甚至进程重启后，研究过程仍然可以完整回放。

生命周期闭环：
1. `start` / `resume` 创建后台任务并登记到 `_running`；
2. `_run` 消费 Agent 事件流，落库 + 发布事件；
3. 进程关闭时 `shutdown()` 取消后台任务，并把任务写成 FAILED（不会留下假 RUNNING）；
4. 进程崩溃 / 被 kill 时来不及写库，`recover_orphaned_tasks()` 会在下次启动时
   依据 LangGraph checkpoint 修正状态（已完成 -> COMPLETED，已中断 -> WAITING_USER，
   其余 -> FAILED）。

第一阶段使用进程内 asyncio 任务；要支持多副本部署时，把 `_launch` 换成
Redis / Celery 等外部队列即可，其余状态机与事件日志逻辑保持不变。
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable
from typing import Any, Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.agent_adapter.research_agent import AgentEvent, ResearchAgentAdapter
from backend.app.models.research_task import ResearchTaskStatus
from backend.app.repository.research_task_repository import ResearchTaskRepository
from backend.app.service.event_bus import TaskEventBroker

logger = logging.getLogger(__name__)

# 兜底错误信息：对外只暴露这句话，详细异常留在后端日志
FALLBACK_ERROR_MESSAGE = "研究任务执行失败，请稍后重试。"
# 进程重启 / 崩溃导致的中断（能从 checkpoint 判定出更准确的状态时不会用这条）
RESTART_INTERRUPT_MESSAGE = "服务重启导致研究任务中断，请重新发起研究。"
# 进程正常关闭时正在执行的任务
SHUTDOWN_INTERRUPT_MESSAGE = "服务正在关闭，研究任务被中断，请稍后重新发起。"
# 节点类事件（直接透传给前端）
NODE_EVENT_TYPES = {"node_started", "node_completed", "research_started", "research_completed"}
# 终态：一旦写入就不允许被任何后续流程（取消 / 迟到异常 / 重复收尾）覆盖，
# 否则会出现“详情页显示 FAILED，但任务其实已经 COMPLETED”这类状态倒退。
FINAL_STATUSES = frozenset({ResearchTaskStatus.COMPLETED, ResearchTaskStatus.FAILED})


class TaskAlreadyRunningError(Exception):
    """同一个研究任务已经在当前进程后台执行中（拒绝重复启动 / 重复 resume）。"""


class TaskService:
    """后台任务执行器：把 Agent 事件流转换成业务状态与平台事件。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        adapter: ResearchAgentAdapter,
        broker: TaskEventBroker,
    ) -> None:
        self._session_factory = session_factory
        self._adapter = adapter
        self._broker = broker
        self._running: dict[str, asyncio.Task[None]] = {}
        self._shutting_down = False

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def start(self, task_id: uuid.UUID) -> asyncio.Task[None]:
        """启动一次新的研究任务（后台执行，立即返回）。"""
        self._ensure_accepting()
        self._broker.begin(str(task_id))
        return self._launch(task_id, lambda: self._run(task_id))

    def resume(self, task_id: uuid.UUID, answer: str) -> asyncio.Task[None]:
        """用户回答澄清问题后恢复研究（后台执行，立即返回）。"""
        self._ensure_accepting()
        self._broker.begin(str(task_id))
        return self._launch(task_id, lambda: self._run(task_id, resume=answer))

    def is_running(self, task_id: uuid.UUID) -> bool:
        """判断任务是否正在本进程后台执行。"""
        running = self._running.get(str(task_id))
        return running is not None and not running.done()

    async def wait_for(self, task_id: uuid.UUID, timeout: float | None = None) -> None:
        """等待指定任务的后台执行结束（用于测试与优雅关闭）。"""
        task = self._running.get(str(task_id))
        if task is None:
            return
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def publish_status(self, task_id: uuid.UUID, status: ResearchTaskStatus) -> None:
        """对外发布“状态事件”：先落库，再广播（供应用服务在 resume 时使用）。"""
        await self._emit(
            task_id,
            "status",
            {"task_id": str(task_id), "status": status.value},
        )

    async def shutdown(self, *, timeout: float = 10.0) -> None:
        """优雅关闭：取消所有后台任务，保证不会留下假的 RUNNING 状态。

        被取消的任务会在 `_run` 的 CancelledError 分支里把自己写成 FAILED；
        极端情况下（写库失败 / 进程被 kill）由 `recover_orphaned_tasks` 兜底。
        """
        self._shutting_down = True
        tasks = [task for task in self._running.values() if not task.done()]
        if not tasks:
            return
        logger.info("正在取消 %d 个后台研究任务", len(tasks))
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:  # pragma: no cover - 防御性分支
            logger.warning("部分后台研究任务未能在 %.1fs 内结束", timeout)
        self._running.clear()

    async def recover_orphaned_tasks(self) -> int:
        """进程启动时修正“孤儿任务”（上次进程崩溃 / 被 kill 留下的 PENDING / RUNNING）。

        判定依据是 LangGraph checkpoint（Agent 侧真实状态）：
        - 已有 final_report -> COMPLETED（补回报告，避免用户以为任务丢了）；
        - 停在 interrupt 上 -> WAITING_USER（保留澄清问题，用户可以直接继续）；
        - 其余情况 -> FAILED（如实告知服务重启导致中断，而不是永远显示“正在研究”）。

        安全性：只处理 PENDING / RUNNING；COMPLETED / FAILED 是终态，
        永远不会被对账逻辑覆盖（否则会出现“已完成任务被改写”的数据事故）。

        Returns:
            被修正的任务数量。
        """
        async with self._session_factory() as session:
            repository = ResearchTaskRepository(session)
            orphans = await repository.list_by_statuses(
                [ResearchTaskStatus.PENDING, ResearchTaskStatus.RUNNING]
            )
            recovered_tasks = []
            for task in orphans:
                if task.status in FINAL_STATUSES:
                    # 双保险：即便将来 list_by_statuses 的过滤条件被改错也不会误伤终态任务
                    continue
                if self.is_running(task.task_id):
                    continue
                state = None
                try:
                    state = await self._adapter.get_state(task.thread_id)
                except Exception:  # noqa: BLE001 - checkpoint 读取失败按“无法恢复”处理
                    logger.exception("读取 Agent checkpoint 失败 task_id=%s", task.task_id)

                if state is not None and state.final_report:
                    await repository.mark_completed(task, state.final_report)
                    logger.info("孤儿任务恢复为 COMPLETED task_id=%s", task.task_id)
                elif state is not None and state.interrupt_question:
                    await repository.mark_waiting_user(task, state.interrupt_question)
                    logger.info("孤儿任务恢复为 WAITING_USER task_id=%s", task.task_id)
                else:
                    await repository.mark_failed(task, RESTART_INTERRUPT_MESSAGE)
                    logger.warning("孤儿任务标记为 FAILED task_id=%s", task.task_id)
                recovered_tasks.append(task)
            if recovered_tasks:
                await session.commit()
                for task in recovered_tasks:
                    await self._emit(
                        task.task_id,
                        "status",
                        {"task_id": str(task.task_id), "status": task.status.value},
                    )
            return len(recovered_tasks)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _ensure_accepting(self) -> None:
        """应用正在关闭时不再接受新的研究任务。"""
        if self._shutting_down:
            raise TaskAlreadyRunningError("服务正在关闭，暂时无法启动新的研究任务。")

    def _launch(
        self, task_id: uuid.UUID, coroutine_factory: Callable[[], Awaitable[None]]
    ) -> asyncio.Task[None]:
        """创建后台 asyncio 任务，并登记运行状态。

        同一个 task_id 不允许并发跑两次（例如用户重复点击 resume）：
        否则两个后台任务会同时对同一个 LangGraph thread 执行，
        既可能互相覆盖 checkpoint，也会让状态机出现竞态。
        """
        key = str(task_id)
        existing = self._running.get(key)
        if existing is not None and not existing.done():
            raise TaskAlreadyRunningError(f"研究任务已在执行中：{key}")
        task = asyncio.create_task(coroutine_factory(), name=f"research-task-{key}")
        self._running[key] = task
        task.add_done_callback(lambda _: self._running.pop(key, None))
        return task

    # ------------------------------------------------------------------ #
    # 执行流程
    # ------------------------------------------------------------------ #
    async def _run(
        self,
        task_id: uuid.UUID,
        *,
        query: str | None = None,
        resume: str | None = None,
    ) -> None:
        """执行（或恢复）一次研究任务。"""
        key = str(task_id)
        thread_id: str | None = None
        run_finished = False
        try:
            async with self._session_factory() as session:
                repository = ResearchTaskRepository(session)
                task = await repository.get_by_task_id(task_id)
                if task is None:
                    logger.warning("任务不存在，跳过执行 task_id=%s", key)
                    return
                if task.status in FINAL_STATUSES:
                    # 终态任务不允许被重新驱动（例如对账已判定 FAILED 之后又来了迟到的 resume）
                    logger.warning(
                        "任务已处于终态 %s，跳过执行 task_id=%s", task.status.value, key
                    )
                    return
                thread_id = task.thread_id
                # 首次执行使用任务里保存的问题；恢复执行使用用户回答
                run_query = task.user_query if resume is None else None#`run_query`：**新任务**用库里存的问题（`user_query`）；**恢复任务**时置 `None`（因为恢复要靠 `resume` 参数，不重复带问题）
                await repository.mark_running(task)
                await session.commit()#状态必须先落库成 RUNNING。**先落库再广播**的同一原则
            #对外广播一条 `status: RUNNING` 事件
            await self.publish_status(task_id, ResearchTaskStatus.RUNNING)

            report: str | None = None
            awaiting_user = False
            #`adapter.stream` 驱动 Agent 跑起来（内部调 `graph.astream`），并逐条 yield 真实执行事件
            #这个 `async for` 会一直等到 Agent 执行结束（或进入中断暂停）才退出。
            async for event in self._adapter.stream(thread_id, query=run_query, resume=resume):
                await self._dispatch(task_id, event)#`_dispatch`：把事件转成平台事件并落库 + 广播（节点 / 研究 / 报告 / 澄清都在这处理）
                if event.type == "report":
                    report = event.content
                elif event.type == "clarification_required":
                    # 已进入等待用户澄清：状态已写库，事件流将继续挂在 checkpoint 上
                    awaiting_user = True

            await self._finalize(task_id, thread_id, report, awaiting_user=awaiting_user)
            run_finished = True

        #任务被取消（`shutdown()` 或外部 cancel）时，**只要还没收尾，就必须落库成 FAILED**，文案区分 "服务关闭" 还是 "重启中断"
        except asyncio.CancelledError:
            # 进程关闭 / 主动取消：必须落库，避免留下永远 RUNNING 的假状态
            if not run_finished:
                await self._safe_mark_failed(
                    task_id,
                    SHUTDOWN_INTERRUPT_MESSAGE
                    if self._shutting_down
                    else RESTART_INTERRUPT_MESSAGE,
                )
            raise
        except Exception:
            if run_finished:
                # 终态已经写好，收尾阶段的异常不能把 COMPLETED 改写成 FAILED
                logger.exception("任务完成后的收尾异常（不影响终态）task_id=%s", key)
            else:
                logger.exception("研究任务执行异常 task_id=%s", key)
                await self._mark_failed(task_id, FALLBACK_ERROR_MESSAGE)

    async def _finalize(
        self,
        task_id: uuid.UUID,
        thread_id: str,
        report: str | None,
        *,
        awaiting_user: bool = False,
    ) -> None:
        """事件流结束后的收尾：区分「等待用户澄清」与「已完成」两种情况。"""
        key = str(task_id)
        if awaiting_user:
            # 已经写入 WAITING_USER（clarification_required 分支），无需重复收尾
            return

        if report:
            await self._mark_completed(task_id, report)
            return

        # 兜底：从 Agent 状态确认是否处于中断或已经产出报告
        state = await self._adapter.get_state(thread_id)
        if state.interrupt_question:
            await self._mark_waiting_user(task_id, state.interrupt_question)
            return
        if state.final_report:
            await self._mark_completed(task_id, state.final_report)
            return

        logger.error("研究任务既未完成也未等待澄清 task_id=%s", key)
        await self._mark_failed(task_id, FALLBACK_ERROR_MESSAGE)

    # ------------------------------------------------------------------ #
    # 事件分发与状态落库
    # ------------------------------------------------------------------ #
    async def _dispatch(self, task_id: uuid.UUID, event: AgentEvent) -> None:
        """把单条 Agent 事件转换为平台事件，并按需更新任务状态。"""
        if event.type in NODE_EVENT_TYPES:
            await self._emit(task_id, event.type, self._event_payload(task_id, event))
            return

        if event.type == "report":
            # Agent 当前用一次性模型调用生成报告，因此这里只有一个完整分片
            await self._emit(
                task_id,
                "report_chunk",
                {**self._event_payload(task_id, event), "content": event.content or ""},
            )
            return

        if event.type == "clarification_required":
            question = event.question or ""
            await self._emit(
                task_id,
                "clarification_required",
                {**self._event_payload(task_id, event), "question": question},
            )
            await self._mark_waiting_user(task_id, question)
            # 结束当前 SSE 连接：前端收到澄清问题后重新提问，resume 时会重新建立连接。
            # 注意：这里结束的只是「本次实时事件流」，任务本身停在 checkpoint 上，
            # 用户重新进入详情页时仍能通过历史回放看到澄清问题。
            self._broker.close(str(task_id))
            return

        logger.debug("忽略未处理的 Agent 事件类型：%s", event.type)

    @staticmethod
    def _event_payload(task_id: uuid.UUID, event: AgentEvent) -> dict[str, Any]:
        """统一的平台事件负载：始终带 task_id。"""
        payload = event.to_payload()
        payload.pop("event", None)
        payload["task_id"] = str(task_id)
        return payload

    async def _emit(self, task_id: uuid.UUID, event: str, data: dict[str, Any]) -> None:
        """发布一条事件：先持久化（可回放），再写入实时总线。

        持久化的自增主键会作为 SSE event id，因此“数据库回放”和“实时流”
        使用同一套 ID，前端 since_id 断点续传不会重复或漏事件。
        事件落库失败不应该打断研究，因此这里对异常只记录日志并降级为内存事件。
        """
        event_id: int | None = None
        try:
            async with self._session_factory() as session:
                repository = ResearchTaskRepository(session)
                row = await repository.append_event(task_id, event, data)
                await session.commit()
                event_id = row.id
        except Exception:  # noqa: BLE001 - 事件日志失败不影响研究本身
            logger.exception("研究事件落库失败，降级为仅内存推送 task_id=%s", task_id)
        self._broker.publish(task_id, event, data, event_id=event_id)

    async def _mark_waiting_user(self, task_id: uuid.UUID, question: str) -> None:
        """写入 WAITING_USER 状态并推送状态事件（终态任务不会被改写）。"""
        async with self._session_factory() as session:
            repository = ResearchTaskRepository(session)
            task = await repository.get_by_task_id(task_id)
            if task is None:
                return
            if task.status in FINAL_STATUSES:
                logger.warning(
                    "任务已是终态 %s，忽略 WAITING_USER 写入 task_id=%s",
                    task.status.value,
                    task_id,
                )
                return
            await repository.mark_waiting_user(task, question)
            await session.commit()
        await self.publish_status(task_id, ResearchTaskStatus.WAITING_USER)

    async def _mark_completed(self, task_id: uuid.UUID, report: str) -> None:
        """写入 COMPLETED 状态、保存报告，并关闭事件流（终态任务不会被改写）。"""
        async with self._session_factory() as session:
            repository = ResearchTaskRepository(session)
            task = await repository.get_by_task_id(task_id)
            if task is None:
                return
            if task.status in FINAL_STATUSES:
                logger.warning(
                    "任务已是终态 %s，忽略 COMPLETED 写入 task_id=%s",
                    task.status.value,
                    task_id,
                )
                return
            await repository.mark_completed(task, report)
            await session.commit()
        await self._emit(
            task_id,
            "completed",
            {"task_id": str(task_id), "status": ResearchTaskStatus.COMPLETED.value},
        )
        self._broker.close(str(task_id))

    async def _mark_failed(self, task_id: uuid.UUID, message: str) -> None:
        """写入 FAILED 状态并推送错误事件（对外不含 traceback）。

        这是“状态倒退”的最后一道闸门：已经 COMPLETED 的任务绝不会被改写成 FAILED。
        """
        async with self._session_factory() as session:
            repository = ResearchTaskRepository(session)
            task = await repository.get_by_task_id(task_id)
            if task is None:
                return
            if task.status in FINAL_STATUSES:
                logger.warning(
                    "任务已是终态 %s，忽略 FAILED 写入（原因：%s）task_id=%s",
                    task.status.value,
                    message,
                    task_id,
                )
                return
            await repository.mark_failed(task, message)
            await session.commit()
        await self._emit(
            task_id,
            "error",
            {
                "task_id": str(task_id),
                "status": ResearchTaskStatus.FAILED.value,
                "message": message,
            },
        )
        self._broker.close(str(task_id))

    async def _safe_mark_failed(self, task_id: uuid.UUID, message: str) -> None:
        """取消流程中的兜底落库：即使写库失败也不能把异常抛给事件循环。"""
        try:
            await self._mark_failed(task_id, message)
        except Exception:  # noqa: BLE001 - 进程正在关闭，尽力而为
            logger.exception("任务中断状态写入失败 task_id=%s", task_id)
