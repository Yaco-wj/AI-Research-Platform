"""Deep Research Agent 适配器。

这是 Backend 与 Agent 之间唯一的适配层，职责：
1. 使用 checkpointer 编译 Agent 主图；
2. 构造 Agent 需要的输入（首次执行 / Command(resume=...) 恢复）；
3. 调用 graph.astream 把 Agent 的真实执行过程转换成平台事件（AgentEvent）；
4. 通过 graph.aget_state 读取 Agent 状态（中断、最终报告）。

本模块之外（service / api / repository）不允许直接 import
open_deep_research 内部的 supervisor / researcher 节点，也不允许读写 Agent State。
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from open_deep_research.deep_researcher import build_deep_researcher

logger = logging.getLogger(__name__)

# 研究员相关节点（位于 researcher 子图内）
RESEARCHER_NODES = {"researcher", "researcher_tools", "compress_research"}
# 主管相关节点（位于 supervisor 子图内）
SUPERVISOR_NODES = {"supervisor", "supervisor_tools"}
# 被压缩后的研究摘要回传时，写入事件的摘要最大长度
SUMMARY_MAX_LENGTH = 300


@dataclass(slots=True)
class AgentEvent:
    """平台层事件：Agent 真实执行过程的一次映射结果。"""

    type: str
    node: str | None = None
    scope: str = "main"
    topic: str | None = None
    summary: str | None = None
    question: str | None = None
    content: str | None = None
    namespace: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        """转换为可序列化的事件负载（忽略空字段）。"""
        payload: dict[str, Any] = {"event": self.type}
        if self.node:
            payload["node"] = self.node
        if self.scope and self.scope != "main":
            payload["scope"] = self.scope
        if self.topic:
            payload["topic"] = self.topic
        if self.summary:
            payload["summary"] = self.summary
        if self.question:
            payload["question"] = self.question
        if self.content:
            payload["content"] = self.content
        return payload


@dataclass(slots=True)
class AgentStateView:
    """Agent 侧状态视图：只暴露 Backend 需要的少量信息。"""

    thread_id: str
    has_checkpoint: bool
    next_nodes: tuple[str, ...]
    interrupt_question: str | None = None
    final_report: str | None = None
    research_brief: str | None = None
    sub_questions: list[str] = field(default_factory=list)

    @property
    def waiting_for_user(self) -> bool:
        """Agent 是否正在等待用户回答澄清问题。"""
        return self.interrupt_question is not None


class ResearchAgentAdapter:
    """把 Agent 的 LangGraph 执行过程适配为平台可消费的事件流。"""

    def __init__(self, checkpointer: Any) -> None:
        # 编译后的图随应用生命周期复用，checkpointer 由 core/checkpoint.py 提供
        self._graph = build_deep_researcher(checkpointer=checkpointer)

    # ------------------------------------------------------------------ #
    # 配置与输入
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_config(thread_id: str) -> dict[str, Any]:
        """构造 LangGraph 运行配置（thread_id 与 ResearchTask.task_id 一一对应）。"""
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _build_input(query: str | None, resume: str | None) -> dict[str, Any] | Command:
        """构造 Agent 输入：首次执行使用用户问题，恢复执行使用 Command(resume=...)。"""
        if query is not None and resume is not None:
            raise ValueError("query 与 resume 不能同时提供。")
        if query is None and resume is None:
            raise ValueError("必须提供 query 或 resume 其中之一。")
        if resume is not None:
            return Command(resume=resume)
        return {"messages": [HumanMessage(content=str(query))]}

    # ------------------------------------------------------------------ #
    # 事件流
    # ------------------------------------------------------------------ #
    async def stream(
        self,
        thread_id: str,
        *,
        query: str | None = None,
        resume: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """流式执行 Agent，并产出真实的平台事件。

        Args:
            thread_id: LangGraph 线程 ID（等于 ResearchTask.task_id）
            query: 首次研究的问题
            resume: 用户对澄清问题的回答（恢复执行时使用）

        Yields:
            AgentEvent：节点开始 / 结束、研究派发、澄清请求、最终报告
        """
        config = self.build_config(thread_id)
        agent_input = self._build_input(query, resume)

        # 记录“子图命名空间 -> 研究主题”，用于把压缩结果关联回具体子课题
        topic_by_namespace: dict[tuple[str, ...], str] = {}
        # 记录已经发送过 research_started 的子图命名空间。
        # researcher 子图内 researcher 节点会被多次调用
        #（researcher -> researcher_tools -> researcher ...），同一次子图运行
        # 只应对外发送一次研究开始事件，避免前端重复展示同一子课题。
        started_research_namespaces: set[tuple[str, ...]] = set()# 已发过 research_started 的命名空间（去重）
        report_emitted = False# 报告是否已在流中出现
        interrupt_question: str | None = None# 澄清问题是否已在流中出现

        async for namespace, mode, payload in self._iter_stream(agent_input, config):
            if not isinstance(payload, dict):
                continue

            if mode == "tasks":
                async for event in self._handle_task_chunk(
                    namespace, payload, topic_by_namespace, started_research_namespaces
                ):
                    yield event
                continue

            if mode == "updates":
                for event in self._handle_update_chunk(namespace, payload, topic_by_namespace):
                    if event.type == "report":
                        report_emitted = True
                    if event.type == "clarification_required":
                        interrupt_question = event.question
                    yield event

        # 兜底：某些中断 / 报告不会出现在 updates 中，直接读取 Agent 状态
        state = await self.get_state(thread_id)
        if interrupt_question is None and state.interrupt_question:
            yield AgentEvent(
                type="clarification_required",
                node="wait_for_user_clarification",
                question=state.interrupt_question,
            )
        elif state.final_report and not report_emitted:
            yield AgentEvent(type="report", node="final_report_generation", content=state.final_report)

    async def _iter_stream(
        self, agent_input: dict[str, Any] | Command, config: dict[str, Any]
    ) -> AsyncIterator[tuple[tuple[str, ...], str, Any]]:
        """统一 astream 输出形状为 (namespace, mode, payload)。"""
        stream = self._graph.astream(
            agent_input,
            config,
            stream_mode=["tasks", "updates"],
            subgraphs=True,
        )
        async for chunk in stream:
            if isinstance(chunk, tuple) and len(chunk) == 3:
                namespace, mode, payload = chunk
                yield tuple(namespace or ()), str(mode), payload
            elif isinstance(chunk, tuple) and len(chunk) == 2:
                mode, payload = chunk
                yield (), str(mode), payload
            else:  # pragma: no cover - 防御性分支
                logger.debug("忽略未知 stream chunk: %r", chunk)

    async def _handle_task_chunk(
        self,
        namespace: tuple[str, ...],
        payload: dict[str, Any],
        topic_by_namespace: dict[tuple[str, ...], str],
        started_research_namespaces: set[tuple[str, ...]],
    ) -> AsyncIterator[AgentEvent]:
        """把 tasks 模式的 chunk 转换成节点事件。"""
        name = payload.get("name")
        if not name:
            return
        is_start = "input" in payload and "result" not in payload

        if is_start:
            yield AgentEvent(
                type="node_started",
                node=str(name),
                scope=self._resolve_scope(namespace, str(name)),
                namespace=namespace,
            )
            # researcher 子图的任务输入里带有真实研究主题
            if name == "researcher":
                node_input = payload.get("input")
                topic = node_input.get("research_topic") if isinstance(node_input, dict) else None
                if topic:
                    topic_by_namespace[namespace] = str(topic)
                    # 同一次 researcher 子图运行只发送一次 research_started
                    if namespace not in started_research_namespaces:
                        started_research_namespaces.add(namespace)
                        yield AgentEvent(
                            type="research_started",
                            node="researcher",
                            scope="researcher",
                            topic=str(topic),
                            namespace=namespace,
                        )
            return

        error = payload.get("error")
        if error is not None:
            logger.warning("Agent 节点执行出错 node=%s error=%s", name, error)
        yield AgentEvent(
            type="node_completed",
            node=str(name),
            scope=self._resolve_scope(namespace, str(name)),
            summary=None if error is None else f"节点执行出错：{str(error)[:200]}",
            namespace=namespace,
        )

    def _handle_update_chunk(
        self,
        namespace: tuple[str, ...],
        payload: dict[str, Any],
        topic_by_namespace: dict[tuple[str, ...], str],
    ) -> list[AgentEvent]:
        """把 updates 模式的 chunk 转换成中断 / 报告 / 研究员完成事件。"""
        events: list[AgentEvent] = []

        # 1. 中断：wait_for_user_clarification 内调用 interrupt() 时出现
        if "__interrupt__" in payload:
            question = self._extract_interrupt_question(payload["__interrupt__"])
            if question:
                events.append(
                    AgentEvent(
                        type="clarification_required",
                        node="wait_for_user_clarification",
                        question=question,
                        namespace=namespace,
                    )
                )
            return events

        # 2. 节点输出：只提取平台需要的字段，绝不透传完整 Agent State
        for node, update in payload.items():
            if not isinstance(update, dict):
                continue
            if node == "final_report_generation" and update.get("final_report"):
                events.append(
                    AgentEvent(
                        type="report",
                        node="final_report_generation",
                        content=str(update["final_report"]),
                        namespace=namespace,
                    )
                )
            elif node == "compress_research" and update.get("compressed_research"):
                summary = str(update["compressed_research"])[:SUMMARY_MAX_LENGTH]
                events.append(
                    AgentEvent(
                        type="research_completed",
                        node="compress_research",
                        scope="researcher",
                        topic=topic_by_namespace.get(namespace),
                        summary=summary,
                        namespace=namespace,
                    )
                )
        return events

    @staticmethod
    def _extract_interrupt_question(interrupts: Any) -> str | None:
        """从 interrupt 载荷中提取澄清问题。"""
        items = interrupts if isinstance(interrupts, (list, tuple)) else [interrupts]
        for item in items:
            value = getattr(item, "value", item)
            if isinstance(value, dict):
                question = value.get("question") or value.get("message")
                if question:
                    return str(question)
            elif isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _resolve_scope(namespace: tuple[str, ...], node: str) -> str:
        """判断节点属于主图 / 主管子图 / 研究员子图。"""
        if node in RESEARCHER_NODES:
            return "researcher"
        if node in SUPERVISOR_NODES:
            return "supervisor"
        return "main"

    # ------------------------------------------------------------------ #
    # 状态读取
    # ------------------------------------------------------------------ #
    async def get_state(self, thread_id: str) -> AgentStateView:
        """读取 Agent 当前状态（只读，不修改 Agent State）。"""
        config = self.build_config(thread_id)
        snapshot = await self._graph.aget_state(config)
        values = getattr(snapshot, "values", None) or {}
        interrupt_question = self._extract_interrupt_question(
            getattr(snapshot, "interrupts", None) or ()
        )
        return AgentStateView(
            thread_id=thread_id,
            has_checkpoint=bool(values),
            next_nodes=tuple(getattr(snapshot, "next", ()) or ()),
            interrupt_question=interrupt_question,
            final_report=values.get("final_report"),
            research_brief=values.get("research_brief"),
            sub_questions=list(values.get("sub_questions") or []),
        )
