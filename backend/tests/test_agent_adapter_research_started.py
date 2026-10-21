"""真实 ResearchAgentAdapter 的事件映射测试。

用假模型 / 假工具驱动真实 LangGraph 图（不联网），验证事件层不会重复发送：
- researcher 子图内 researcher 节点会被多次调用
  （researcher -> researcher_tools -> researcher ...）；
- 但 research_started 事件在同一次子图运行中只应发送一次；
- 否则前端 ResearchTimeline 会把同一个子课题追加展示多次。
"""

import asyncio
from collections import Counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from backend.app.agent_adapter.research_agent import AgentEvent, ResearchAgentAdapter
from open_deep_research import deep_researcher as dr
from open_deep_research.state import ClarifyWithUser, ResearchQuestions
from open_deep_research.utils import think_tool

SUB_QUESTIONS = ["A_主题", "B_主题"]
# 压缩阶段的系统提示词标记：用于让假模型区分“压缩研究员结果”与“继续研究”
COMPRESS_MARKER = "研究整理员"


class _FakeConfigurableModel:
    """按阶段返回预设结果的假模型，并统计每个主题被研究员模型调用的次数。"""

    def __init__(self) -> None:
        self._schema = None
        self._bound_tools: list = []
        self.researcher_calls: Counter = Counter()

    def with_structured_output(self, schema, **kwargs):
        self._schema = schema
        return self

    def with_retry(self, **kwargs):
        return self

    def with_config(self, *args, **kwargs):
        return self

    def bind_tools(self, tools, **kwargs):
        self._bound_tools = list(tools)
        self._schema = None
        return self

    async def ainvoke(self, messages, config=None):
        if self._schema is ClarifyWithUser:
            return ClarifyWithUser(
                need_clarification=False, question="", verification="信息已收到。"
            )
        if self._schema is ResearchQuestions:
            return ResearchQuestions(
                research_brief="对比 A 与 B", sub_questions=list(SUB_QUESTIONS)
            )

        first_message = messages[0]
        # 最终报告阶段：提示词是 HumanMessage
        if isinstance(first_message, HumanMessage):
            return AIMessage(content="# 最终研究报告")
        if isinstance(first_message, SystemMessage):
            if COMPRESS_MARKER in first_message.content:
                return AIMessage(content="压缩后的研究结果")
            # 主管阶段：直接结束研究，保证只跑一轮种子批次
            if dr.ResearchComplete in self._bound_tools:
                return AIMessage(
                    content="信息足够",
                    tool_calls=[{
                        "name": "ResearchComplete",
                        "args": {},
                        "id": "complete-1",
                        "type": "tool_call",
                    }],
                )
            topic = next(
                (message.content for message in messages if isinstance(message, HumanMessage)),
                "?",
            )
            self.researcher_calls[topic] += 1
            if self.researcher_calls[topic] == 1:
                # 第一次先反思：使 researcher 节点在同一次子图运行内被调用两次
                return AIMessage(
                    content="先反思",
                    tool_calls=[{
                        "name": "think_tool",
                        "args": {"reflection": "规划研究计划"},
                        "id": f"think-{topic}",
                        "type": "tool_call",
                    }],
                )
            return AIMessage(content="资料已经收集充分，不再调用工具。")
        raise AssertionError(f"未知阶段的模型调用：{messages!r}")


def _collect_events() -> list[AgentEvent]:
    """执行一次真实适配器事件流（内部用 asyncio.run，测试本身保持同步）。"""
    adapter = ResearchAgentAdapter(InMemorySaver())

    async def _run() -> list[AgentEvent]:
        return [
            event
            async for event in adapter.stream("thread-adapter-dedup", query="对比 A 与 B")
        ]

    return asyncio.run(_run())


def test_research_started_emitted_once_per_researcher_run(monkeypatch):
    """researcher 节点多次执行，但 research_started 每个子课题只发一次。"""
    fake_model = _FakeConfigurableModel()
    monkeypatch.setattr(dr, "configurable_model", fake_model)

    async def fake_get_all_tools(config):
        return [think_tool]

    monkeypatch.setattr(dr, "get_all_tools", fake_get_all_tools)

    events = _collect_events()

    researcher_node_starts = [
        event
        for event in events
        if event.type == "node_started" and event.node == "researcher"
    ]
    started = [event for event in events if event.type == "research_started"]
    completed = [event for event in events if event.type == "research_completed"]

    # researcher 节点确实被多次调用（先 think、再收尾）
    assert len(researcher_node_starts) == 2 * len(SUB_QUESTIONS)
    # 但研究开始事件每个子课题只发送一次，前端不会出现重复子课题
    assert len(started) == len(SUB_QUESTIONS)
    assert Counter(event.topic for event in started) == Counter(SUB_QUESTIONS)
    assert len(completed) == len(SUB_QUESTIONS)
    assert Counter(event.topic for event in completed) == Counter(SUB_QUESTIONS)
