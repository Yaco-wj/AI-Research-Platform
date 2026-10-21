"""Agent 适配器单元测试：验证 LangGraph 流数据到平台事件的映射。"""

import pytest
from langgraph.types import Command, Interrupt

from backend.app.agent_adapter import research_agent as research_agent_module
from backend.app.agent_adapter.research_agent import ResearchAgentAdapter


class FakeSnapshot:
    """模拟 LangGraph StateSnapshot。"""

    def __init__(self, values=None, next_nodes=(), interrupts=()):
        self.values = values or {}
        self.next = next_nodes
        self.interrupts = interrupts


class FakeGraph:
    """模拟编译后的 LangGraph 图。"""

    def __init__(self, chunks=(), snapshot=None):
        self._chunks = list(chunks)
        self._snapshot = snapshot or FakeSnapshot()
        self.inputs = []
        self.configs = []
        self.stream_modes = []

    async def astream(self, agent_input, config, stream_mode=None, subgraphs=False):
        """按顺序回放预设的 stream chunk。"""
        self.inputs.append(agent_input)
        self.configs.append(config)
        self.stream_modes.append((stream_mode, subgraphs))
        for chunk in self._chunks:
            yield chunk

    async def aget_state(self, config):
        """返回预设快照。"""
        return self._snapshot


def _patch_graph(monkeypatch, chunks=(), snapshot=None):
    """把适配器内部的 build_deep_researcher 替换为假图。"""
    graph = FakeGraph(chunks, snapshot)
    monkeypatch.setattr(
        research_agent_module, "build_deep_researcher", lambda checkpointer=None: graph
    )
    return ResearchAgentAdapter(checkpointer=object()), graph


async def _collect(adapter, **kwargs):
    """收集适配器产出的全部事件。"""
    return [event async for event in adapter.stream("thread-1", **kwargs)]


async def test_stream_maps_tasks_and_updates_to_platform_events(monkeypatch):
    """tasks / updates 两类 chunk 都能映射为平台事件。"""
    namespace = ("research_supervisor:abc", "supervisor_tools:def", "1")
    chunks = [
        ((), "tasks", {"id": "1", "name": "clarify_with_user", "input": {"messages": []}}),
        ((), "updates", {"clarify_with_user": {"messages": []}}),
        ((), "updates", {"clarification_question": "忽略"}),
        (
            (),
            "updates",
            {"__interrupt__": [Interrupt(value={"type": "clarification", "question": "想重点研究哪些方面？"}, id="i1")]},
        ),
        (namespace, "tasks", {"id": "2", "name": "researcher", "input": {"research_topic": "研究 LangGraph"}}),
        (namespace, "updates", {"compress_research": {"compressed_research": "LangGraph 的研究结论"}}),
        ((), "updates", {"final_report_generation": {"final_report": "# 报告\n\n正文"}}),
        ((), "tasks", {"id": "3", "name": "final_report_generation", "error": None, "result": None}),
    ]
    adapter, graph = _patch_graph(monkeypatch, chunks)

    events = await _collect(adapter, query="研究 2026 年 AI Agent 趋势")

    # 输入的构造方式：dict + HumanMessage
    agent_input = graph.inputs[0]
    assert agent_input["messages"][0].content == "研究 2026 年 AI Agent 趋势"
    assert graph.configs[0] == {"configurable": {"thread_id": "thread-1"}}
    # 子图事件必须开启，否则拿不到 researcher / supervisor 节点
    assert graph.stream_modes[0] == (["tasks", "updates"], True)

    types = [event.type for event in events]
    assert "node_started" in types
    assert "node_completed" in types
    assert "clarification_required" in types
    assert "research_started" in types
    assert "research_completed" in types
    assert "report" in types

    clarification = next(e for e in events if e.type == "clarification_required")
    assert clarification.question == "想重点研究哪些方面？"

    research_started = next(e for e in events if e.type == "research_started")
    assert research_started.topic == "研究 LangGraph"
    assert research_started.scope == "researcher"

    research_completed = next(e for e in events if e.type == "research_completed")
    # 通过命名空间把压缩结果关联回具体研究主题
    assert research_completed.topic == "研究 LangGraph"
    assert research_completed.summary == "LangGraph 的研究结论"

    report = next(e for e in events if e.type == "report")
    assert report.content.startswith("# 报告")
    # 事件负载中不包含 Agent 内部 State 的原始字段
    assert all("supervisor_messages" not in event.to_payload() for event in events)


async def test_stream_resume_uses_command_resume(monkeypatch):
    """恢复执行必须使用 Command(resume=...) 与同一个 thread_id。"""
    adapter, graph = _patch_graph(
        monkeypatch,
        [((), "updates", {"final_report_generation": {"final_report": "# 报告"}})],
    )

    await _collect(adapter, resume="重点关注企业级落地")

    agent_input = graph.inputs[0]
    assert isinstance(agent_input, Command)
    assert agent_input.resume == "重点关注企业级落地"
    assert graph.configs[0] == {"configurable": {"thread_id": "thread-1"}}


async def test_stream_falls_back_to_agent_state(monkeypatch):
    """流里没有报告 / 中断时，通过 aget_state 兜底读取。"""
    snapshot = FakeSnapshot(
        values={"final_report": "# 兜底报告"},
        next_nodes=(),
    )
    adapter, _ = _patch_graph(monkeypatch, [], snapshot)

    events = await _collect(adapter, query="问题")

    assert [event.type for event in events] == ["report"]
    assert events[0].content == "# 兜底报告"

    interrupt_snapshot = FakeSnapshot(
        values={},
        next_nodes=("wait_for_user_clarification",),
        interrupts=(Interrupt(value={"question": "还需要澄清什么？"}, id="i2"),),
    )
    adapter, _ = _patch_graph(monkeypatch, [], interrupt_snapshot)

    events = await _collect(adapter, query="问题")

    assert [event.type for event in events] == ["clarification_required"]
    assert events[0].question == "还需要澄清什么？"


async def test_get_state_reads_agent_state(monkeypatch):
    """get_state 只读取 Backend 需要的字段。"""
    snapshot = FakeSnapshot(
        values={
            "final_report": "# 报告",
            "research_brief": "研究简报",
            "sub_questions": ["子问题 1"],
        },
        next_nodes=("final_report_generation",),
    )
    adapter, _ = _patch_graph(monkeypatch, [], snapshot)

    state = await adapter.get_state("thread-9")

    assert state.thread_id == "thread-9"
    assert state.has_checkpoint is True
    assert state.next_nodes == ("final_report_generation",)
    assert state.final_report == "# 报告"
    assert state.research_brief == "研究简报"
    assert state.sub_questions == ["子问题 1"]
    assert state.waiting_for_user is False


def test_build_input_requires_exactly_one_source(monkeypatch):
    """query 与 resume 必须且只能提供一个。"""
    adapter, _ = _patch_graph(monkeypatch)

    with pytest.raises(ValueError):
        adapter._build_input("query", "resume")
    with pytest.raises(ValueError):
        adapter._build_input(None, None)
