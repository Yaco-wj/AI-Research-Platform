"""Deep Research 主流程测试。

全部测试都使用假模型与假工具，不会发起任何真实的大模型或搜索 API 调用。
覆盖：澄清分流、interrupt 暂停与恢复、问题拆解、主管并发派发研究任务、
研究员结果回传、主管继续研究 / 结束研究，以及最终报告生成。
"""

import asyncio
from collections import deque
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from open_deep_research import deep_researcher as dr
from open_deep_research.configuration import Configuration
from open_deep_research.state import (
    ClarifyWithUser,
    ResearchQuestions,
    SupervisorOutputState,
    SupervisorState,
)
from open_deep_research.utils import think_tool

# 提示词标记：用于让假模型判断当前处于哪个阶段
COMPRESS_MARKER = "研究整理员"
DECOMPOSE_BRIEF = "整体研究简报：梳理 2026 年 AI Agent 框架发展趋势"
SUB_QUESTIONS = ["研究 LangGraph", "研究 CrewAI", "研究 AutoGen", "研究 MCP"]
CLARIFY_QUESTION = "请进一步说明你希望重点研究哪些方面？"


class FakeConfigurableModel:
    """测试用假模型：按结构化输出类型 / 绑定工具 / 提示词区分阶段并返回预设结果。"""

    def __init__(self, structured, responder):
        self._structured = structured
        self._responder = responder
        self._schema = None
        self._bound_tools = []
        # 记录被请求过的结构化输出类型，便于断言“是否进入某个阶段”
        self.structured_requests = []

    # 以下方法用于模拟 LangChain 的链式调用，均返回自身
    def with_structured_output(self, schema, **kwargs):
        self._schema = schema
        self.structured_requests.append(schema)
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
        if self._schema is not None:
            return self._structured[self._schema]
        return await self._responder(self._classify(messages), list(messages))

    def _classify(self, messages):
        """判断当前是哪个阶段在调用模型。"""
        first_message = messages[0]
        # 最终报告阶段：提示词是 HumanMessage
        if isinstance(first_message, HumanMessage):
            return "final_report"
        if isinstance(first_message, SystemMessage):
            # 压缩阶段使用压缩系统提示词
            if COMPRESS_MARKER in first_message.content:
                return "compress"
            # 主管模型绑定了 ResearchComplete 工具
            if any(tool is dr.ResearchComplete for tool in self._bound_tools):
                return "supervisor"
            return "researcher"
        return "unknown"


class Harness:
    """把假模型、假工具与图组装到一起，供测试使用。"""

    def __init__(
        self,
        monkeypatch,
        clarify_response: ClarifyWithUser,
        decompose_response: ResearchQuestions,
        supervisor_responses=None,
        fail_topics=None,
    ):
        self.clarify_response = clarify_response
        self.decompose_response = decompose_response
        self.supervisor_responses = deque(supervisor_responses or [])
        # 需要模拟“研究员执行失败”的研究主题
        self.fail_topics = set(fail_topics or [])
        # 统计并发研究员数量，用于验证“多个 researcher 并发执行”
        self.active_researchers = 0
        self.max_active_researchers = 0
        # 记录每个研究主题被研究员模型调用的次数
        self.researcher_calls = {}
        # 记录每个研究主题实际启动了几次 researcher 子图运行
        self.researcher_runs = {}
        self.researcher_run_topics = []
        # 记录主管派发过的研究主题
        self.conducted_topics = []
        # 记录主管模型（LLM 决策轮）被调用的次数
        self.supervisor_calls = 0
        # 记录每次主管模型调用时看到的完整消息（用于验证研究结果回传）
        self.supervisor_seen_messages = []
        self.final_report_calls = 0

        self.fake_model = FakeConfigurableModel(
            structured={
                ClarifyWithUser: clarify_response,
                ResearchQuestions: decompose_response,
            },
            responder=self._respond,
        )
        monkeypatch.setattr(dr, "configurable_model", self.fake_model)

        async def fake_get_all_tools(config):
            # 只提供 think_tool，避免测试触碰真实搜索 API
            return [think_tool]

        monkeypatch.setattr(dr, "get_all_tools", fake_get_all_tools)

    async def _respond(self, stage, messages):
        if stage == "supervisor":
            self.supervisor_calls += 1
            self.supervisor_seen_messages.append(list(messages))
            if self.supervisor_responses:
                return self.supervisor_responses.popleft()
            # 默认：主管认为信息已经足够，调用 ResearchComplete 结束研究
            return AIMessage(
                content="信息已经足够，结束研究阶段。",
                tool_calls=[{
                    "name": "ResearchComplete",
                    "args": {},
                    "id": "complete-1",
                    "type": "tool_call",
                }],
            )

        if stage == "researcher":
            topic = self._topic(messages)
            if topic not in self.researcher_calls:
                # 每个研究主题只记录一次“被派发”
                self.conducted_topics.append(topic)
            self.researcher_calls[topic] = self.researcher_calls.get(topic, 0) + 1
            # 一次 researcher 子图运行内会调用模型两次（先 think、再收尾），
            # 只有首条非系统消息代表“启动了一次全新的研究”
            conversation = [message for message in messages if not isinstance(message, SystemMessage)]
            if len(conversation) == 1:
                self.researcher_runs[topic] = self.researcher_runs.get(topic, 0) + 1
                self.researcher_run_topics.append(topic)
            self.active_researchers += 1
            self.max_active_researchers = max(self.max_active_researchers, self.active_researchers)
            try:
                # 制造并发窗口：如果研究员是并发执行的，这里会同时有多个协程处于活动状态
                await asyncio.sleep(0.02)
            finally:
                self.active_researchers -= 1

            # 模拟该主题的研究执行失败：用于验证失败主题不会写入 researched_topics
            if topic in self.fail_topics:
                raise RuntimeError(f"模拟研究员执行失败：{topic}")

            if self.researcher_calls[topic] == 1:
                # 第一次先调用 think_tool 反思，验证 researcher <-> researcher_tools 循环
                return AIMessage(
                    content="先反思一下研究计划。",
                    tool_calls=[{
                        "name": "think_tool",
                        "args": {"reflection": f"研究 {topic} 的计划"},
                        "id": f"think-{topic}",
                        "type": "tool_call",
                    }],
                )
            return AIMessage(content="资料已经收集充分，不再调用工具。")

        if stage == "compress":
            topic = self._topic(messages)
            return AIMessage(content=f"【{topic}】压缩后的完整研究结果，含来源引用。")

        if stage == "final_report":
            self.final_report_calls += 1
            return AIMessage(content="# 最终研究报告\n\n基于并发研究结果生成的报告。")

        raise AssertionError(f"未知阶段：{stage}")

    @staticmethod
    def _topic(messages):
        """从消息中取出当前研究主题（第一条 HumanMessage）。"""
        for message in messages:
            if isinstance(message, HumanMessage):
                return message.content
        return "unknown"

    def build_main_graph(self):
        """编译带内存 checkpointer 的主图，以支持 interrupt 恢复。"""
        return dr.build_deep_researcher(checkpointer=InMemorySaver())

    def supervisor_state(self, sub_questions=None):
        """构造主管子图的输入状态。"""
        questions = list(SUB_QUESTIONS if sub_questions is None else sub_questions)
        return {
            "supervisor_messages": [
                SystemMessage(content="你是深度研究任务负责人。"),
                HumanMessage(content=DECOMPOSE_BRIEF),
            ],
            "research_brief": DECOMPOSE_BRIEF,
            "sub_questions": questions,
        }


def clear_clarify_response() -> ClarifyWithUser:
    """用户问题已经足够清晰。"""
    return ClarifyWithUser(
        need_clarification=False,
        question="",
        verification="信息已收到，马上开始研究。",
    )


def unclear_clarify_response() -> ClarifyWithUser:
    """用户问题不够清晰，需要澄清。"""
    return ClarifyWithUser(
        need_clarification=True,
        question=CLARIFY_QUESTION,
        verification="",
    )


# 测试 1：用户问题足够清晰时，clarify -> decompose_question -> research_supervisor
def test_clear_question_skips_clarification(monkeypatch):
    harness = Harness(monkeypatch, clear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(SUB_QUESTIONS)
    ))
    result = asyncio.run(harness.build_main_graph().ainvoke(
        {"messages": [HumanMessage(content="请研究 2026 年 AI Agent 框架发展趋势。")]},
        {"configurable": {"thread_id": "clear-question"}},
    ))

    assert "__interrupt__" not in result
    assert ResearchQuestions in harness.fake_model.structured_requests
    assert result["research_brief"] == DECOMPOSE_BRIEF
    assert result["final_report"]


# 测试 2：用户问题不清晰时，进入 interrupt，且不能进入 decompose_question / research_supervisor
def test_unclear_question_interrupts_before_decompose(monkeypatch):
    harness = Harness(monkeypatch, unclear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(SUB_QUESTIONS)
    ))
    result = asyncio.run(harness.build_main_graph().ainvoke(
        {"messages": [HumanMessage(content="帮我研究一下 AI Agent。")]},
        {"configurable": {"thread_id": "unclear-question"}},
    ))

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "clarification"
    assert payload["question"] == CLARIFY_QUESTION
    assert result.get("clarification_question") == CLARIFY_QUESTION
    # 没有进入问题拆解，也没有启动任何研究员
    assert ResearchQuestions not in harness.fake_model.structured_requests
    assert harness.conducted_topics == []
    assert "final_report" not in result


# 测试 3：Command(resume=用户回答) 后，interrupt 恢复并继续 decompose_question -> research_supervisor
def test_resume_after_interrupt_continues_research(monkeypatch):
    harness = Harness(monkeypatch, unclear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(SUB_QUESTIONS)
    ))
    graph = harness.build_main_graph()
    thread_config = {"configurable": {"thread_id": "resume-thread"}}

    async def run():
        first = await graph.ainvoke(
            {"messages": [HumanMessage(content="帮我研究一下 AI Agent。")]},
            thread_config,
        )
        assert "__interrupt__" in first
        resumed = await graph.ainvoke(Command(resume="我主要关注企业级 AI Agent 框架。"), thread_config)
        return first, resumed

    _, resumed = asyncio.run(run())

    assert "__interrupt__" not in resumed
    # 消息历史完整保留：原始问题 -> 澄清问题 -> 用户回答
    message_contents = [m.content for m in resumed["messages"]]
    assert "帮我研究一下 AI Agent。" in message_contents
    assert CLARIFY_QUESTION in message_contents
    assert "我主要关注企业级 AI Agent 框架。" in message_contents
    # 恢复后完成了问题拆解与研究（主管派发了全部子问题）
    assert ResearchQuestions in harness.fake_model.structured_requests
    # 关键：恢复执行不会重复调用澄清判断模型，也不会重复触发 interrupt
    assert harness.fake_model.structured_requests.count(ClarifyWithUser) == 1
    assert harness.fake_model.structured_requests.count(ResearchQuestions) == 1
    assert harness.conducted_topics == SUB_QUESTIONS
    assert resumed["final_report"]


# 测试 4：一个用户问题被拆成多个 sub_questions，并写入 state
def test_decompose_creates_multiple_sub_questions(monkeypatch):
    harness = Harness(monkeypatch, clear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(SUB_QUESTIONS)
    ))
    result = asyncio.run(harness.build_main_graph().ainvoke(
        {"messages": [HumanMessage(content="对比 LangGraph、CrewAI、AutoGen 与 MCP。")]},
        {"configurable": {"thread_id": "decompose-thread"}},
    ))

    assert result["sub_questions"] == SUB_QUESTIONS
    # 主管的系统提示词与初始任务消息中都包含了子问题清单
    supervisor_context = "\n".join(str(m.content) for m in result["supervisor_messages"])
    for question in SUB_QUESTIONS:
        assert question in supervisor_context


# 测试 5：主管根据多个 sub_questions 生成多个 ConductResearch
def test_supervisor_generates_one_conduct_research_per_sub_question(monkeypatch):
    harness = Harness(monkeypatch, clear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(SUB_QUESTIONS)
    ))
    asyncio.run(dr.supervisor_subgraph.ainvoke(
        harness.supervisor_state(),
        {"configurable": {"thread_id": "supervisor-fanout"}},
    ))

    assert harness.conducted_topics == SUB_QUESTIONS


# 测试 6：多个 Researcher 并发执行
def test_multiple_researchers_run_concurrently(monkeypatch):
    harness = Harness(monkeypatch, clear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(SUB_QUESTIONS)
    ))
    asyncio.run(dr.supervisor_subgraph.ainvoke(
        harness.supervisor_state(),
        {"configurable": {"thread_id": "supervisor-concurrency"}},
    ))

    # 4 个子问题的研究员同时处于运行状态，说明是并发而不是串行
    assert harness.max_active_researchers == len(SUB_QUESTIONS)


# 测试 7：研究员完成 compress_research 后，结果通过 ToolMessage 返回主管
def test_researcher_results_flow_back_to_supervisor(monkeypatch):
    harness = Harness(monkeypatch, clear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(SUB_QUESTIONS)
    ))
    result = asyncio.run(dr.supervisor_subgraph.ainvoke(
        harness.supervisor_state(),
        {"configurable": {"thread_id": "supervisor-results"}},
    ))

    # 每个子问题都有一份压缩后的研究结果写入 note
    assert len(result["notes"]) == len(SUB_QUESTIONS)
    for question in SUB_QUESTIONS:
        assert any(f"【{question}】" in str(note) for note in result["notes"])
    # 研究结果以 ToolMessage 的形式写回主管消息，主管模型确实读到了这些结果
    assert harness.supervisor_seen_messages, "主管模型应该被调用"
    received_tool_messages = [
        m for m in harness.supervisor_seen_messages[-1] if isinstance(m, ToolMessage)
    ]
    assert len(received_tool_messages) == len(SUB_QUESTIONS)
    assert all("压缩后的完整研究结果" in str(m.content) for m in received_tool_messages)
    # 原始素材 raw_notes 一并回传
    assert result["raw_notes"]


# 测试 8：主管判断信息不足时，继续调用 ConductResearch
def test_supervisor_requests_more_research_when_information_is_insufficient(monkeypatch):
    follow_up_topic = "补充研究 MCP 生态与实际落地案例"
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(SUB_QUESTIONS)),
        supervisor_responses=[
            AIMessage(
                content="还缺少 MCP 生态的信息，继续补充研究。",
                tool_calls=[{
                    "name": "ConductResearch",
                    "args": {"research_topic": follow_up_topic},
                    "id": "extra-1",
                    "type": "tool_call",
                }],
            ),
        ],
    )
    result = asyncio.run(dr.supervisor_subgraph.ainvoke(
        harness.supervisor_state(),
        {"configurable": {"thread_id": "supervisor-more-research"}},
    ))

    assert harness.conducted_topics == SUB_QUESTIONS + [follow_up_topic]
    assert len(result["notes"]) == len(SUB_QUESTIONS) + 1


# 测试 9：主管判断信息足够时调用 ResearchComplete，进入 final_report_generation
def test_supervisor_completes_research_and_generates_final_report(monkeypatch):
    harness = Harness(monkeypatch, clear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(SUB_QUESTIONS)
    ))
    result = asyncio.run(harness.build_main_graph().ainvoke(
        {"messages": [HumanMessage(content="请研究 2026 年 AI Agent 框架趋势。")]},
        {"configurable": {"thread_id": "final-report"}},
    ))

    assert harness.final_report_calls == 1
    assert result["final_report"].startswith("# 最终研究报告")
    # 最终报告作为消息写回
    assert any("最终研究报告" in str(m.content) for m in result["messages"])


# 额外测试：配置 allow_clarification=False 时直接跳过澄清判断
def test_clarification_can_be_disabled(monkeypatch):
    harness = Harness(monkeypatch, unclear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(SUB_QUESTIONS)
    ))
    result = asyncio.run(harness.build_main_graph().ainvoke(
        {"messages": [HumanMessage(content="帮我研究一下 AI Agent。")]},
        {"configurable": {"thread_id": "no-clarify", "allow_clarification": False}},
    ))

    assert "__interrupt__" not in result
    assert ClarifyWithUser not in harness.fake_model.structured_requests
    assert harness.conducted_topics == SUB_QUESTIONS


# ---------------------------------------------------------------------------- #
# 新增测试：ConductResearch 代码层去重、研究批次上限与 notes 过滤
# ---------------------------------------------------------------------------- #

DEDUP_TOPICS = ["研究 Python 的语言类型与运行机制", "研究 Java 的虚拟机与运行机制"]


def conduct_research_message(*topics, content="继续补充研究。"):
    """构造一条包含若干 ConductResearch 调用的主管响应。"""
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": "ConductResearch",
                "args": {"research_topic": topic},
                "id": f"conduct-{index}-{topic}",
                "type": "tool_call",
            }
            for index, topic in enumerate(topics)
        ],
    )


def think_tool_message(reflection="先反思一下研究进度。"):
    """构造一条只调用 think_tool 的主管响应。"""
    return AIMessage(
        content="先反思。",
        tool_calls=[{
            "name": "think_tool",
            "args": {"reflection": reflection},
            "id": "think-supervisor",
            "type": "tool_call",
        }],
    )


def research_complete_message():
    """构造一条调用 ResearchComplete 的主管响应。"""
    return AIMessage(
        content="信息已经足够。",
        tool_calls=[{
            "name": "ResearchComplete",
            "args": {},
            "id": "complete-1",
            "type": "tool_call",
        }],
    )


def run_supervisor(harness, sub_questions=None, **configurable):
    """执行带内存 checkpointer 的主管子图，返回（输出结果, 子图状态快照）。"""
    graph = dr.supervisor_builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"dedup-{uuid4().hex[:8]}", **configurable}}

    async def _run():
        result = await graph.ainvoke(harness.supervisor_state(sub_questions), config)
        snapshot = await graph.aget_state(config)
        return result, snapshot

    return asyncio.run(_run())


def run_supervisor_recording(harness, sub_questions=None, **configurable):
    """执行主管子图，并记录每次 supervisor_tools 调用时看到的 researched_topics。

    返回（输出结果, 状态快照, 每轮 supervisor_tools 进入时的 researched_topics 历史）。
    用于回归验证历史已研究主题不会在后续回合被覆盖。
    """
    seen_history: list[list[str]] = []

    async def recording_supervisor_tools(state, config):
        seen_history.append(list(state.get("researched_topics", [])))
        return await dr.supervisor_tools(state, config)

    builder = StateGraph(
        SupervisorState,
        output=SupervisorOutputState,
        config_schema=Configuration,
    )
    builder.add_node("supervisor", dr.supervisor)
    builder.add_node("supervisor_tools", recording_supervisor_tools)
    builder.add_edge(START, "supervisor")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"dedup-history-{uuid4().hex[:8]}", **configurable}}

    async def _run():
        result = await graph.ainvoke(harness.supervisor_state(sub_questions), config)
        snapshot = await graph.aget_state(config)
        return result, snapshot

    result, snapshot = asyncio.run(_run())
    return result, snapshot, seen_history


# 新增测试 1：第一轮 Python + Java 正常各执行一次，并记录到 researched_topics
def test_first_round_researches_each_sub_question_once(monkeypatch):
    harness = Harness(monkeypatch, clear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(DEDUP_TOPICS)
    ))
    result, snapshot = run_supervisor(harness, DEDUP_TOPICS)

    assert harness.researcher_run_topics == DEDUP_TOPICS
    assert all(harness.researcher_runs[topic] == 1 for topic in DEDUP_TOPICS)
    assert snapshot.values.get("researched_topics") == DEDUP_TOPICS
    assert snapshot.values.get("research_batches") == 1
    assert len(result["notes"]) == 2


# 新增测试 2：第二轮 LLM 再次请求 Python + Java 时，不得再次执行
def test_second_round_duplicate_topics_are_not_researched(monkeypatch):
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(DEDUP_TOPICS)),
        supervisor_responses=[conduct_research_message(*DEDUP_TOPICS)],
    )
    result, snapshot = run_supervisor(harness, DEDUP_TOPICS)

    assert harness.researcher_run_topics == DEDUP_TOPICS
    assert all(harness.researcher_runs[topic] == 1 for topic in DEDUP_TOPICS)
    # 全部重复 → 提前结束，只调用 1 次主管 LLM，批次仍是 1
    assert harness.supervisor_calls == 1
    assert snapshot.values.get("research_batches") == 1
    assert len(result["notes"]) == 2

    # 被跳过的调用仍然返回了一一配对的 ToolMessage
    skipped = [
        message
        for message in snapshot.values.get("supervisor_messages", [])
        if isinstance(message, ToolMessage) and "本次跳过" in str(message.content)
    ]
    assert len(skipped) == 2
    assert all(message.tool_call_id for message in skipped)


# 新增测试 3：第二轮请求 Python + 新主题，Python 被跳过，新主题正常执行
def test_duplicate_topic_skipped_but_new_topic_runs(monkeypatch):
    new_topic = "研究 Python 的 JIT 编译与 PyPy"
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(DEDUP_TOPICS)),
        supervisor_responses=[conduct_research_message(DEDUP_TOPICS[0], new_topic)],
    )
    result, snapshot = run_supervisor(harness, DEDUP_TOPICS)

    assert harness.researcher_runs[DEDUP_TOPICS[0]] == 1    # 重复主题被跳过
    assert harness.researcher_runs[new_topic] == 1          # 不同但相关的新主题正常执行
    assert snapshot.values.get("research_batches") == 2
    assert new_topic in snapshot.values.get("researched_topics", [])
    assert len(result["notes"]) == 3


# 新增测试 4：同一轮出现 Python + Python + Java，Python 只执行一次，Java 执行一次
def test_same_batch_duplicate_topics_execute_only_once(monkeypatch):
    seed_topics = ["研究 Java 的虚拟机与运行机制"]
    duplicate = "研究 Python 的语言类型与运行机制"
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(seed_topics)),
        supervisor_responses=[conduct_research_message(duplicate, duplicate, seed_topics[0])],
    )
    result, snapshot = run_supervisor(harness, seed_topics)

    assert harness.researcher_runs[duplicate] == 1         # 同批次重复只执行一次
    assert harness.researcher_runs[seed_topics[0]] == 1    # 已研究过的主题被跳过
    assert harness.researcher_run_topics == seed_topics + [duplicate]
    assert len(result["notes"]) == 2


# 新增测试 5：researcher 执行失败时，不把该主题加入 researched_topics，且允许后续重试
def test_failed_research_is_not_marked_and_can_retry(monkeypatch):
    failing_topic = DEDUP_TOPICS[1]
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(DEDUP_TOPICS)),
        supervisor_responses=[conduct_research_message(failing_topic)],
        fail_topics={failing_topic},
    )
    result, snapshot = run_supervisor(harness, DEDUP_TOPICS)

    researched = snapshot.values.get("researched_topics", [])
    assert failing_topic not in researched                     # 失败主题不标记为已研究
    assert DEDUP_TOPICS[0] in researched                       # 成功主题正常记录
    assert harness.researcher_runs[failing_topic] == 2         # 失败后允许再次研究
    # 失败结果不进入 notes，只有成功主题的结果进入
    assert len(result["notes"]) == 1
    assert DEDUP_TOPICS[0] in str(result["notes"][0])


# 新增测试 6：连续重复研究时，Supervisor 应提前结束，而不是跑到最大 supervisor iteration
def test_repeated_duplicates_end_supervisor_early(monkeypatch):
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(DEDUP_TOPICS)),
        supervisor_responses=[
            conduct_research_message(DEDUP_TOPICS[0]),
            research_complete_message(),
        ],
    )
    result, snapshot = run_supervisor(harness, DEDUP_TOPICS)

    assert harness.supervisor_calls == 1                 # 全重复 → 直接结束，没有下一轮 LLM
    assert len(harness.supervisor_responses) == 1        # 第二个响应没有被消费
    assert snapshot.values.get("research_batches") == 1
    assert len(result["notes"]) == 2


# 新增测试 7：max_researcher_iterations 真正限制“研究批次”
def test_max_researcher_iterations_limits_research_batches(monkeypatch):
    follow_up_topics = ["补充研究 A", "补充研究 B"]
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(DEDUP_TOPICS)),
        supervisor_responses=[
            conduct_research_message(follow_up_topics[0]),
            conduct_research_message(follow_up_topics[1]),
        ],
    )
    result, snapshot = run_supervisor(harness, DEDUP_TOPICS, max_researcher_iterations=2)

    # 种子批次算第 1 批，补充 A 是第 2 批，达到上限后补充 B 不再执行
    assert harness.conducted_topics == DEDUP_TOPICS + [follow_up_topics[0]]
    assert follow_up_topics[1] not in harness.researcher_calls
    assert snapshot.values.get("research_batches") == 2
    assert len(result["notes"]) == 3


# 新增测试 8：ResearchComplete 原有行为不被破坏
def test_research_complete_still_ends_supervisor(monkeypatch):
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(DEDUP_TOPICS)),
        supervisor_responses=[research_complete_message()],
    )
    result, snapshot = run_supervisor(harness, DEDUP_TOPICS)

    assert harness.supervisor_calls == 1
    assert harness.conducted_topics == DEDUP_TOPICS
    assert snapshot.values.get("research_batches") == 1
    assert len(result["notes"]) == 2


# 新增测试 9：think_tool 原有流程不被破坏，且反思不进入 notes
def test_think_tool_flow_still_works_and_is_excluded_from_notes(monkeypatch):
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(DEDUP_TOPICS)),
        supervisor_responses=[think_tool_message("检查子问题覆盖情况")],
    )
    result, snapshot = run_supervisor(harness, DEDUP_TOPICS)

    # think 之后回到 supervisor，再走默认的 ResearchComplete
    assert harness.supervisor_calls == 2
    seen_messages = harness.supervisor_seen_messages[-1]
    assert any(
        isinstance(message, ToolMessage) and message.name == "think_tool"
        for message in seen_messages
    )
    assert snapshot.values.get("research_batches") == 1
    assert all("反思已记录" not in str(note) for note in result["notes"])
    assert len(result["notes"]) == 2


# 新增测试 10：最终 notes 不包含 think_tool、重复跳过提示与批次上限错误
def test_notes_exclude_think_and_duplicate_results(monkeypatch):
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(DEDUP_TOPICS)),
        supervisor_responses=[
            think_tool_message("先反思"),
            conduct_research_message(*DEDUP_TOPICS),   # 全部重复 → 结束
        ],
    )
    result, _ = run_supervisor(harness, DEDUP_TOPICS)

    notes = [str(note) for note in result["notes"]]
    assert len(notes) == 2
    assert all("反思已记录" not in note for note in notes)
    assert all("本次跳过" not in note for note in notes)
    assert all("错误：已达到最大研究批次上限" not in note for note in notes)
    # 每个研究结果都带主题标记，且同一主题只出现一次
    assert all("【研究主题】" in note for note in notes)
    topics_in_notes = [note.split("【研究结果】")[0] for note in notes]
    assert len(set(topics_in_notes)) == 2


# 新增测试 11：max_researcher_iterations 不应缩小单个批次的并发容量
def test_batch_limit_does_not_shrink_single_batch_capacity(monkeypatch):
    topics = [f"研究主题 {index}" for index in range(4)]
    harness = Harness(monkeypatch, clear_clarify_response(), ResearchQuestions(
        research_brief=DECOMPOSE_BRIEF, sub_questions=list(topics)
    ))
    result, snapshot = run_supervisor(harness, topics, max_researcher_iterations=2)

    # 上限是 2 个批次、每批最多 5 个主题：种子批次应当把 4 个主题一次跑完
    assert harness.researcher_run_topics == topics
    assert snapshot.values.get("research_batches") == 1
    assert len(result["notes"]) == 4


# 新增测试 12：超出单批并发上限的新主题，允许在下一轮重新派发并正常执行
def test_concurrency_overflow_topic_can_be_dispatched_next_round(monkeypatch):
    topics = [f"研究主题 {index}" for index in range(6)]
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(topics)),
        supervisor_responses=[conduct_research_message(topics[5])],
    )
    result, snapshot = run_supervisor(harness, topics)   # 默认 max_concurrent_research_units=5

    # 前 5 个在种子批次执行，第 6 个因并发上限被推迟，下一轮重新派发后正常执行
    assert set(harness.researcher_run_topics) == set(topics)
    assert all(harness.researcher_runs[topic] == 1 for topic in topics)
    assert snapshot.values.get("research_batches") == 2
    assert len(result["notes"]) == 6


# 新增测试 13：主题规范化覆盖首尾空白、连续空白与大小写
def test_topic_normalization_detects_whitespace_and_case_duplicates(monkeypatch):
    topic = "Python 的语言类型与运行机制"
    normalized_variant = "   PYTHON   的语言类型与运行机制   "
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=[topic]),
        supervisor_responses=[conduct_research_message(normalized_variant)],
    )
    result, snapshot = run_supervisor(harness, [topic])

    # 规范化后与已研究主题相同 → 不再执行 researcher
    assert harness.researcher_run_topics == [topic]
    assert harness.researcher_runs[topic] == 1
    assert snapshot.values.get("research_batches") == 1
    assert len(result["notes"]) == 1


# 新增测试 14：Supervisor 连续多轮都返回 A/B/C，也不能重复研究
def test_supervisor_never_researches_abc_across_rounds(monkeypatch):
    topics = ["A", "B", "C"]
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(topics)),
        supervisor_responses=[
            conduct_research_message(*topics),
            conduct_research_message(*topics),
        ],
    )
    result, snapshot, seen_history = run_supervisor_recording(harness, topics)

    # 第 1 次 supervisor_tools 之前没有历史；第 2 次已带上完整的 A/B/C，没有被覆盖
    assert seen_history[0] == []
    assert set(seen_history[1]) == set(topics)
    assert harness.researcher_runs == {"A": 1, "B": 1, "C": 1}
    assert set(snapshot.values.get("researched_topics", [])) == set(topics)
    # 第二轮全是重复主题 → 提前结束，不再调用主管 LLM
    assert harness.supervisor_calls == 1
    assert snapshot.values.get("research_batches") == 1
    assert len(result["notes"]) == 3


# 新增测试 15：跨多个 supervisor_tools 回合，历史 researched_topics 不丢失
def test_researched_topics_accumulates_across_supervisor_tools_rounds(monkeypatch):
    seed_topics = ["A", "B"]
    harness = Harness(
        monkeypatch,
        clear_clarify_response(),
        ResearchQuestions(research_brief=DECOMPOSE_BRIEF, sub_questions=list(seed_topics)),
        supervisor_responses=[
            conduct_research_message("A", "C"),   # A 已研究过被跳过，C 是新增主题
            conduct_research_message("A", "B"),   # A/B 都在历史里 → 全重复，结束
        ],
    )
    result, snapshot, seen_history = run_supervisor_recording(harness, seed_topics)

    assert seen_history[0] == []
    assert set(seen_history[1]) == {"A", "B"}
    # 第 3 轮仍然能看到累计历史 A/B/C，说明第 2 轮的新增没有把历史覆盖成只有 C
    assert set(seen_history[2]) == {"A", "B", "C"}
    assert harness.researcher_runs == {"A": 1, "B": 1, "C": 1}
    assert set(snapshot.values.get("researched_topics", [])) == {"A", "B", "C"}
    assert snapshot.values.get("research_batches") == 2
    assert len(result["notes"]) == 3
