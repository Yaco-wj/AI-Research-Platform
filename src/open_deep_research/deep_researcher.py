import asyncio
from typing import Literal
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    decompose_question_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestions,
    SupervisorOutputState,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# 初始化可配置模型，供整个 Agent 复用
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[
    Literal["wait_for_user_clarification", "decompose_question"]
]:
    """判断用户需求是否需要澄清。

    该节点只负责“判断”：调用模型做结构化分析，然后分流：
    - 不需要澄清：直接进入 decompose_question，并写入确认话术。
    - 需要澄清：写入澄清问题，进入 wait_for_user_clarification，
      由该节点通过 langgraph 的 interrupt 暂停图执行并等待用户回答。

    Args:
        state: 包含用户消息的主图状态
        config: 运行时配置（模型、API Key、是否允许澄清等）

    Returns:
        跳转指令：要么进入等待用户澄清的节点，要么直接进入问题拆解节点
    """
    # 第 1 步：检查配置是否允许发起澄清提问
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # 配置为不允许澄清时，直接进入问题拆解
        return Command(goto="decompose_question")

    # 第 2 步：准备用于澄清判断的结构化输出模型
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # 第 3 步：分析是否需要澄清
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # 第 4 步：根据分析结果分流
    if response.need_clarification:
        # 需要澄清：记录澄清问题，转由 wait_for_user_clarification 节点执行 interrupt 暂停
        # 注意：这里不结束整个图，而是把问题写入 state 后继续流转
        return Command(
            goto="wait_for_user_clarification",
            update={
                "messages": [AIMessage(content=response.question)],
                "clarification_question": response.question,
            }
        )
    else:
        # 不需要澄清：写入确认话术，直接进入问题拆解
        return Command(
            goto="decompose_question",
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def wait_for_user_clarification(state: AgentState, config: RunnableConfig) -> Command[
    Literal["decompose_question"]
]:
    """通过 LangGraph interrupt 暂停图执行，等待用户回答澄清问题。

    该节点是“等待用户回答”的独立节点：节点内部在调用 interrupt 之前不做任何
    模型调用，因此 interrupt 恢复、节点重新执行时不会重复触发澄清判断。

    执行流程：
    1. 第一次执行：interrupt(payload) 抛出中断，图停止，payload 返回给调用方并保存 checkpoint；
    2. 用户使用 Command(resume=用户回答) 恢复同一 thread_id；
    3. 节点重新执行，interrupt(...) 直接返回用户回答；
    4. 把用户回答写入 messages，跳转到 decompose_question。

    Args:
        state: 包含澄清问题的主图状态
        config: 运行时配置

    Returns:
        跳转指令：把用户回答写入消息后进入问题拆解节点
    """
    # interrupt 的返回值在“恢复执行”时就是调用方传入的 resume 值
    answer = interrupt({
        "type": "clarification",
        "question": state.get("clarification_question", ""),
        "context": "用户需求还不够明确，需要用户补充信息后再拆解研究问题",
    })

    # 兼容不同形式的 resume 传参（字符串或带 answer 字段的字典）
    if isinstance(answer, dict):
        answer_text = str(answer.get("answer") or answer.get("content") or answer)
    else:
        answer_text = str(answer)

    # 把用户回答写入消息历史，随后进入问题拆解
    return Command(
        goto="decompose_question",
        update={"messages": [HumanMessage(content=answer_text)]}
    )


async def decompose_question(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """把用户需求拆解成多个可独立研究的子问题，并初始化 Supervisor。

    该节点只负责需求理解与拆解，不执行搜索、不调用研究员：
    1. 读取用户最终确认后的完整消息；
    2. 生成整体研究简报 research_brief；
    3. 生成结构化研究子问题列表 sub_questions；
    4. 初始化 supervisor_messages（系统提示词 + 研究任务说明）；
    5. 跳转到 research_supervisor。

    Args:
        state: 包含完整用户消息的主图状态
        config: 运行时配置

    Returns:
        跳转指令：带着 research_brief / sub_questions / supervisor_messages 进入研究主管
    """
    # 第 1 步：准备用于问题拆解的结构化输出模型
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    decompose_model = (
        configurable_model
        .with_structured_output(ResearchQuestions)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 第 2 步：基于完整消息生成研究简报与研究子问题
    prompt_content = decompose_question_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await decompose_model.ainvoke([HumanMessage(content=prompt_content)])

    # 第 3 步：清洗子问题列表（去空白、去空字符串）
    sub_questions = [
        question.strip()
        for question in (response.sub_questions or [])
        if question and question.strip()
    ]

    sub_questions_text = "\n".join(
        f"{index + 1}. {question}" for index, question in enumerate(sub_questions)
    ) or "（本轮没有拆解出额外的研究子问题，请根据研究简报自行判断需要研究的方向。）"

    # 第 4 步：初始化 Supervisor 上下文（系统提示词 + 研究任务说明）
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        sub_questions=sub_questions_text,
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )

    initial_supervisor_message = (
        "【研究简报】\n"
        f"{response.research_brief}\n\n"
        "【研究子问题清单】\n"
        f"{sub_questions_text}\n\n"
        "请围绕上述研究简报统筹研究进度：每个子问题由一个独立的研究任务负责。"
        "系统已经针对子问题清单并发派发了第一批研究任务；"
        "收到研究结果后，请判断信息是否充分：如果不足，继续调用 ConductResearch 补充研究；"
        "如果已经足以支撑最终报告，调用 ResearchComplete 结束研究阶段。"
    )

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "sub_questions": sub_questions,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=initial_supervisor_message)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """研究主管节点：制定研究策略并把研究任务委派给研究员。

    第一轮：直接读取 state["sub_questions"]，为每个子问题批量生成一个
    ConductResearch 工具调用，保证多个研究员可以并发启动。

    后续轮次：调用大模型，让主管根据已有研究结果决定继续派发
    ConductResearch（信息不足）还是调用 ResearchComplete（信息足够）。

    Args:
        state: 主管子图状态（消息、研究简报、子问题、迭代轮次）
        config: 运行时配置

    Returns:
        跳转指令：携带主管的决策消息进入 supervisor_tools
    """
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 可用的主管工具：研究委派、研究完成、策略思考
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    supervisor_messages = state.get("supervisor_messages", [])
    sub_questions = state.get("sub_questions", [])
    research_iterations = state.get("research_iterations", 0)

    # 第 1 步：第一轮直接按子问题批量生成 ConductResearch 调用（并发派发）
    if research_iterations == 0 and sub_questions:
        tool_calls = [
            {
                "name": "ConductResearch",
                "args": {"research_topic": question},
                "id": f"conduct-{index}-{uuid4().hex[:8]}",
                "type": "tool_call",
            }
            for index, question in enumerate(sub_questions)
        ]
        seed_message = AIMessage(
            content=f"已根据问题拆解结果，批量派发 {len(tool_calls)} 个并发研究任务。",
            tool_calls=tool_calls,
        )
        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [seed_message],
                "research_iterations": research_iterations + 1
            }
        )

    # 第 2 步：后续轮次由大模型决策（继续研究 / 结束研究）
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    response = await research_model.ainvoke(supervisor_messages)

    # 第 3 步：更新消息与迭代计数，进入工具执行节点
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": research_iterations + 1
        }
    )


# 研究结果的三个标记：用于拼接 ToolMessage 内容、解析主题与过滤 notes
RESEARCH_TOPIC_MARKER = "【研究主题】"
RESEARCH_RESULT_MARKER = "【研究结果】"
RESEARCH_FAILURE_MARKER = "【研究失败】"


def normalize_research_topic(topic: str) -> str:
    """规范化研究主题，用于代码层去重。

    只做字面规范化：去除首尾空白、压缩连续空白、统一大小写。
    刻意不做 embedding / 语义相似度判断，保证“不同但相关”的补充研究
    仍然会被当成新主题正常执行。
    """
    return " ".join(str(topic).split()).casefold()


def format_research_result(topic: str, result: str) -> str:
    """把研究结果包装成带主题的 ToolMessage 内容，便于主管对应子问题。"""
    return f"{RESEARCH_TOPIC_MARKER}\n{topic}\n\n{RESEARCH_RESULT_MARKER}\n{result}"


def format_research_failure(topic: str, error: str) -> str:
    """把研究失败包装成带主题的 ToolMessage 内容（不带【研究结果】标记）。"""
    return f"{RESEARCH_TOPIC_MARKER}\n{topic}\n\n{RESEARCH_FAILURE_MARKER}\n{error}"


def is_research_result_success(observation: dict) -> bool:
    """判断一次研究是否成功：结果非空且不是压缩阶段的降级错误。"""
    compressed = str(observation.get("compressed_research") or "")
    if not compressed:
        return False
    return not compressed.startswith("错误：")


def extract_researched_topic(content: str) -> str | None:
    """从研究结果 ToolMessage 中解析出研究主题（用于 notes 去重）。"""
    text = str(content)
    if RESEARCH_TOPIC_MARKER not in text or RESEARCH_RESULT_MARKER not in text:
        return None
    start = text.index(RESEARCH_TOPIC_MARKER) + len(RESEARCH_TOPIC_MARKER)
    end = text.index(RESEARCH_RESULT_MARKER)
    topic = text[start:end].strip()
    return topic or None


def collect_research_notes(messages: list) -> list[str]:
    """收集真正由 ConductResearch 成功产生的笔记。

    规则：
    - 只保留带【研究结果】标记的 ToolMessage；
    - 排除 think_tool 反思、重复主题跳过提示、并发/批次上限等错误提示；
    - 按规范化研究主题去重，避免同一主题的重复结果进入最终报告。
    """
    notes: list[str] = []
    seen_topics: set[str] = set()
    for message in filter_messages(messages, include_types=["tool"]):
        content = str(message.content)
        if RESEARCH_RESULT_MARKER not in content:
            continue
        topic = extract_researched_topic(content)
        dedup_key = normalize_research_topic(topic) if topic else content
        if dedup_key in seen_topics:
            continue
        seen_topics.add(dedup_key)
        notes.append(content)
    return notes


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """执行主管调用的工具：think_tool 反思、ConductResearch 并发研究、ResearchComplete 结束。

    该节点负责：
    1. 判断是否满足退出条件（节点调用兜底超限 / 没有工具调用 / 调用 ResearchComplete）；
    2. 处理 think_tool，把主管的思考写入消息上下文；
    3. 对 ConductResearch 做代码层去重：已研究过的主题不再启动研究员，
       同批次内重复的主题只执行一次，被跳过的调用仍然返回配对的 ToolMessage；
    4. 并发执行去重后的新主题（每个对应一个 researcher 子图），结果带主题写回；
    5. 聚合 raw_notes，并在退出时只收集真正成功的研究结果作为 notes。

    Args:
        state: 主管子图状态
        config: 运行时配置（并发上限、批次上限等）

    Returns:
        跳转指令：回到 supervisor 继续决策，或结束主管子图
    """
    # 第 1 步：读取当前状态并判断退出条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    research_batches = state.get("research_batches", 0)
    researched_topics = state.get("researched_topics", [])
    most_recent_message = supervisor_messages[-1]

    # 三个终止条件（满足任意一个即退出主管子图）：
    # 1. supervisor 节点调用次数超过兜底安全上限，防止纯 think_tool 死循环；
    # 2. 模型没有输出任何 tool_call，没有任务可执行；
    # 3. 模型调用了 ResearchComplete，表示主管判断信息已经足够。
    # 注意：真正的“研究批次上限”由 research_batches 控制，不再用节点调用次数当批次。
    max_supervisor_steps = configurable.max_researcher_iterations * 3 + 3
    exceeded_allowed_iterations = research_iterations > max_supervisor_steps
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": collect_research_notes(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

    # 第 2 步：处理本轮所有工具调用（think_tool 与 ConductResearch）
    all_tool_messages: list[ToolMessage] = []
    update_payload: dict = {"supervisor_messages": []}

    # 处理 think_tool：只把主管的反思记录写入消息上下文，没有外部执行动作
    for tool_call in most_recent_message.tool_calls:
        if tool_call["name"] != "think_tool":
            continue
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"反思已记录：{reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # 提取本轮的 ConductResearch 调用
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        # 第 3 步：代码层去重。
        # 已经研究过（researched_topics）与同一批次内重复的主题都只保留第一次，
        # 其余调用登记跳过原因，稍后仍然返回一一配对的 ToolMessage。
        researched_keys = {normalize_research_topic(topic) for topic in researched_topics}
        unique_new_topics: list[tuple[str, str]] = []   # [(规范化 key, 原始主题)]
        primary_tool_call: dict[str, dict] = {}
        skip_reasons: dict[str, str] = {}

        for tool_call in conduct_research_calls:
            raw_topic = str(tool_call["args"]["research_topic"])
            topic_key = normalize_research_topic(raw_topic)
            if topic_key in researched_keys:
                skip_reasons[tool_call["id"]] = (
                    f"该主题此前已经研究完成，本次跳过，不重复启动研究员：{raw_topic}"
                )
                continue
            if topic_key in primary_tool_call:
                skip_reasons[tool_call["id"]] = (
                    f"该主题在同一批次中重复请求，已由前一个调用执行，本次跳过：{raw_topic}"
                )
                continue
            primary_tool_call[topic_key] = tool_call
            unique_new_topics.append((topic_key, raw_topic))

        # 第 4 步：批次上限保护。
        # max_researcher_iterations 限制“实际执行过的 ConductResearch 批次”，
        # max_concurrent_research_units 只限制单批并发数量。
        remaining_batches = configurable.max_researcher_iterations - research_batches
        # 还剩至少 1 个批次时，本轮可以跑满一个并发批次；
        # max_concurrent_research_units 只限制单批数量，不能缩小本轮批次容量。
        if remaining_batches >= 1:
            max_new_topics = configurable.max_concurrent_research_units
            overflow_reason = (
                "错误：本轮没有执行该研究任务，因为已经超出最大并发研究单元数量"
                f"（上限 {configurable.max_concurrent_research_units} 个）。"
                "该主题仍是未研究的新主题，可在下一轮重新派发："
            )
        else:
            max_new_topics = 0
            overflow_reason = (
                f"错误：已达到最大研究批次上限（上限 {configurable.max_researcher_iterations} 批），"
                "本次不再启动新的研究员："
            )
        executable_topics = unique_new_topics[:max_new_topics]
        blocked_topics = unique_new_topics[max_new_topics:]

        for topic_key, raw_topic in blocked_topics:
            skip_reasons[primary_tool_call[topic_key]["id"]] = (
                overflow_reason + raw_topic
            )

        result_content_by_key: dict[str, str] = {}
        succeeded_topics: list[str] = []

        # 第 5 步：只对去重后的新主题启动 researcher 子图
        if executable_topics:
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [HumanMessage(content=raw_topic)],
                    "research_topic": raw_topic
                }, config)
                for _, raw_topic in executable_topics
            ]

            # 单个研究员失败不应该让整批失败；token 超限仍沿用原有“结束研究”策略
            tool_results = await asyncio.gather(*research_tasks, return_exceptions=True)

            for (topic_key, raw_topic), observation in zip(executable_topics, tool_results):
                if isinstance(observation, asyncio.CancelledError):
                    raise observation
                if isinstance(observation, BaseException):
                    if is_token_limit_exceeded(observation, configurable.research_model):
                        return Command(
                            goto=END,
                            update={
                                "notes": collect_research_notes(supervisor_messages),
                                "research_brief": state.get("research_brief", "")
                            }
                        )
                    result_content_by_key[topic_key] = format_research_failure(
                        raw_topic, f"研究执行出错：{observation}"
                    )
                    continue

                compressed = str(observation.get("compressed_research") or "")
                if is_research_result_success(observation):
                    result_content_by_key[topic_key] = format_research_result(raw_topic, compressed)
                    succeeded_topics.append(raw_topic)
                else:
                    # 失败的研究不写入 researched_topics，允许后续继续补充研究
                    result_content_by_key[topic_key] = format_research_failure(
                        raw_topic, compressed or "研究结果为空"
                    )

            # 聚合本轮所有研究员的原始素材
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
                if not isinstance(observation, BaseException)
            ])
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

            # research_batches 记录“实际执行过的批次”，失败批次同样计数
            update_payload["research_batches"] = research_batches + 1

        # 第 6 步：按原始 tool_call 顺序生成配对的 ToolMessage，
        # 保证每个 tool_call_id 都有且只有一个结果，不会因为去重而丢配对。
        for tool_call in conduct_research_calls:
            tool_call_id = tool_call["id"]
            topic_key = normalize_research_topic(tool_call["args"]["research_topic"])
            if tool_call_id in skip_reasons:
                content = skip_reasons[tool_call_id]
            elif topic_key in result_content_by_key:
                content = result_content_by_key[topic_key]
            else:
                content = format_research_failure(
                    str(tool_call["args"]["research_topic"]), "研究任务未被执行"
                )
            all_tool_messages.append(ToolMessage(
                content=content, name="ConductResearch", tool_call_id=tool_call_id
            ))

        # 第 7 步：本轮明确存在 ConductResearch，但去重后没有任何新主题
        #（全部是重复主题），或者已经达到批次上限 → 不再空转，直接进入最终报告。
        if not executable_topics:
            update_payload["supervisor_messages"] = all_tool_messages
            return Command(
                goto=END,
                update={
                    **update_payload,
                    "notes": collect_research_notes(supervisor_messages + all_tool_messages),
                    "research_brief": state.get("research_brief", "")
                }
            )

        # 只有成功执行的主题才记入 researched_topics
        if succeeded_topics:
            update_payload["researched_topics"] = succeeded_topics

    # 第 8 步：把本轮所有工具结果写回消息，回到 supervisor 继续决策
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    )


# 主管子图构建：负责研究任务的调度、并发执行与完成判断
# output 指定子图对外输出的字段，避免把子图内部消息重复写回主图
supervisor_builder = StateGraph(
    SupervisorState,
    output=SupervisorOutputState,
    config_schema=Configuration
)

# 添加主管节点：supervisor（决策层）与 supervisor_tools（执行层）
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

# 主管子图入口
supervisor_builder.add_edge(START, "supervisor")

# 编译主管子图，供主图作为节点使用
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """单个研究员节点：围绕指定研究主题决定调用哪些工具。

    研究员由主管分配一个具体研究主题，使用可用工具（搜索、MCP 工具、
    think_tool）收集信息，并在搜索之间进行策略性反思。

    Args:
        state: 研究员子图状态（消息、研究主题）
        config: 运行时配置

    Returns:
        跳转指令：携带模型输出（可能包含 tool_calls）进入 researcher_tools
    """
    # 第 1 步：加载配置并校验工具可用性
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # 获取全部研究工具（搜索、MCP、think_tool）
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )

    # 第 2 步：为研究员模型绑定工具
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str()
    )

    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 第 3 步：生成研究员的响应（系统提示词 + 研究对话历史）
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # 第 4 步：更新状态并进入工具执行节点
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )


async def execute_tool_safely(tool, args, config):
    """安全执行工具：捕获异常并返回错误文本。"""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"工具执行出错：{str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """执行研究员调用的工具：搜索工具、MCP 工具与策略反思。

    该节点负责：
    1. think_tool：记录研究员的反思；
    2. 搜索工具（tavily_search / web_search）与 MCP 工具：并发执行并收集结果；
    3. ResearchComplete：表示单个研究任务的研究已经完成。

    Args:
        state: 研究员子图状态
        config: 运行时配置

    Returns:
        跳转指令：继续研究循环，或进入 compress_research 压缩研究结果
    """
    # 第 1 步：读取状态并做提前退出判断
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # 是否调用了普通工具；是否调用了模型原生的内置网页搜索
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or
        anthropic_websearch_called(most_recent_message)
    )

    # 既没有工具调用也没有原生搜索：说明研究员认为不需要继续搜索，直接进入压缩阶段
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    # 第 2 步：处理其他工具调用（搜索、MCP 工具等）
    tools = await get_all_tools(config)

    # 构建“工具名 -> 工具实例”的映射，便于按 tool_call 的名称找到工具
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    # 并发执行本次所有 tool call（execute_tool_safely 负责捕获异常）
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # 把工具执行结果包装成 ToolMessage，与 tool_call 一一绑定 tool_call_id
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # 第 3 步：处理工具后的退出条件
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # 两个终止条件（满足任一即进入压缩阶段）：
    # 1. 工具调用轮次达到 max_react_tool_calls；
    # 2. 研究员调用了 ResearchComplete，表示该子课题信息已经收集充分。
    if exceeded_iterations or research_complete_called:
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    # 继续研究循环
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """把研究员收集到的研究结果压缩成简洁、结构化的摘要。

    该节点会整理研究员全部消息（工具输出与 AI 消息），提炼出完整的
    compressed_research，同时保留 raw_notes 原始素材，供主管与最终报告使用。

    Args:
        state: 累积了全部研究消息的研究员状态
        config: 运行时配置

    Returns:
        包含 compressed_research 与 raw_notes 的字典
    """
    # 第 1 步：配置压缩模型
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })

    # 第 2 步：准备压缩输入
    researcher_messages = state.get("researcher_messages", [])

    # 追加一条 HumanMessage，指示模型停止搜索、切换到整理压缩模式
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

    # 第 3 步：带重试的压缩流程（处理 token 超限）
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # 构造压缩提示词：系统提示词 + 研究员全部对话历史
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages

            response = await synthesizer_model.ainvoke(messages)

            # 提取原始素材：工具返回内容 + AI 消息
            raw_notes_content = "\n".join([
                str(message.content)
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])

            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }

        except Exception as e:
            # token 超限时删除较早的消息后重试，最多重试 3 次
            synthesis_attempts += 1

            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            continue

    # 第 4 步：全部重试失败时的降级结果，raw_notes 仍然保留原始素材
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "compressed_research": "错误：研究结果整理失败，已达到最大重试次数。",
        "raw_notes": [raw_notes_content]
    }


# 研究员子图构建：针对单个研究主题执行“研究-工具调用-压缩”闭环
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration
)

# 添加研究员节点：researcher（决策）、researcher_tools（工具执行）、compress_research（压缩）
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)

# 研究员子图入口与出口
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)

# 编译研究员子图，供主管并发调用
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """基于全部研究结果生成最终研究报告（带 token 超限重试）。

    Args:
        state: 包含研究笔记与研究简报的主图状态
        config: 运行时配置

    Returns:
        包含最终报告与清理后状态的字典
    """
    # 第 1 步：提取研究笔记并准备状态清理
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    # 第 2 步：配置最终报告生成模型
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 第 3 步：带 token 超限重试的报告生成
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            # 组装最终报告提示词：研究简报 + 全部消息 + 研究结果 + 当前日期
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )

            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])

            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }

        except Exception as e:
            # token 超限时逐步截断 findings 后重试（每次减少 10%）
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if current_retry == 1:
                    # 第一次重试：按模型上下文上限估算允许的字符数（约 1 token ≈ 4 字符）
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    # 后续重试：每次把允许长度减少 10%
                    findings_token_limit = int(findings_token_limit * 0.9)

                findings = findings[:findings_token_limit]
                continue
            else:
                # 非 token 超限错误：直接返回错误信息
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }

    # 第 4 步：重试次数耗尽
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }


# 主图构建：从用户输入到最终报告的完整流程
deep_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration
)

# 主图节点：澄清判断、等待用户澄清（interrupt）、问题拆解、研究主管、最终报告
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("wait_for_user_clarification", wait_for_user_clarification)
deep_researcher_builder.add_node("decompose_question", decompose_question)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# 主图固定边：入口 -> 澄清判断；研究主管 -> 最终报告 -> 结束
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)


def build_deep_researcher(checkpointer=None):
    """按需编译主图。

    传入 checkpointer（例如 InMemorySaver）后，图在执行 interrupt 时
    能够保存 checkpoint，并通过同一个 thread_id + Command(resume=...) 恢复执行。
    """
    return deep_researcher_builder.compile(checkpointer=checkpointer)


# 默认编译主图（部署时由 LangGraph 平台注入持久化 checkpointer）
deep_researcher = deep_researcher_builder.compile()
