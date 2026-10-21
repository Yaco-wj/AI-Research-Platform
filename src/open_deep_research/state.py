"""Deep Research Agent 的图状态定义与数据结构。"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 结构化输出模型
###################
class ConductResearch(BaseModel):
    """调用该工具，让子研究员针对某一个具体主题开展研究。"""

    research_topic: str = Field(
        description="需要研究的主题。必须是单个主题，并且要用至少一段话详细描述研究范围与重点。",
    )


class ResearchComplete(BaseModel):
    """调用该工具，表示研究阶段已经完成。"""


class Summary(BaseModel):
    """网页内容的摘要与关键摘录。"""

    summary: str
    key_excerpts: str


class ClarifyWithUser(BaseModel):
    """判断是否需要向用户发起澄清提问的结构化输出。"""

    need_clarification: bool = Field(
        description="是否需要向用户提出澄清问题。",
    )
    question: str = Field(
        description="需要向用户提出的澄清问题，用于确认报告范围与调研范围。",
    )
    verification: str = Field(
        description="不需要澄清时的确认话术：告知用户信息已经收到，即将开始研究。",
    )


class ResearchQuestions(BaseModel):
    """研究问题拆解结果：整体研究简报 + 若干可独立研究的子问题。"""

    research_brief: str = Field(
        description="对用户研究需求的整体概述，需说明研究目标、范围、受众与预期产出。",
    )
    sub_questions: list[str] = Field(
        description="拆分后的研究子问题列表，每个子问题都要能被一个研究员独立研究，且子问题之间尽量不重叠。",
    )


###################
# 状态定义
###################

def override_reducer(current_value, new_value):
    """状态 reducer：默认按追加合并，特殊情况下（type=override）整体覆盖。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """主图输入状态，只包含 messages。"""


class AgentState(MessagesState):
    """主图全局状态：消息记录 + 研究过程数据。

    - supervisor_messages：主管节点的消息记录（追加，可用 override 覆盖）
    - research_brief：整体研究简报（用户需求提炼）
    - sub_questions：decompose_question 拆解出的研究子问题
    - clarification_question：需要澄清时向用户提出的问题
    - raw_notes：研究员收集到的原始素材
    - notes：整理后的精简笔记
    - final_report：最终生成的报告
    """

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    sub_questions: list[str]
    clarification_question: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str


class SupervisorState(TypedDict):
    """主管子图状态。

    - supervisor_messages：主管对话消息
    - research_brief：整体研究简报
    - sub_questions：待研究的研究子问题
    - notes：整理后的研究笔记
    - researched_topics：已经成功研究过的主题（规范化后），用于代码层去重
    - research_batches：实际执行过的 ConductResearch 批次数量
    - research_iterations：supervisor 节点调用次数，仅作为防止无限循环的兜底
    - raw_notes：研究员原始素材
    """

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    sub_questions: list[str]
    notes: Annotated[list[str], override_reducer] = []
    researched_topics: Annotated[list[str], override_reducer] = []
    research_batches: int = 0
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []


class ResearcherState(TypedDict):
    """单个研究员子图状态。

    - researcher_messages：研究员自己的对话消息，只追加不覆盖（operator.add）
    - tool_call_iterations：该研究员调用工具的轮次，用于最大次数限流
    - research_topic：分配给该研究员的研究子课题
    - compressed_research：压缩精简后的研究结果
    - raw_notes：研究员收集到的原始素材
    """

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


class SupervisorOutputState(TypedDict):
    """主管子图对外输出。

    只回传研究结果相关的字段，避免把子图内部的消息（supervisor_messages）
    重复追加回主图状态。
    """

    notes: Annotated[list[str], override_reducer] = []
    raw_notes: Annotated[list[str], override_reducer] = []


class ResearcherOutputState(BaseModel):
    """单个研究员的输出：压缩后的研究结果 + 原始素材。"""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
