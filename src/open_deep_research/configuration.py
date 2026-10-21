"""Open Deep Research 系统的配置管理。"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """可用的搜索 API 提供方枚举。"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"


class MCPConfig(BaseModel):
    """MCP（Model Context Protocol）服务器配置。"""

    url: Optional[str] = Field(
        default=None,
        optional=True,
        description="MCP 服务器的 URL",
    )
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
        description="允许提供给大模型使用的 MCP 工具列表",
    )
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
        description="MCP 服务器是否需要认证",
    )


class Configuration(BaseModel):
    """Deep Research Agent 的主配置类。"""

    # 通用配置
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "模型结构化输出调用的最大重试次数"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否允许研究开始前向用户提出澄清问题"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "允许并发执行的最大研究任务数，即同时启动的子研究员数量上限。注意：并发数越高，越容易触发模型速率限制。"
            }
        }
    )
    # 研究配置
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "研究使用的搜索 API。注意：请确认所选搜索 API 与研究模型互相兼容。",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI 原生网页搜索", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic 原生网页搜索", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "不使用搜索", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "研究主管的最大研究迭代次数，即主管反思研究进展并发起后续追问的次数上限。"
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "单个研究员在一次研究任务中允许的最大工具调用轮次。"
            }
        }
    )
    # 模型配置
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "用于对 Tavily 搜索结果等网页内容进行摘要的模型"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "摘要模型的最大输出 token 数"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "网页内容在送入摘要模型前的最大字符长度"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "用于开展研究（澄清、问题拆解、主管调度、子研究员）的模型。注意：请确认该模型与所选搜索 API 兼容。"
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "研究模型的最大输出 token 数"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "用于压缩子研究员研究结果的模型。注意：请确认该模型与所选搜索 API 兼容。"
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "压缩模型的最大输出 token 数"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "根据全部研究结果撰写最终报告的模型"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "最终报告模型的最大输出 token 数"
            }
        }
    )
    # MCP 服务器配置
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP 服务器配置"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "关于可用 MCP 工具的补充说明，会作为提示词传给 Agent。"
            }
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig（LangChain 运行时配置）与环境变量构造配置实例。

        优先级：环境变量（字段名大写）> configurable 中的显式配置 > 字段默认值。
        """
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            # 先读系统环境变量（字段名大写），读不到再取 configurable 中的值
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic 配置。"""

        arbitrary_types_allowed = True
