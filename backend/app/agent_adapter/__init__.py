"""Agent 适配层：Backend 与 Deep Research Agent 之间唯一的调用入口。"""

from backend.app.agent_adapter.research_agent import (
    AgentEvent,
    AgentStateView,
    ResearchAgentAdapter,
)

__all__ = ["AgentEvent", "AgentStateView", "ResearchAgentAdapter"]
