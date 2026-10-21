"""本地运行入口：演示新的主流程。

主流程：clarify_with_user ->（需要澄清时 interrupt 等待用户回答）-> decompose_question
        -> research_supervisor（并发 researcher）-> final_report_generation
"""

import asyncio

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from open_deep_research.deep_researcher import build_deep_researcher

# 加载 .env 中的模型与搜索 API 密钥
load_dotenv()

# 用户研究需求（真实运行时这里可以换成命令行输入）
RESEARCH_QUERY = """
调研 2025-2026 年国内 AI Agent 在企业业务场景的落地情况。
报告类型：技术综述报告。
范围：中国大陆，面向企业内部办公、客服、数据查询场景；不包含 C 端聊天机器人。
受众：后端 AI 开发工程师，需要中等技术深度。
重点：落地难点、技术架构、使用的 Agent 框架。
"""

# 模拟用户对澄清问题的回答（真实场景下应从用户输入读取）
USER_CLARIFICATION_ANSWER = "我主要关注企业级 AI Agent，重点看 LangGraph、CrewAI、AutoGen、MCP 这几个方向。"


async def run_research():
    """运行一次完整研究：遇到 interrupt 时模拟用户回答并恢复执行。"""
    # 使用内存 checkpointer，interrupt 才有 checkpoint 可以恢复
    graph = build_deep_researcher(checkpointer=InMemorySaver())
    # 必须使用同一个 thread_id 才能恢复同一个中断
    thread_config = {"configurable": {"thread_id": "local-demo-1"}}

    result = await graph.ainvoke(
        {"messages": [("user", RESEARCH_QUERY)]},
        thread_config,
    )

    # 图被 interrupt 暂停：取出澄清问题，模拟用户回答后通过 Command(resume=...) 恢复
    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        print("===== 需要用户澄清 =====")
        print(interrupt_payload.get("question"))
        result = await graph.ainvoke(Command(resume=USER_CLARIFICATION_ANSWER), thread_config)

    print("===== state 里的所有 key =====")
    print(list(result.keys()))
    print("===== 拆解出的研究子问题 =====")
    for index, question in enumerate(result.get("sub_questions", []), 1):
        print(f"{index}. {question}")
    print("===== 最终研究报告 =====")
    print(result.get("final_report", ""))


if __name__ == "__main__":
    asyncio.run(run_research())
