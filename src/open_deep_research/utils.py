"""Deep Research Agent 的工具函数与辅助方法。"""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

##########################
# Tavily 搜索工具
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "面向全面、准确、可信结果的搜索引擎，适合回答需要最新信息的问题。"
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """调用 Tavily 搜索 API，抓取并摘要搜索结果。

    Args:
        queries: 需要执行的搜索查询列表
        max_results: 每个查询返回的最大结果数
        topic: 搜索结果的主题过滤（general、news 或 finance）
        config: 运行时配置（API Key 与模型设置）

    Returns:
        包含摘要后搜索结果的格式化字符串
    """
    # 第 1 步：并发执行多个搜索查询
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )

    # 第 2 步：按 URL 去重，避免重复处理同一份内容
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    # 第 3 步：根据配置初始化摘要模型
    configurable = Configuration.from_runnable_config(config)

    # 控制字符长度，避免超出模型 token 上限（可配置）
    max_char_to_include = configurable.max_content_length

    # 初始化带重试逻辑的摘要模型
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )

    # 第 4 步：构建摘要任务（跳过没有正文的结果）
    async def noop():
        """没有原始正文时使用的空操作函数。"""
        return None

    summarization_tasks = [
        noop() if not result.get("raw_content")
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]

    # 第 5 步：并发执行全部摘要任务
    summaries = await asyncio.gather(*summarization_tasks)

    # 第 6 步：把搜索结果与摘要合并
    summarized_results = {
        url: {
            'title': result['title'],
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(),
            unique_results.values(),
            summaries
        )
    }

    # 第 7 步：格式化最终输出
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output

async def tavily_search_async(
    search_queries,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig = None
):
    """并发执行多个 Tavily 搜索查询。

    Args:
        search_queries: 需要执行的搜索查询字符串列表
        max_results: 每个查询返回的最大结果数
        topic: 用于过滤结果的主题分类
        include_raw_content: 是否返回完整网页正文
        config: 运行时配置（用于读取 API Key）

    Returns:
        Tavily API 返回的搜索结果字典列表
    """
    # 使用配置中的 API Key 初始化 Tavily 客户端
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))

    # 构建可并发执行的搜索任务
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]

    # 并发执行全部搜索查询并返回结果
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """使用 AI 模型对网页内容做摘要，并带超时保护。

    Args:
        model: 配置好的摘要模型
        webpage_content: 需要摘要的网页原始内容

    Returns:
        带关键摘录的格式化摘要；摘要失败时返回原始内容
    """
    try:
        # 构造带当前日期上下文的提示词
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=get_today_str()
        )

        # 带超时执行摘要，避免长时间阻塞
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 摘要超时时间：60 秒
        )

        # 按结构化小节格式化摘要
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except asyncio.TimeoutError:
        # 摘要超时：返回原始内容
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        # 摘要过程中的其他错误：记录日志并返回原始内容
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

##########################
# 反思工具
##########################

@tool(description="用于研究规划与进展反思的策略思考工具")
def think_tool(reflection: str) -> str:
    """用于对研究进展与决策进行策略性反思的工具。

    每次搜索之后使用该工具，系统性地分析结果并规划下一步。
    它会在研究流程中制造一次刻意的停顿，以便做出高质量的决策。

    使用时机：
    - 收到搜索结果之后：我找到了哪些关键信息？
    - 决定下一步之前：现有信息是否足以完整回答？
    - 评估研究缺口时：我还缺少哪些具体信息？
    - 结束研究之前：我现在能给出完整答案了吗？

    反思内容应涵盖：
    1. 当前发现分析——我收集到了哪些具体信息？
    2. 缺口评估——还有哪些关键信息缺失？
    3. 质量评估——我是否有足够的证据或案例支撑一个好的答案？
    4. 策略决策——应该继续搜索，还是给出答案？

    Args:
        reflection: 对研究进展、发现、缺口与下一步计划的详细反思

    Returns:
        确认反思已记录，可用于后续决策
    """
    return f"反思已记录：{reflection}"

##########################
# MCP 工具
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """通过 OAuth token exchange 把 Supabase token 换成 MCP 访问 token。

    Args:
        supabase_token: 有效的 Supabase 认证 token
        base_mcp_url: MCP 服务器的基础 URL

    Returns:
        成功时返回 token 数据字典，失败时返回 None
    """
    try:
        # 准备 OAuth token exchange 请求参数
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }

        # 发起 token exchange 请求
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # 成功获取 token
                    token_data = await response.json()
                    return token_data
                else:
                    # 记录错误详情，便于排查
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")

    except Exception as e:
        logging.error(f"Error during token exchange: {e}")

    return None

async def get_tokens(config: RunnableConfig):
    """读取已存储的认证 token，并校验是否过期。

    Args:
        config: 包含 thread 与 user 标识的运行时配置

    Returns:
        有效且未过期时返回 token 字典，否则返回 None
    """
    store = get_store()

    # 从配置中读取必要的标识
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None

    # 读取已存储的 token
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None

    # 校验 token 是否过期
    expires_in = tokens.value.get("expires_in")  # 距离过期还有多少秒
    created_at = tokens.created_at  # token 创建时间
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)

    if current_time > expiration_time:
        # token 已过期：清理并返回 None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """把认证 token 写入配置存储。

    Args:
        config: 包含 thread 与 user 标识的运行时配置
        tokens: 需要存储的 token 字典
    """
    store = get_store()

    # 从配置中读取必要的标识
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return

    # 存储 token
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """获取并刷新 MCP token，必要时重新申请。

    Args:
        config: 包含认证信息的运行时配置

    Returns:
        有效的 token 字典；无法获取时返回 None
    """
    # 优先使用已有的有效 token
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens

    # 取出 Supabase token，用于换取新的 MCP token
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None

    # 读取 MCP 配置
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None

    # 用 Supabase token 换取 MCP token
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # 存储并返回新的 token
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """为 MCP 工具补充完整的认证与错误处理逻辑。

    Args:
        tool: 需要包装的 MCP 结构化工具

    Returns:
        带认证错误处理的增强工具
    """
    original_coroutine = tool.coroutine

    async def authentication_wrapper(**kwargs):
        """增强版协程：处理 MCP 错误并给出对用户友好的提示。"""

        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """在异常链中递归查找 MCP 错误。"""
            if isinstance(exc, McpError):
                return exc

            # 处理 ExceptionGroup（Python 3.11+）：检查其子异常
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None

        try:
            # 执行工具原有逻辑
            return await original_coroutine(**kwargs)

        except BaseException as original_error:
            # 在异常链中查找 MCP 专属错误
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # 不是 MCP 错误：原样抛出
                raise original_error

            # 处理 MCP 专属错误
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}

            # 检查是否为“需要用户交互/认证”的错误
            if error_code == -32003:  # 需要交互的错误码
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"

                # 提取对用户友好的提示信息
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message

                # 如果提供了 URL，一并附上供用户参考
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"

                raise ToolException(error_message) from original_error

            # 其他 MCP 错误：原样抛出
            raise original_error

    # 用增强版协程替换工具原有协程
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """加载并配置带认证的 MCP（Model Context Protocol）工具。

    Args:
        config: 包含 MCP 服务器信息的运行时配置
        existing_tool_names: 已被占用的工具名集合，用于避免命名冲突

    Returns:
        配置完成、可直接使用的 MCP 工具列表
    """
    configurable = Configuration.from_runnable_config(config)

    # 第 1 步：按需处理认证
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None

    # 第 2 步：校验配置完整性
    config_valid = (
        configurable.mcp_config and
        configurable.mcp_config.url and
        configurable.mcp_config.tools and
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )

    if not config_valid:
        return []

    # 第 3 步：建立 MCP 服务器连接
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"

    # 如果有 token，则配置认证请求头
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}

    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO：等 OAP 合并多 MCP 服务器支持后，再更新这里

    # 第 4 步：从 MCP 服务器加载工具
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # MCP 服务器连接失败时返回空列表
        return []

    # 第 5 步：过滤并配置工具
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # 跳过同名的冲突工具
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue

        # 只保留配置中指定要暴露的工具
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue

        # 包装认证处理逻辑后加入结果列表
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)

    return configured_tools


##########################
# 工具集合
##########################

async def get_search_tool(search_api: SearchAPI):
    """根据指定的搜索 API 提供方构造并返回搜索工具。

    Args:
        search_api: 使用的搜索 API（Anthropic、OpenAI、Tavily 或 None）

    Returns:
        对应提供方的搜索工具列表
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic 原生网页搜索（带调用次数限制）
        return [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5
        }]

    elif search_api == SearchAPI.OPENAI:
        # OpenAI 原生网页搜索
        return [{"type": "web_search_preview"}]

    elif search_api == SearchAPI.TAVILY:
        # 配置 Tavily 搜索工具及其元数据
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search"
        }
        return [search_tool]

    elif search_api == SearchAPI.NONE:
        # 未配置任何搜索能力
        return []

    # 未知搜索 API 类型的兜底处理
    return []

async def get_all_tools(config: RunnableConfig):
    """组装完整工具集：研究工具 + 搜索工具 + MCP 工具。

    Args:
        config: 指定搜索 API 与 MCP 设置的运行时配置

    Returns:
        研究流程可用的全部工具列表
    """
    # 先加入核心研究工具
    tools = [tool(ResearchComplete), think_tool]

    # 加入配置好的搜索工具
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)

    # 记录已有工具名，避免命名冲突
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search")
        for tool in tools
    }

    # 若配置了 MCP，则加入 MCP 工具
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)

    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """从工具消息中提取研究笔记。"""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# 模型原生网页搜索工具
##########################

def anthropic_websearch_called(response):
    """判断本次响应是否使用了 Anthropic 原生网页搜索。

    Args:
        response: Anthropic API 返回的响应对象

    Returns:
        使用了网页搜索返回 True，否则返回 False
    """
    try:
        # 逐层读取响应元数据
        usage = response.response_metadata.get("usage")
        if not usage:
            return False

        # 检查服务端工具使用信息
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False

        # 查找网页搜索的调用次数
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False

        # 只要发生过网页搜索就返回 True
        return web_search_requests > 0

    except (AttributeError, TypeError):
        # 响应结构不符合预期时的兜底处理
        return False

def openai_websearch_called(response):
    """判断本次响应是否使用了 OpenAI 网页搜索能力。

    Args:
        response: OpenAI API 返回的响应对象

    Returns:
        使用了网页搜索返回 True，否则返回 False
    """
    # 检查响应元数据中的工具输出
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False

    # 在工具输出中查找网页搜索调用
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True

    return False


##########################
# Token 超限判断工具
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """判断某个异常是否表示触发了 token / 上下文长度上限。

    Args:
        exception: 需要分析的异常
        model_name: 可选的模型名，用于更准确地识别提供方

    Returns:
        异常表示 token 超限时返回 True，否则返回 False
    """
    error_str = str(exception).lower()

    # 第 1 步：尽可能根据模型名判断提供方
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'

    # 第 2 步：按提供方检查对应的 token 超限特征
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)

    # 第 3 步：无法判断提供方时，逐一检查所有提供方的特征
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """判断异常是否表示 OpenAI token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 判断是否为 OpenAI 的异常
    is_openai_exception = (
        'openai' in exception_type.lower() or
        'openai' in module_name.lower()
    )

    # 检查 OpenAI 典型的 token 超限错误类型
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']

    if is_openai_exception and is_request_error:
        # 在错误信息中查找与 token 相关的关键词
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True

    # 检查 OpenAI 的特定错误码
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')

        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True

    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """判断异常是否表示 Anthropic token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 判断是否为 Anthropic 的异常
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or
        'anthropic' in module_name.lower()
    )

    # 检查 Anthropic 特有的错误特征
    is_bad_request = class_name == 'BadRequestError'

    if is_anthropic_exception and is_bad_request:
        # Anthropic 在 token 超限时会返回特定错误信息
        if 'prompt is too long' in error_str:
            return True

    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """判断异常是否表示 Google/Gemini token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 判断是否为 Google/Gemini 的异常
    is_google_exception = (
        'google' in exception_type.lower() or
        'google' in module_name.lower()
    )

    # 检查 Google 特有的资源耗尽错误
    is_resource_exhausted = class_name in [
        'ResourceExhausted',
        'GoogleGenerativeAIFetchError'
    ]

    if is_google_exception and is_resource_exhausted:
        return True

    # 检查 Google API 资源耗尽的特定特征
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True

    return False

# 注意：该表可能已经过时，或与你的模型不匹配，请按需更新。
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """查询指定模型的 token 上限。

    Args:
        model_string: 需要查询的模型标识字符串

    Returns:
        查到时返回 token 上限整数，未查到返回 None
    """
    # 在已知模型 token 上限表中查找
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit

    # 未在表中找到该模型
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """通过删除最后一条 AI 消息之前的内容来截断消息历史。

    这适用于在 token 超限时裁剪上下文。

    Args:
        messages: 需要截断的消息列表

    Returns:
        截断后的消息列表（保留到但不包含最后一条 AI 消息）
    """
    # 从后往前查找最后一条 AI 消息
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # 返回最后一条 AI 消息之前的所有内容
            return messages[:i]

    # 没有找到 AI 消息时返回原列表
    return messages

##########################
# 其他工具函数
##########################

def get_today_str() -> str:
    """获取用于提示词与输出展示的当前日期字符串。

    Returns:
        便于阅读的日期字符串，例如 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """从配置中取值，兼容枚举与 None。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """从环境变量或配置中获取指定模型的 API Key。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"):
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    """从环境变量或配置中获取 Tavily API Key。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")
