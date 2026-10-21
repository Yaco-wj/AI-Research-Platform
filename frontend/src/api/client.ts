/**
 * Backend API 客户端：所有请求模型与后端 Pydantic Schema 一一对应。
 */

import type {
  ResearchCreateRequest,
  ResearchCreateResponse,
  ResearchReportResponse,
  ResearchResumeRequest,
  ResearchResumeResponse,
  ResearchTask,
  ResearchTaskListResponse,
} from "../types/research";

/** Backend API 前缀（默认走 Vite 代理到 http://localhost:8000） */
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

/** 统一的请求封装：非 2xx 时抛出后端返回的错误信息。 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    let detail = `请求失败（HTTP ${response.status}）`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // 忽略非 JSON 响应体
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

/** 创建研究任务 */
export function createResearch(payload: ResearchCreateRequest): Promise<ResearchCreateResponse> {
  return request<ResearchCreateResponse>("/research", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** 查询研究任务详情 */
export function getResearch(taskId: string): Promise<ResearchTask> {
  return request<ResearchTask>(`/research/${taskId}`);
}

/** 查询研究历史 */
export function listResearch(limit = 20, offset = 0): Promise<ResearchTaskListResponse> {
  return request<ResearchTaskListResponse>(`/research?limit=${limit}&offset=${offset}`);
}

/** 获取最终报告 */
export function getResearchReport(taskId: string): Promise<ResearchReportResponse> {
  return request<ResearchReportResponse>(`/research/${taskId}/report`);
}

/** 回答澄清问题并恢复研究 */
export function resumeResearch(
  taskId: string,
  payload: ResearchResumeRequest,
): Promise<ResearchResumeResponse> {
  return request<ResearchResumeResponse>(`/research/${taskId}/resume`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** 构造 SSE 地址；sinceId 用于断点续传，避免重连时重复回放事件 */
export function buildStreamUrl(taskId: string, sinceId = 0): string {
  const suffix = sinceId > 0 ? `?since_id=${sinceId}` : "";
  return `${API_BASE}/research/${taskId}/stream${suffix}`;
}
