/**
 * 与 Backend Pydantic Schema 保持一致的 TypeScript 类型定义。
 */

export type ResearchStatus =
  | "PENDING"
  | "RUNNING"
  | "WAITING_USER"
  | "COMPLETED"
  | "FAILED";

/** POST /api/v1/research 请求体 */
export interface ResearchCreateRequest {
  query: string;
}

/** POST /api/v1/research 响应体 */
export interface ResearchCreateResponse {
  task_id: string;
  thread_id: string;
  status: ResearchStatus;
}

/** GET /api/v1/research/{task_id} 响应体 */
export interface ResearchTask {
  task_id: string;
  thread_id: string;
  query: string;
  title: string | null;
  status: ResearchStatus;
  clarification_question: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
}

/** GET /api/v1/research 响应体 */
export interface ResearchTaskListResponse {
  total: number;
  items: ResearchTask[];
}

/** POST /api/v1/research/{task_id}/resume 请求体 */
export interface ResearchResumeRequest {
  answer: string;
}

/** POST /api/v1/research/{task_id}/resume 响应体 */
export interface ResearchResumeResponse {
  task_id: string;
  thread_id: string;
  status: ResearchStatus;
}

/** GET /api/v1/research/{task_id}/report 响应体 */
export interface ResearchReportResponse {
  task_id: string;
  report: string;
}

/** SSE 事件名 */
export type ResearchEventName =
  | "status"
  | "node_started"
  | "node_completed"
  | "research_started"
  | "research_completed"
  | "clarification_required"
  | "report_chunk"
  | "completed"
  | "error"
  | "stream_unavailable";

/** SSE 事件负载（不同事件携带的字段不同） */
export interface ResearchEvent {
  event: ResearchEventName;
  task_id: string;
  status?: ResearchStatus;
  node?: string;
  scope?: string;
  topic?: string;
  summary?: string;
  question?: string;
  content?: string;
  message?: string;
}
