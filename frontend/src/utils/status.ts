/** 任务状态的中文文案与样式类名。 */

import type { ResearchStatus } from "../types/research";

export const STATUS_LABELS: Record<ResearchStatus, string> = {
  PENDING: "排队中",
  RUNNING: "正在研究",
  WAITING_USER: "等待澄清",
  COMPLETED: "已完成",
  FAILED: "失败",
};

/** 把 ISO 时间格式化为可读文本 */
export function formatTime(value: string | null): string {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }
  return date.toLocaleString("zh-CN", { hour12: false });
}
