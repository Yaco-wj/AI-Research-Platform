/**
 * 研究过程 SSE Hook：连接 Backend 的 /stream 接口，把真实事件转换成 UI 状态。
 *
 * 设计要点：
 * - 只使用 EventSource，不做轮询；
 * - 记录 lastEventId，重连时通过 since_id 跳过已处理事件；
 * - SSE 只是“过程订阅”，不是状态权威来源：事件流结束后由页面回读
 *   GET /research/{id}（业务库）作为最终状态，避免出现“详情页 FAILED 但库里 COMPLETED”；
 * - taskId 变化时重置全部状态，禁止把一个任务的状态/报告/错误带到另一个任务上。
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { buildStreamUrl } from "../api/client";
import type { ResearchEvent, ResearchEventName, ResearchStatus } from "../types/research";

/** 主流程阶段（与 Agent 主图节点一一对应） */
export interface ResearchPhase {
  key: string;
  label: string;
  state: "pending" | "active" | "done";
}

/** 单个研究子课题的进度 */
export interface ResearchTopic {
  topic: string;
  state: "active" | "done";
  summary?: string;
}

const MAIN_PHASES: { node: string; label: string }[] = [
  { node: "clarify_with_user", label: "分析研究需求" },
  { node: "decompose_question", label: "拆分研究问题" },
  { node: "research_supervisor", label: "安排与执行研究" },
  { node: "final_report_generation", label: "撰写最终报告" },
];

function initialPhases(): ResearchPhase[] {
  return MAIN_PHASES.map((phase) => ({ key: phase.node, label: phase.label, state: "pending" }));
}

/** 事件名白名单，避免把未知事件当作正常事件处理 */
const KNOWN_EVENTS: ResearchEventName[] = [
  "status",
  "node_started",
  "node_completed",
  "research_started",
  "research_completed",
  "clarification_required",
  "report_chunk",
  "completed",
  "error",
  "stream_unavailable",
];

/** 收到这些事件后服务端会主动结束当前 SSE 连接 */
const TERMINAL_EVENTS = new Set<ResearchEventName>([
  "completed",
  "error",
  "clarification_required",
]);

export interface UseResearchStreamResult {
  status: ResearchStatus;
  question: string | null;
  report: string;
  error: string | null;
  phases: ResearchPhase[];
  topics: ResearchTopic[];
  /** 事件流是否已经结束：页面据此回读业务库的权威状态（而不是一直停留在 SSE 的最后一条） */
  finished: boolean;
  reconnect: () => void;
}

export function useResearchStream(taskId: string | undefined): UseResearchStreamResult {
  const [status, setStatus] = useState<ResearchStatus>("PENDING");
  const [question, setQuestion] = useState<string | null>(null);
  const [report, setReport] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [phases, setPhases] = useState<ResearchPhase[]>(initialPhases);
  const [topics, setTopics] = useState<ResearchTopic[]>([]);
  const [finished, setFinished] = useState(false);

  const sourceRef = useRef<EventSource | null>(null);
  const lastEventIdRef = useRef(0);
  const statusRef = useRef<ResearchStatus>("PENDING");
  const retryTimerRef = useRef<number | null>(null);
  const retryCountRef = useRef(0);

  const applyEvent = useCallback((name: ResearchEventName, payload: ResearchEvent, id: number) => {
    if (id > 0) {
      lastEventIdRef.current = Math.max(lastEventIdRef.current, id);
    }
    // 收到任何事件说明连接正常，重置重连计数
    retryCountRef.current = 0;

    switch (name) {
      case "status": {
        if (payload.status) {
          statusRef.current = payload.status;
          setStatus(payload.status);
          if (payload.status !== "WAITING_USER") {
            setQuestion(null);
          }
        }
        break;
      }
      case "node_started": {
        setPhases((current) => {
          const index = current.findIndex((phase) => phase.key === payload.node);
          if (index === -1) {
            return current;
          }
          return current.map((phase, position) => {
            if (position < index) {
              return { ...phase, state: "done" };
            }
            if (position === index) {
              return { ...phase, state: "active" };
            }
            return phase;
          });
        });
        break;
      }
      case "node_completed": {
        setPhases((current) =>
          current.map((phase) =>
            phase.key === payload.node ? { ...phase, state: "done" } : phase,
          ),
        );
        break;
      }
      case "research_started": {
        if (payload.topic) {
          setTopics((current) => [...current, { topic: payload.topic as string, state: "active" }]);
        }
        break;
      }
      case "research_completed": {
        setTopics((current) => {
          const topic = payload.topic;
          const index = topic ? current.findIndex((item) => item.topic === topic) : -1;
          if (index === -1) {
            return [
              ...current,
              {
                topic: topic ?? "子课题研究",
                state: "done",
                summary: payload.summary,
              },
            ];
          }
          return current.map((item, position) =>
            position === index ? { ...item, state: "done", summary: payload.summary } : item,
          );
        });
        break;
      }
      case "clarification_required": {
        statusRef.current = "WAITING_USER";
        setQuestion(payload.question ?? "");
        setStatus("WAITING_USER");
        break;
      }
      case "report_chunk": {
        setReport((current) => current + (payload.content ?? ""));
        break;
      }
      case "completed": {
        statusRef.current = "COMPLETED";
        setStatus("COMPLETED");
        break;
      }
      case "error": {
        statusRef.current = "FAILED";
        setError(payload.message ?? "研究任务执行失败");
        setStatus("FAILED");
        break;
      }
      case "stream_unavailable": {
        // 当前进程没有实时事件流（例如服务重启）：历史回放已经结束，
        // 页面下一步会回读 GET /research/{id} 拿到权威状态与报告。
        break;
      }
      default:
        break;
    }

    if (TERMINAL_EVENTS.has(name)) {
      // 服务端在这类事件后会关闭连接：标记事件流已结束，交给页面回读业务库状态
      setFinished(true);
    }
  }, []);

  const connect = useCallback(() => {
    if (!taskId) {
      return;
    }
    if (retryTimerRef.current !== null) {
      window.clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    sourceRef.current?.close();
    setFinished(false);

    const source = new EventSource(buildStreamUrl(taskId, lastEventIdRef.current));
    sourceRef.current = source;

    const closeFromServer = () => {
      source.close();
      if (sourceRef.current === source) {
        sourceRef.current = null;
      }
    };

    KNOWN_EVENTS.forEach((name) => {
      source.addEventListener(name, (rawEvent) => {
        const message = rawEvent as MessageEvent<string>;
        let payload: ResearchEvent = { event: name, task_id: taskId };
        try {
          payload = { ...payload, ...(JSON.parse(message.data) as ResearchEvent) };
        } catch {
          // 忽略无法解析的事件
        }
        const id = Number(message.lastEventId || 0);
        applyEvent(name, payload, Number.isFinite(id) ? id : 0);

        // 服务端在这些事件后会主动结束当前 SSE 连接
        if (TERMINAL_EVENTS.has(name)) {
          closeFromServer();
        }
      });
    });

    source.onerror = () => {
      closeFromServer();
      // 连接意外中断时做有限次重连（不是轮询），断点续传避免事件重复
      const stillRunning = statusRef.current === "PENDING" || statusRef.current === "RUNNING";
      if (stillRunning && retryCountRef.current < 3) {
        retryCountRef.current += 1;
        retryTimerRef.current = window.setTimeout(() => connect(), 1500);
        return;
      }
      // 服务端已结束本次流（或重试耗尽）：由页面回读业务库确认最终状态
      setFinished(true);
    };
  }, [applyEvent, taskId]);

  // taskId 变化（浏览器前进/后退切换详情页、或直接跳到另一个任务）时先清空旧状态，
  // 否则新任务会显示上一个任务的状态 / 报告 / 错误信息，并沿用上一个任务的 lastEventId。
  useEffect(() => {
    lastEventIdRef.current = 0;
    statusRef.current = "PENDING";
    retryCountRef.current = 0;
    setStatus("PENDING");
    setQuestion(null);
    setReport("");
    setError(null);
    setPhases(initialPhases());
    setTopics([]);
    setFinished(false);
  }, [taskId]);

  useEffect(() => {
    retryCountRef.current = 0;
    connect();
    return () => {
      if (retryTimerRef.current !== null) {
        window.clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
      sourceRef.current?.close();
      sourceRef.current = null;
    };
  }, [connect]);

  return { status, question, report, error, phases, topics, finished, reconnect: connect };
}
