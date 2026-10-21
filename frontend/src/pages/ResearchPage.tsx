/** 研究过程 / 报告页面：/research/:taskId */

import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getResearch, getResearchReport, resumeResearch } from "../api/client";
import { ClarificationPanel } from "../components/ClarificationPanel";
import { ReportView } from "../components/ReportView";
import { ResearchTimeline } from "../components/ResearchTimeline";
import { useResearchStream } from "../hooks/useResearchStream";
import type { ResearchStatus, ResearchTask } from "../types/research";
import { STATUS_LABELS, formatTime } from "../utils/status";

/**
 * 路由入口：用 taskId 作为 key 强制重新挂载详情组件。
 *
 * 这样浏览器前进/后退在两个任务之间切换时，不会复用上一个任务的
 * 状态 / 报告 / 错误信息（这是“详情页显示 FAILED，但库里是 COMPLETED”的一个来源）。
 */
export function ResearchPage() {
  const { taskId } = useParams<{ taskId: string }>();
  if (!taskId) {
    return <p className="muted">缺少研究任务 ID。</p>;
  }
  return <ResearchDetail key={taskId} taskId={taskId} />;
}

function ResearchDetail({ taskId }: { taskId: string }) {
  const [task, setTask] = useState<ResearchTask | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportFromApi, setReportFromApi] = useState("");
  const [answerSubmitting, setAnswerSubmitting] = useState(false);

  const { status, question, report, error, phases, topics, finished, reconnect } =
    useResearchStream(taskId);
  // SSE 中的 report_chunk 优先，其次使用报告接口兜底获取的内容
  const displayReport = report || reportFromApi;

  const refreshTask = useCallback(async () => {
    try {
      const payload = await getResearch(taskId);
      setTask(payload);
      setLoadError(null);
      return payload;
    } catch (cause) {
      setLoadError((cause as Error).message);
      return null;
    }
  }, [taskId]);

  // 首次进入页面时读取任务基本信息（问题、时间、状态）
  useEffect(() => {
    void refreshTask();
  }, [refreshTask]);

  // 事件流结束（完成 / 失败 / 等待澄清 / 重试耗尽）后回读业务库：
  // 业务库才是状态的权威来源，避免 SSE 的最后一条事件覆盖真实终态。
  useEffect(() => {
    if (finished) {
      void refreshTask();
    }
  }, [finished, refreshTask]);

  // 事件流结束后以业务库状态为准；流未结束时 SSE 状态更新更及时
  const dbStatus = task?.status;
  const liveStatus: ResearchStatus = status !== "PENDING" ? status : (dbStatus ?? status);
  const effectiveStatus: ResearchStatus = finished && dbStatus ? dbStatus : liveStatus;

  // 任务已完成时不再展示过程错误文案（例如断线重连期间的旧 error 事件）
  const displayError = effectiveStatus === "FAILED" ? error : null;

  // 任务完成后若 SSE 未带来报告内容，则主动获取报告
  useEffect(() => {
    if (effectiveStatus !== "COMPLETED" || report) {
      return;
    }
    let active = true;
    setReportLoading(true);
    getResearchReport(taskId)
      .then((payload) => {
        if (active) {
          setReportFromApi(payload.report);
        }
      })
      .catch(() => undefined)
      .finally(() => {
        if (active) {
          setReportLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [taskId, effectiveStatus, report]);

  const handleAnswer = useCallback(
    async (answer: string) => {
      setAnswerSubmitting(true);
      setLoadError(null);
      try {
        await resumeResearch(taskId, { answer });
        // 服务端在澄清后会结束旧连接，这里重新建立 SSE 连接继续接收研究过程
        void refreshTask();
        reconnect();
      } catch (cause) {
        setLoadError((cause as Error).message);
      } finally {
        setAnswerSubmitting(false);
      }
    },
    [reconnect, refreshTask, taskId],
  );

  const clarificationQuestion = question ?? task?.clarification_question ?? "";

  return (
    <div className="page page--research">
      <div className="research-layout">
        <aside className="research-side">
          <Link to="/" className="back-link">
            ← 返回首页
          </Link>

          <section className="panel">
            <h2 className="panel__title">研究任务</h2>
            <p className="research-query">{task?.query ?? "正在加载任务信息…"}</p>
            <dl className="meta">
              <div className="meta__row">
                <dt>状态</dt>
                <dd>
                  <span className={`status status--${effectiveStatus.toLowerCase()}`}>
                    {STATUS_LABELS[effectiveStatus]}
                  </span>
                </dd>
              </div>
              <div className="meta__row">
                <dt>创建时间</dt>
                <dd>{formatTime(task?.created_at ?? null)}</dd>
              </div>
              <div className="meta__row">
                <dt>开始时间</dt>
                <dd>{formatTime(task?.started_at ?? null)}</dd>
              </div>
              <div className="meta__row">
                <dt>完成时间</dt>
                <dd>{formatTime(task?.completed_at ?? null)}</dd>
              </div>
            </dl>
            {loadError && <p className="error-text">{loadError}</p>}
            {displayError && <p className="error-text">{displayError}</p>}
          </section>

          {effectiveStatus === "WAITING_USER" && (
            <ClarificationPanel
              question={clarificationQuestion}
              submitting={answerSubmitting}
              onSubmit={handleAnswer}
            />
          )}
        </aside>

        <main className="research-main">
          <header className="research-header">
            <h1 className="research-header__title">{task?.query ?? "研究任务"}</h1>
            <div className="research-header__meta">
              <span className={`status status--${effectiveStatus.toLowerCase()}`}>
                {STATUS_LABELS[effectiveStatus]}
              </span>
              <span>创建于 {formatTime(task?.created_at ?? null)}</span>
              {task?.completed_at && <span>完成于 {formatTime(task.completed_at)}</span>}
            </div>
          </header>
          <ResearchTimeline phases={phases} topics={topics} />
          <ReportView report={displayReport} loading={reportLoading} />
        </main>
      </div>
    </div>
  );
}
