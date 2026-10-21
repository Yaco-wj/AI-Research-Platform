/** 基础研究历史列表。 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { listResearch } from "../api/client";
import type { ResearchTask } from "../types/research";
import { STATUS_LABELS } from "../utils/status";

export function HistoryList() {
  const [tasks, setTasks] = useState<ResearchTask[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    listResearch(8, 0)
      .then((payload) => {
        if (active) {
          setTasks(payload.items);
        }
      })
      .catch((cause: Error) => {
        if (active) {
          setError(cause.message);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  if (error) {
    return <p className="muted">研究历史暂时不可用：{error}</p>;
  }

  if (tasks.length === 0) {
    return <p className="muted">还没有研究记录，输入一个问题开始吧。</p>;
  }

  return (
    <ul className="history">
      {tasks.map((task) => (
        <li key={task.task_id} className="history__item">
          <Link className="history__link" to={`/research/${task.task_id}`}>
            <span className="history__title">{task.title ?? task.query}</span>
            <span className={`status status--${task.status.toLowerCase()}`}>
              {STATUS_LABELS[task.status]}
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}
