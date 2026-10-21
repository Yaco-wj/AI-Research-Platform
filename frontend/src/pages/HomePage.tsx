/** 首页：新建研究 + 基础研究历史。 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { createResearch } from "../api/client";
import { HistoryList } from "../components/HistoryList";
import { ResearchInput } from "../components/ResearchInput";

export function HomePage() {
  const navigate = useNavigate();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (query: string) => {
    setSubmitting(true);
    setError(null);
    try {
      const created = await createResearch({ query });
      navigate(`/research/${created.task_id}`);
    } catch (cause) {
      setError((cause as Error).message);
      setSubmitting(false);
    }
  };

  return (
    <div className="page page--home">
      <header className="hero">
        <h1 className="hero__title">AI Research</h1>
        <p className="hero__subtitle">输入你想深入研究的问题</p>
      </header>

      <ResearchInput onSubmit={handleSubmit} submitting={submitting} />
      {error && <p className="error-text">{error}</p>}

      <section className="panel panel--plain">
        <h2 className="panel__title">研究历史</h2>
        <HistoryList />
      </section>
    </div>
  );
}
