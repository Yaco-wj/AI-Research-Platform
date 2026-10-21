/** 澄清问题面板：Agent interrupt 后由用户补充信息。 */

import { useState } from "react";

interface ClarificationPanelProps {
  question: string;
  submitting: boolean;
  onSubmit: (answer: string) => void;
}

export function ClarificationPanel({ question, submitting, onSubmit }: ClarificationPanelProps) {
  const [answer, setAnswer] = useState("");

  const handleSubmit = () => {
    const trimmed = answer.trim();
    if (trimmed && !submitting) {
      onSubmit(trimmed);
      setAnswer("");
    }
  };

  return (
    <section className="panel panel--accent">
      <h2 className="panel__title">需要进一步澄清</h2>
      <p className="clarification__question">{question || "请补充你的研究重点。"}</p>
      <textarea
        className="research-input__field"
        rows={3}
        placeholder="例如：我希望重点研究企业级落地"
        value={answer}
        onChange={(event) => setAnswer(event.target.value)}
      />
      <div className="research-input__actions">
        <span className="research-input__hint">回答会通过 Command(resume=...) 传回 Agent</span>
        <button
          type="button"
          className="button button--primary"
          onClick={handleSubmit}
          disabled={submitting || answer.trim().length === 0}
        >
          {submitting ? "正在继续研究…" : "继续研究"}
        </button>
      </div>
    </section>
  );
}
