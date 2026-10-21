/** 首页研究问题输入框。 */

import { useState } from "react";

interface ResearchInputProps {
  onSubmit: (query: string) => void;
  submitting: boolean;
}

export function ResearchInput({ onSubmit, submitting }: ResearchInputProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = () => {
    const trimmed = query.trim();
    if (trimmed && !submitting) {
      onSubmit(trimmed);
    }
  };

  return (
    <div className="research-input">
      <textarea
        className="research-input__field"
        placeholder="例如：研究 2026 年 AI Agent 的技术发展趋势"
        value={query}
        rows={4}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
            handleSubmit();
          }
        }}
      />
      <div className="research-input__actions">
        <span className="research-input__hint">Ctrl / ⌘ + Enter 快速提交</span>
        <button
          type="button"
          className="button button--primary"
          onClick={handleSubmit}
          disabled={submitting || query.trim().length === 0}
        >
          {submitting ? "正在创建研究任务…" : "开始研究"}
        </button>
      </div>
    </div>
  );
}
