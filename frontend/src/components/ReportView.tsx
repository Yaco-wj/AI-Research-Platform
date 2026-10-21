/** 最终报告展示：Markdown 渲染。 */

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ReportViewProps {
  report: string;
  loading: boolean;
}

export function ReportView({ report, loading }: ReportViewProps) {
  return (
    <section className="panel">
      <h2 className="panel__title">最终报告</h2>
      {report ? (
        <article className="markdown">
          <Markdown remarkPlugins={[remarkGfm]}>{report}</Markdown>
        </article>
      ) : (
        <p className="muted">{loading ? "正在加载报告…" : "研究报告生成后会显示在这里。"}</p>
      )}
    </section>
  );
}
