/** 研究执行过程时间线：阶段 + 子课题。 */

import type { ResearchPhase, ResearchTopic } from "../hooks/useResearchStream";

interface ResearchTimelineProps {
  phases: ResearchPhase[];
  topics: ResearchTopic[];
}

function phaseIcon(state: ResearchPhase["state"]): string {
  if (state === "done") {
    return "✓";
  }
  if (state === "active") {
    return "⟳";
  }
  return "○";
}

export function ResearchTimeline({ phases, topics }: ResearchTimelineProps) {
  return (
    <section className="panel">
      <h2 className="panel__title">执行过程</h2>

      <ol className="timeline">
        {phases.map((phase) => (
          <li key={phase.key} className={`timeline__item timeline__item--${phase.state}`}>
            <span className="timeline__icon">{phaseIcon(phase.state)}</span>
            <span className="timeline__label">{phase.label}</span>
          </li>
        ))}
      </ol>

      {topics.length > 0 && (
        <div className="topics">
          <h3 className="topics__title">研究子课题</h3>
          <ul className="topics__list">
            {topics.map((topic) => (
              <li key={topic.topic} className="topics__item">
                <span className="topics__name">
                  {topic.state === "done" ? "✓" : "⟳"} {topic.topic}
                </span>
                {topic.summary && <span className="topics__summary">{topic.summary}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
