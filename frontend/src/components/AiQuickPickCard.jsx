import React from "react";
import ModelBadge from "./ModelBadge";

function parseLeadMark(text) {
  const source = String(text || "");
  const match = source.match(/◎\s*([0-9]+)/);
  return match ? match[1] : "-";
}

function toTicketLines(text) {
  return String(text || "")
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 3);
}

export default function AiQuickPickCard({ card }) {
  const lines = toTicketLines(card?.ticket_plan_text);

  return (
    <article className={`ai-quick-pick ai-quick-pick--${card?.engine || "neutral"}`}>
      <div className="ai-quick-pick__head">
        <ModelBadge engine={card?.engine} label={card?.label || "-"} subtle />
        <span className="ai-quick-pick__confidence">{card?.confidence_text || "N/A"}</span>
      </div>
      <div className="ai-quick-pick__hero">
        <span className="ai-quick-pick__mark-label">本命</span>
        <strong className="ai-quick-pick__horse">{parseLeadMark(card?.marks_text)}</strong>
      </div>
      <div className="ai-quick-pick__stats">
        <span>ROI {card?.roi_text || "-"}</span>
        <span>{card?.status_label || "公開中"}</span>
      </div>
      <ul className="ai-quick-pick__tickets">
        {lines.length ? lines.map((line) => <li key={`${card?.engine}-${line}`}>{line}</li>) : <li>買い目なし</li>}
      </ul>
    </article>
  );
}
