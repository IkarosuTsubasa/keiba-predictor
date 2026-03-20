import React, { useState } from "react";
import AiPickSummary from "./AiPickSummary";
import ExpandablePredictionPanel from "./ExpandablePredictionPanel";
import RaceCardHeader from "./RaceCardHeader";

function estimateReadyTime(text, minutesBefore = 25) {
  const source = String(text || "").trim();
  if (!source) return "";
  const match = source.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  if (!match) return "";
  const [, year, month, day, hour, minute] = match;
  const date = new Date(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute));
  if (Number.isNaN(date.getTime())) return "";
  date.setMinutes(date.getMinutes() - minutesBefore);
  const hh = String(date.getHours()).padStart(2, "0");
  const mm = String(date.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

export default function RaceCard({ race, style = undefined }) {
  const [expanded, setExpanded] = useState(false);
  const cards = race?.cards || [];
  const isPlaceholder = Boolean(race?.is_placeholder);
  const hasCards = cards.length > 0;
  const estimatedReadyTime = estimateReadyTime(race?.scheduled_off_time, 25);

  return (
    <article
      className={`race-card${expanded ? " race-card--expanded" : ""}${isPlaceholder ? " race-card--placeholder" : ""}`}
      style={style}
    >
      <RaceCardHeader
        race={race}
        actions={
          !isPlaceholder && hasCards ? (
            <button type="button" className="race-card__toggle" onClick={() => setExpanded((value) => !value)}>
              {expanded ? "閉じる" : "詳細を見る"}
            </button>
          ) : null
        }
      />

      {isPlaceholder ? (
        <div className="race-card__summary-grid">
          <article className="ai-pick-summary ai-pick-summary--generic">
            <div className="ai-pick-summary__head">
              <strong className="ai-pick-summary__model">予測中</strong>
            </div>
            <p className="race-card__placeholder-time">
              {estimatedReadyTime ? `予測完了見込み ${estimatedReadyTime}ごろ` : "予測完了見込みを準備中です。"}
            </p>
          </article>
        </div>
      ) : (
        <div className="race-card__summary-grid">
          {cards.map((card) => (
            <AiPickSummary key={`${race.run_id}-${card.engine}`} card={card} />
          ))}
        </div>
      )}

      {!isPlaceholder && expanded && hasCards ? (
        <div className="race-card__detail-overlay" role="dialog" aria-modal="true" aria-label={`${race?.race_title || ""} 詳細`}>
          <div className="race-card__detail-backdrop" onClick={() => setExpanded(false)} />
          <div className="race-card__detail-sheet">
            <div className="race-card__detail-head">
              <div>
                <span className="race-card__detail-kicker">Race Detail</span>
                <strong>{race?.race_title || "-"}</strong>
              </div>
              <button type="button" className="race-card__detail-close" onClick={() => setExpanded(false)}>
                閉じる
              </button>
            </div>
            <ExpandablePredictionPanel cards={cards} />
          </div>
        </div>
      ) : null}
    </article>
  );
}
