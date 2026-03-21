import React, { useState } from "react";
import AiPickSummary from "./AiPickSummary";
import ExpandablePredictionPanel from "./ExpandablePredictionPanel";
import RaceCardHeader from "./RaceCardHeader";

export default function RaceCard({ race, style = undefined }) {
  const [expanded, setExpanded] = useState(false);
  const cards = race?.cards || [];
  const variant = String(race?.display_variant || "").trim();
  const isPlaceholder = variant === "placeholder";
  const isSettled = variant === "settled";
  const hasCards = cards.length > 0;
  const placeholderTitle = String(race?.display_body?.title || "予測中");
  const placeholderMessage = String(
    race?.display_body?.message || "予測完了見込みを準備中です。",
  );

  return (
    <article
      className={`race-card${expanded ? " race-card--expanded" : ""}${isPlaceholder ? " race-card--placeholder" : ""}`}
      style={style}
    >
      <RaceCardHeader
        race={race}
        actions={
          !isPlaceholder && hasCards ? (
            <button
              type="button"
              className="race-card__toggle"
              onClick={() => setExpanded((value) => !value)}
            >
              {expanded ? "閉じる" : "詳細を見る"}
            </button>
          ) : null
        }
      />

      {isPlaceholder ? (
        <div className="race-card__summary-grid">
          <article className="ai-pick-summary ai-pick-summary--generic">
            <div className="ai-pick-summary__head">
              <strong className="ai-pick-summary__model">
                {placeholderTitle}
              </strong>
            </div>
            <p className="race-card__placeholder-time">{placeholderMessage}</p>
          </article>
        </div>
      ) : (
        <div className="race-card__summary-grid">
          {cards.map((card) => (
            <AiPickSummary
              key={`${race.run_id}-${card.engine}`}
              card={card}
              highlightRoi={isSettled}
            />
          ))}
        </div>
      )}

      {!isPlaceholder && expanded && hasCards ? (
        <div
          className="race-card__detail-overlay"
          role="dialog"
          aria-modal="true"
          aria-label={`${race?.display_header?.title || "-"} 詳細`}
        >
          <div
            className="race-card__detail-backdrop"
            onClick={() => setExpanded(false)}
          />
          <div className="race-card__detail-sheet">
            <div className="race-card__detail-head">
              <div>
                <span className="race-card__detail-kicker">Race Detail</span>
                <strong>{race?.display_header?.title || "-"}</strong>
              </div>
              <button
                type="button"
                className="race-card__detail-close"
                onClick={() => setExpanded(false)}
              >
                閉じる
              </button>
            </div>
            <ExpandablePredictionPanel cards={cards} highlightRoi={isSettled} />
          </div>
        </div>
      ) : null}
    </article>
  );
}
