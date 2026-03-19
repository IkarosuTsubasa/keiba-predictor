import React, { useState } from "react";
import AiPickSummary from "./AiPickSummary";
import ExpandablePredictionPanel from "./ExpandablePredictionPanel";
import RaceCardHeader from "./RaceCardHeader";

export default function RaceCard({ race }) {
  const [expanded, setExpanded] = useState(false);
  const cards = race?.cards || [];

  return (
    <article className={`race-card${expanded ? " race-card--expanded" : ""}`}>
      <RaceCardHeader
        race={race}
        actions={
          <button type="button" className="race-card__toggle" onClick={() => setExpanded((value) => !value)}>
            {expanded ? "閉じる" : "詳細を見る"}
          </button>
        }
      />

      <div className="race-card__summary-grid">
        {cards.map((card) => (
          <AiPickSummary key={`${race.run_id}-${card.engine}`} card={card} />
        ))}
      </div>

      {expanded ? (
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
