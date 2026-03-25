import React from "react";
import AiPickSummary from "./AiPickSummary";
import RaceCardHeader from "./RaceCardHeader";
import { buildRaceDetailHref } from "../lib/publicRace";

export default function RaceCard({ race, style = undefined }) {
  const cards = race?.cards || [];
  const variant = String(race?.display_variant || "").trim();
  const isPlaceholder = variant === "placeholder";
  const isSettled = variant === "settled";
  const hasCards = cards.length > 0;
  const placeholderTitle = String(race?.display_body?.title || "公開準備中");
  const placeholderMessage = String(
    race?.display_body?.message || "現在レースデータを反映しています。",
  );
  const detailHref = buildRaceDetailHref(race, window.location.search);

  return (
    <article
      className={`race-card${isPlaceholder ? " race-card--placeholder" : ""}`}
      style={style}
    >
      <RaceCardHeader
        race={race}
        actions={
          !isPlaceholder && hasCards ? (
            <a href={detailHref} className="race-card__toggle">
              詳細を見る
            </a>
          ) : null
        }
      />

      {isPlaceholder ? (
        <div className="race-card__summary-grid">
          <article className="ai-pick-summary ai-pick-summary--generic">
            <div className="ai-pick-summary__head">
              <strong className="ai-pick-summary__model">{placeholderTitle}</strong>
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
    </article>
  );
}
