import React from "react";
import AiPickSummary from "./AiPickSummary";
import MorningRaceSummary from "./MorningRaceSummary";
import RaceCardHeader from "./RaceCardHeader";
import { buildRaceDetailHref } from "../lib/publicRace";

export default function RaceCard({ race, style = undefined }) {
  const cards = Array.isArray(race?.predictor_compare_cards) && race.predictor_compare_cards.length
    ? race.predictor_compare_cards
    : [];
  const variant = String(race?.display_variant || "").trim();
  const isPlaceholder = variant === "placeholder";
  const isSettled = variant === "settled";
  const isMorningPreview = variant === "morning_preview";
  const hasCards = cards.length > 0;
  const hasDetail = Boolean(String(race?.run_id || race?.race_id || "").trim());
  const placeholderTitle = String(race?.display_body?.title || "公開準備中");
  const placeholderMessage = String(
    race?.display_body?.message || "現在レースデータを反映しています。",
  );
  const detailHref = buildRaceDetailHref(race, window.location.search);
  const handleNavigate = () => {
    if (!hasDetail) return;
    window.location.assign(detailHref);
  };
  const handleCardClick = (event) => {
    if (!hasDetail) return;
    if (event.target instanceof Element && event.target.closest("a, button")) {
      return;
    }
    handleNavigate();
  };
  const handleCardKeyDown = (event) => {
    if (!hasDetail) return;
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    if (event.target instanceof Element && event.target.closest("a, button")) {
      return;
    }
    event.preventDefault();
    handleNavigate();
  };
  const isLinkable = !isPlaceholder && hasDetail && (isMorningPreview || hasCards);

  return (
    <article
      className={`race-card${isPlaceholder ? " race-card--placeholder" : ""}${isLinkable ? " race-card--linkable" : ""}`}
      style={style}
      onClick={isLinkable ? handleCardClick : undefined}
      onKeyDown={isLinkable ? handleCardKeyDown : undefined}
      role={isLinkable ? "link" : undefined}
      tabIndex={isLinkable ? 0 : undefined}
    >
      <RaceCardHeader
        race={race}
        actions={
          isLinkable ? (
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
      ) : isMorningPreview ? (
        <div className="race-card__summary-grid race-card__summary-grid--single">
          <MorningRaceSummary race={race} />
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
