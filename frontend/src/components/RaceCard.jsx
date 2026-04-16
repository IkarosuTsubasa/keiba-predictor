import React from "react";
import MorningRaceSummary from "./MorningRaceSummary";
import RaceCardHeader from "./RaceCardHeader";
import { buildRaceDetailHref } from "../lib/publicRace";

export default function RaceCard({ race, style = undefined }) {
  const cards = Array.isArray(race?.predictor_compare_cards) && race.predictor_compare_cards.length
    ? race.predictor_compare_cards
    : [];
  const variant = String(race?.display_variant || "").trim();
  const isPlaceholder = variant === "placeholder";
  const hasCards = cards.length > 0;
  const hasDetail = Boolean(String(race?.run_id || race?.race_id || "").trim());
  const hasTop5 = Array.isArray(race?.top5) && race.top5.length > 0;
  const canRenderAggregate = !isPlaceholder && (hasTop5 || hasCards);
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
  const isLinkable = !isPlaceholder && hasDetail;

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
      ) : canRenderAggregate ? (
        <div className="race-card__summary-grid race-card__summary-grid--single">
          <MorningRaceSummary race={race} />
        </div>
      ) : (
        <div className="race-card__summary-grid">
          <article className="ai-pick-summary ai-pick-summary--generic">
            <div className="ai-pick-summary__head">
              <strong className="ai-pick-summary__model">総合予測を準備中</strong>
            </div>
            <p className="race-card__placeholder-time">詳細ページで最新の公開状況を確認できます。</p>
          </article>
        </div>
      )}
    </article>
  );
}
