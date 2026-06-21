import React from "react";
import AutoFitLine from "./AutoFitLine";
import MorningRaceSummary from "./MorningRaceSummary";
import { resolvePublicDecision } from "../lib/confidencePolicy";
import { buildRaceDetailHref } from "../lib/publicRace";

const MARK_LABELS = ["◎", "○", "▲", "△", "☆"];

function formatConfidence(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${Math.round(number * 100)}%`;
}

function isTrackConditionBadge(value) {
  return ["良", "稍重", "重", "不良"].includes(String(value || "").trim());
}

function parseResultEntries(text) {
  const source = String(text || "").trim();
  if (!source || source.includes("未") || source.includes("確定前")) {
    return [];
  }
  return source
    .split("/")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 3)
    .map((item, index) => {
      const match = item.match(/^([1-3])着\s*(.+)$/);
      const rank = Number(match?.[1] || index + 1);
      const body = String(match?.[2] || item).trim();
      return { key: `${rank}-${body}`, rank, body };
    })
    .filter((item) => item.body);
}

function resolveStatus(race) {
  const status = race?.display_status || {};
  return {
    label: String(status.label || "").trim() || "公開中",
    tone: String(status.tone || "").trim() || "open",
  };
}

function mainHorse(race) {
  const top5 = Array.isArray(race?.top5) ? race.top5 : [];
  return top5[0] || null;
}

function topMarksText(race) {
  const top5 = Array.isArray(race?.top5) ? race.top5 : [];
  const marks = top5
    .filter(Boolean)
    .slice(0, MARK_LABELS.length)
    .map((item, index) => {
      const horseNo = String(item?.horse_no || "").trim();
      if (!horseNo) return "";
      return `${MARK_LABELS[index] || ""}${horseNo}`;
    })
    .filter(Boolean);
  return marks.length ? marks.join(" / ") : "-";
}

function normalizeHorseNo(value) {
  const match = String(value || "").match(/\d+/);
  if (!match) return "";
  return String(Number(match[0]));
}

function normalizeHorseName(value) {
  return String(value || "").replace(/\s+/g, "").trim();
}

function predictionMarkForResult(race, body) {
  const top5 = Array.isArray(race?.top5) ? race.top5 : [];
  const bodyText = String(body || "").trim();
  const resultNo = normalizeHorseNo(bodyText.match(/^(\d+)/)?.[1] || "");
  const resultName = normalizeHorseName(bodyText.replace(/^\d+\s*/, ""));

  for (let index = 0; index < top5.length && index < MARK_LABELS.length; index += 1) {
    const item = top5[index] || {};
    const mark = MARK_LABELS[index] || "";
    const horseNo = normalizeHorseNo(item?.horse_no);
    const horseName = normalizeHorseName(item?.horse_name);
    if (resultNo && horseNo && resultNo === horseNo) {
      return mark;
    }
    if (resultName && horseName && resultName === horseName) {
      return mark;
    }
  }
  return "";
}

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
  const decision = resolvePublicDecision(race);
  const status = resolveStatus(race);
  const main = mainHorse(race);
  const confidenceText = formatConfidence(race?.confidence_score);
  const markText = topMarksText(race);
  const title = String(race?.display_header?.title || "-");
  const subtitle = String(race?.display_header?.subtitle || "").trim();
  const badges = Array.isArray(race?.display_header?.badges)
    ? race.display_header.badges.filter(
        (item) => item && !isTrackConditionBadge(item),
      )
    : [];
  const isMorningPreview = variant === "morning_preview";
  const showResult = !isPlaceholder && !isMorningPreview;
  const resultText = String(
    race?.display_body?.result_text || "結果は確定後に表示されます",
  );
  const resultEntries = parseResultEntries(resultText);
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
      <div className="race-card__race-cell">
        <div className="race-card-header">
          <div className="race-card-header__main">
            <div className="race-card-header__title-group">
              <h3>{isPlaceholder ? placeholderTitle : title}</h3>
              {subtitle && !isPlaceholder ? (
                <AutoFitLine
                  as="p"
                  className="race-card-header__subtitle"
                  maxFontSize={16}
                  minFontSize={10}
                >
                  {subtitle}
                </AutoFitLine>
              ) : null}
              {badges.length && !isPlaceholder ? (
                <div className="race-card-header__badges">
                  {badges.map((item) => (
                    <span key={item}>{item}</span>
                  ))}
                </div>
              ) : null}
              {isPlaceholder ? (
                <p className="race-card__placeholder-time">{placeholderMessage}</p>
              ) : null}
            </div>
            {!isPlaceholder ? (
              <span
                className={`race-card-header__status race-card-header__status--${status.tone}`}
              >
                {status.label}
              </span>
            ) : null}
          </div>
        </div>
      </div>

      {!isPlaceholder ? (
        <>
          <div className={`race-card__decision-cell race-card__decision-cell--${decision.tone}`}>
            <span className="race-card__cell-label">判断</span>
            <strong>{decision.label}</strong>
          </div>
          <div className="race-card__metric-cell">
            <span className="race-card__cell-label">信頼度</span>
            <strong>{confidenceText}</strong>
          </div>
          <div className="race-card__main-cell">
            <span className="race-card__cell-label">本命</span>
            <strong>{main ? `◎ ${main.horse_no || "-"} ${main.horse_name || ""}`.trim() : "-"}</strong>
          </div>
          <div className="race-card__ticket-cell">
            <span className="race-card__cell-label">上位印</span>
            <strong>{markText}</strong>
          </div>
          <div className="race-card__result-cell">
            <span className="race-card__cell-label">結果</span>
            {showResult && resultEntries.length ? (
              <ul className="race-card__result-list">
                {resultEntries.map((entry) => {
                  const hitMark = predictionMarkForResult(race, entry.body);
                  return (
                    <li key={entry.key}>
                      <span className="race-card__result-medal" aria-hidden="true">
                        {entry.rank}着
                      </span>
                      <span className="race-card__result-body">
                        {hitMark ? (
                          <span
                            className="race-card__result-hit"
                            title={`予測印 ${hitMark}`}
                            aria-label={`予測印 ${hitMark}`}
                          >
                            🌸
                          </span>
                        ) : null}
                        <span>{entry.body}</span>
                      </span>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p>{showResult ? resultText : "結果は確定後に表示されます"}</p>
            )}
          </div>
          <div className="race-card__detail-cell">
            {isLinkable ? (
              <a href={detailHref} className="race-card__toggle">
                詳細を見る
              </a>
            ) : null}
          </div>
        </>
      ) : null}

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
