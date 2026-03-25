import React, { useMemo } from "react";
import BetPreviewList from "./BetPreviewList";
import ModelMetaBadge from "./ModelMetaBadge";
import {
  MARK_ORDER,
  APP_BASE_PATH,
  formatRaceBadges,
  parseMarks,
  parseResultEntries,
  pickHorse,
} from "../lib/publicRace";

function buildBackHref(search) {
  const query = String(search || "").replace(/^\?/, "");
  return query ? `${APP_BASE_PATH}?${query}` : APP_BASE_PATH;
}

function buildConsensus(cards) {
  const tally = new Map();

  for (const card of cards || []) {
    const marks = parseMarks(card?.marks_text);
    const mainHorse = pickHorse(marks, "◎");
    if (!mainHorse) continue;
    tally.set(mainHorse, (tally.get(mainHorse) || 0) + 1);
  }

  const winner = [...tally.entries()]
    .map(([horseNo, count]) => ({ horseNo, count }))
    .sort(
      (left, right) =>
        right.count - left.count || Number(left.horseNo) - Number(right.horseNo),
    )[0];

  return winner || null;
}

function resolveLead(variant, race) {
  if (variant === "placeholder") {
    return String(
      race?.display_body?.message || "現在レースデータを反映しています。",
    );
  }
  if (variant === "settled") {
    return "各モデルの印・買い目・結果をまとめて比較できます。";
  }
  return "発走前の印と買い目を、モデルごとに見やすく整理しています。";
}

function resolveResultTone(text) {
  const source = String(text || "");
  if (source && !source.includes("未") && !source.includes("待ち")) {
    return "positive";
  }
  return "neutral";
}

function DetailSummary({ label, value, accent = false }) {
  return (
    <article
      className={`race-detail-summary-card${accent ? " race-detail-summary-card--accent" : ""}`}
    >
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </article>
  );
}

function CompareRow({ card }) {
  const marks = parseMarks(card?.marks_text);

  return (
    <article className="race-detail-compare-row">
      <div className="race-detail-compare-row__top">
        <strong>{card?.label || "-"}</strong>
        <ModelMetaBadge
          label="ROI"
          value={card?.roi_text || "-"}
          tone="subtle"
          dynamicRoi
        />
      </div>
      <div className="race-detail-compare-row__marks">
        {MARK_ORDER.map((symbol) => {
          const horseNo = pickHorse(marks, symbol) || "-";
          return (
            <span
              key={`${card?.engine || card?.label}-${symbol}`}
              className={symbol === "◎" ? "is-main" : ""}
            >
              <em>{symbol}</em>
              <strong>{horseNo}</strong>
            </span>
          );
        })}
      </div>
    </article>
  );
}

function ModelDetailCard({ card, highlightRoi = false }) {
  const marks = parseMarks(card?.marks_text);
  const resultText = String(card?.result_triplet_text || "結果はまだありません");

  return (
    <article className="race-detail-model-card">
      <div className="race-detail-model-card__head">
        <div>
          <span className="race-detail-model-card__eyebrow">モデル</span>
          <h3>{card?.label || "-"}</h3>
        </div>
        <div className="race-detail-model-card__badges">
          <ModelMetaBadge
            label="ROI"
            value={card?.roi_text || "-"}
            tone="subtle"
            dynamicRoi={highlightRoi}
          />
        </div>
      </div>

      <div className="race-detail-model-card__marks">
        {MARK_ORDER.map((symbol) => {
          const horseNo = pickHorse(marks, symbol) || "-";
          return (
            <span
              key={`${card?.engine || card?.label}-${symbol}`}
              className={symbol === "◎" ? "is-main" : ""}
            >
              <em>{symbol}</em>
              <strong>{horseNo}</strong>
            </span>
          );
        })}
      </div>

      <div className="race-detail-model-card__section">
        <div className="race-detail-model-card__section-head">
          <span>買い目</span>
        </div>
        <BetPreviewList text={card?.ticket_plan_text || ""} />
      </div>

      <div
        className={`race-detail-model-card__result race-detail-model-card__result--${resolveResultTone(resultText)}`}
      >
        <span>結果</span>
        <strong>{resultText}</strong>
      </div>
    </article>
  );
}

export default function RaceDetailPage({ race, search = "" }) {
  const cards = Array.isArray(race?.cards) ? race.cards : [];
  const variant = String(race?.display_variant || "").trim();
  const status = race?.display_status || {};
  const resultText = String(
    race?.display_body?.result_text || "結果は確定後に表示されます",
  );
  const resultEntries = parseResultEntries(resultText);
  const consensus = useMemo(() => buildConsensus(cards), [cards]);
  const badges = formatRaceBadges(race);
  const backHref = buildBackHref(search);

  return (
    <section className="race-detail-page">
      <div className="race-detail-hero" id="race-detail-summary">
        <div className="race-detail-hero__copy">
          <a className="race-detail-back-link" href={backHref}>
            一覧へ戻る
          </a>
          <span className="race-detail-hero__eyebrow">レース詳細</span>
          <h1>{race?.display_header?.title || "-"}</h1>
          {badges.length ? (
            <div className="race-detail-hero__badges">
              {badges.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          ) : null}
          <p>{resolveLead(variant, race)}</p>
        </div>

        <div className="race-detail-hero__meta">
          <DetailSummary label="状態" value={status?.label || "公開中"} accent />
          <DetailSummary label="公開モデル" value={`${cards.length}モデル`} />
          <DetailSummary
            label="本命集中"
            value={
              consensus
                ? `◎ ${consensus.horseNo} / ${consensus.count}モデル`
                : "分散"
            }
          />
          <DetailSummary
            label="結果"
            value={resultEntries.length ? `${resultEntries.length}着まで確定` : resultText}
          />
        </div>
      </div>

      <div className="race-detail-layout">
        <section
          className="race-detail-panel race-detail-panel--compare"
          id="race-detail-compare"
        >
          <div className="race-detail-panel__head">
            <div>
              <span className="race-detail-panel__eyebrow">比較</span>
              <h2>モデル別の本命比較</h2>
            </div>
          </div>
          <div className="race-detail-compare-list">
            {cards.map((card) => (
              <CompareRow key={card?.engine || card?.label} card={card} />
            ))}
          </div>
        </section>

        <section className="race-detail-panel race-detail-panel--result">
          <div className="race-detail-panel__head">
            <div>
              <span className="race-detail-panel__eyebrow">結果</span>
              <h2>レース結果</h2>
            </div>
          </div>
          {resultEntries.length ? (
            <ol className="race-detail-result-list">
              {resultEntries.map((entry) => (
                <li key={entry.key}>
                  <span>{entry.rank}着</span>
                  <strong>{entry.body}</strong>
                </li>
              ))}
            </ol>
          ) : (
            <p className="race-detail-result-placeholder">{resultText}</p>
          )}
        </section>
      </div>

      <section
        className="race-detail-panel race-detail-panel--models"
        id="race-detail-models"
      >
        <div className="race-detail-panel__head">
          <div>
            <span className="race-detail-panel__eyebrow">買い目</span>
            <h2>モデル別の購入プラン</h2>
          </div>
        </div>
        <div className="race-detail-model-grid">
          {cards.map((card) => (
            <ModelDetailCard
              key={card?.engine || card?.label}
              card={card}
              highlightRoi={variant === "settled"}
            />
          ))}
        </div>
      </section>
    </section>
  );
}
