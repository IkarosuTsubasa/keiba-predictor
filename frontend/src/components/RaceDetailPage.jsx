import React, { useMemo } from "react";
import BetPreviewList from "./BetPreviewList";
import ModelMetaBadge from "./ModelMetaBadge";
import {
  APP_BASE_PATH,
  MARK_ORDER,
  formatRaceBadges,
  parseMarks,
  parseResultEntries,
  pickHorse,
} from "../lib/publicRace";

const MAIN_MARK = MARK_ORDER[0];

function buildBackHref(search) {
  const query = String(search || "").replace(/^\?/, "");
  return query ? `${APP_BASE_PATH}?${query}` : APP_BASE_PATH;
}

function parsePlanCount(text) {
  const lines = String(text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return 0;
  if (lines.length === 1 && /買い目なし/.test(lines[0])) {
    return 0;
  }
  return lines.filter((line) => !/買い目なし/.test(line)).length;
}

function buildConsensusRows(cards, predictorCards = []) {
  const tally = new Map();

  for (const card of [...(cards || []), ...(predictorCards || [])]) {
    const marks = parseMarks(card?.marks_text);
    const mainHorse = pickHorse(marks, MAIN_MARK);
    if (!mainHorse) continue;
    tally.set(mainHorse, (tally.get(mainHorse) || 0) + 1);
  }

  return [...tally.entries()]
    .map(([horseNo, count]) => ({ horseNo, count }))
    .sort(
      (left, right) =>
        right.count - left.count || Number(left.horseNo) - Number(right.horseNo),
    );
}

function resolveLead(variant, race) {
  if (variant === "placeholder") {
    return String(
      race?.display_body?.message ||
        "公開データの準備中です。更新後にこのレースの詳細が表示されます。",
    );
  }
  if (variant === "settled") {
    return "AIモデルの買い目と量化モデルの本命比較を、確定結果とあわせて確認できます。";
  }
  return "AIモデルの買い目と量化モデルの本命比較を、同じ画面でまとめて確認できます。";
}

function resolveResultTone(text) {
  const source = String(text || "");
  if (!source || /未|なし/.test(source)) {
    return "neutral";
  }
  return "positive";
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

function PanelEmpty({ children }) {
  return <p className="race-detail-empty-note">{children}</p>;
}

function MarkGrid({ marks, itemKey }) {
  return (
    <div className="race-detail-mark-grid">
      {MARK_ORDER.map((symbol) => {
        const horseNo = pickHorse(marks, symbol) || "-";
        return (
          <span
            key={`${itemKey}-${symbol}`}
            className={symbol === MAIN_MARK ? "is-main" : ""}
          >
            <em>{symbol}</em>
            <strong>{horseNo}</strong>
          </span>
        );
      })}
    </div>
  );
}

function ConsensusPanel({ consensusRows }) {
  return (
    <section className="race-detail-panel race-detail-panel--consensus">
      <div className="race-detail-panel__head">
        <div>
          <span className="race-detail-panel__eyebrow">総合評価</span>
          <h2>AI・量化モデル総合本命</h2>
        </div>
      </div>
      {consensusRows.length ? (
        <div className="race-detail-consensus-list">
          {consensusRows.slice(0, 3).map((item, index) => (
            <article key={item.horseNo} className="race-detail-consensus-item">
              <span>{`0${index + 1}`.slice(-2)}</span>
              <strong>{`${MAIN_MARK}${item.horseNo}`}</strong>
              <em>{`${item.count}モデル支持`}</em>
            </article>
          ))}
        </div>
      ) : (
        <PanelEmpty>総合本命を集計できるデータがまだありません。</PanelEmpty>
      )}
    </section>
  );
}

function PredictorCompareRow({ card }) {
  const marks = parseMarks(card?.marks_text);

  return (
    <article className="race-detail-compare-row">
      <div className="race-detail-compare-row__top">
        <strong>{card?.label || "-"}</strong>
      </div>
      <MarkGrid
        marks={marks}
        itemKey={card?.predictor_id || card?.label || "predictor-compare"}
      />
    </article>
  );
}

function ModelDetailCard({ card, highlightRoi = false }) {
  const marks = parseMarks(card?.marks_text);
  const resultText = String(card?.result_triplet_text || "結果未確定");
  const planCount = parsePlanCount(card?.ticket_plan_text);

  return (
    <article className="race-detail-model-card">
      <div className="race-detail-model-card__head">
        <div>
          <span className="race-detail-model-card__eyebrow">AIモデル</span>
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

      <MarkGrid marks={marks} itemKey={card?.engine || card?.label || "model"} />

      <div className="race-detail-model-card__stats">
        <DetailSummary label="買い目数" value={`${planCount || 0}件`} />
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
  const cards = Array.isArray(race?.cards) ? race.cards.filter(Boolean) : [];
  const predictorCompareCards = Array.isArray(race?.predictor_compare_cards)
    ? race.predictor_compare_cards.filter(Boolean)
    : [];
  const variant = String(race?.display_variant || "").trim();
  const status = race?.display_status || {};
  const resultText = String(race?.display_body?.result_text || "結果はまだありません");
  const resultEntries = parseResultEntries(resultText);
  const consensusRows = useMemo(
    () => buildConsensusRows(cards, predictorCompareCards),
    [cards, predictorCompareCards],
  );
  const badges = formatRaceBadges(race).filter(
    (item) => !["良", "稍重", "重", "不良"].includes(String(item || "").trim()),
  );
  const backHref = buildBackHref(search);
  const totalPlanCount = cards.reduce(
    (sum, card) => sum + parsePlanCount(card?.ticket_plan_text),
    0,
  );

  return (
    <section className="race-detail-page">
      <div className="race-detail-hero" id="race-detail-summary">
        <div className="race-detail-hero__copy">
          <a className="race-detail-back-link" href={backHref}>
            予測一覧に戻る
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
          <DetailSummary label="公開状態" value={status?.label || "公開中"} accent />
          <DetailSummary label="買い目総数" value={`${totalPlanCount}件`} />
        </div>
      </div>

      <section
        className="race-detail-panel race-detail-panel--models"
        id="race-detail-models"
      >
        <div className="race-detail-panel__head">
          <div>
            <span className="race-detail-panel__eyebrow">買い目一覧</span>
            <h2>AIモデル別の買い目</h2>
          </div>
        </div>
        <div className="race-detail-model-grid">
          {cards.length ? (
            cards.map((card) => (
              <ModelDetailCard
                key={card?.engine || card?.label}
                card={card}
                highlightRoi={variant === "settled"}
              />
            ))
          ) : (
            <PanelEmpty>買い目データはまだ公開されていません。</PanelEmpty>
          )}
        </div>
      </section>

      <div className="race-detail-layout">
        <section
          className="race-detail-panel race-detail-panel--compare"
          id="race-detail-compare"
        >
          <div className="race-detail-panel__head">
            <div>
              <span className="race-detail-panel__eyebrow">量化モデル</span>
              <h2>量化モデルの本命比較</h2>
            </div>
          </div>
          <div className="race-detail-compare-list">
            {predictorCompareCards.length ? (
              predictorCompareCards.map((card) => (
                <PredictorCompareRow
                  key={card?.predictor_id || card?.label}
                  card={card}
                />
              ))
            ) : (
              <PanelEmpty>量化モデルの比較データはまだありません。</PanelEmpty>
            )}
          </div>
        </section>

        <div className="race-detail-side-stack">
          <ConsensusPanel consensusRows={consensusRows} />

          <section className="race-detail-panel race-detail-panel--result">
            <div className="race-detail-panel__head">
              <div>
                <span className="race-detail-panel__eyebrow">確定結果</span>
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
      </div>
    </section>
  );
}
