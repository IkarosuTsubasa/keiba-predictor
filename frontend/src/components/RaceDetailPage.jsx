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

const MAIN_MARK = MARK_ORDER[0];

function buildBackHref(search) {
  const query = String(search || "").replace(/^\?/, "");
  return query ? `${APP_BASE_PATH}?${query}` : APP_BASE_PATH;
}

function parsePlanCount(text) {
  return String(text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean).length;
}

function buildConsensusRows(cards) {
  const tally = new Map();

  for (const card of cards || []) {
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
    return String(race?.display_body?.message || "公開データを準備中です。");
  }
  if (variant === "settled") {
    return "各モデルの印、買い目、払戻結果を一画面で比較できるレース詳細です。";
  }
  return "各モデルの本命印と購入プランを、比較しやすい形で整理しています。";
}

function resolveResultTone(text) {
  const source = String(text || "");
  if (!source || /未|準備/.test(source)) {
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

function SpotlightCard({ card }) {
  const marks = parseMarks(card?.marks_text);
  const mainHorse = pickHorse(marks, MAIN_MARK) || "-";
  const planCount = parsePlanCount(card?.ticket_plan_text);

  return (
    <article className="race-detail-spotlight-card">
      <div className="race-detail-spotlight-card__head">
        <strong>{card?.label || "-"}</strong>
        <ModelMetaBadge
          label="ROI"
          value={card?.roi_text || "-"}
          tone="subtle"
          dynamicRoi
        />
      </div>
      <div className="race-detail-spotlight-card__main">
        <span>{MAIN_MARK}</span>
        <strong>{mainHorse}</strong>
      </div>
      <div className="race-detail-spotlight-card__stats">
        <div>
          <span>買い目数</span>
          <strong>{planCount || 0}</strong>
        </div>
        <div>
          <span>結果</span>
          <strong>{card?.result_triplet_text || "未確定"}</strong>
        </div>
      </div>
    </article>
  );
}

function ConsensusPanel({ consensusRows }) {
  return (
    <section className="race-detail-panel race-detail-panel--consensus">
      <div className="race-detail-panel__head">
        <div>
          <span className="race-detail-panel__eyebrow">Consensus</span>
          <h2>本命コンセンサス</h2>
        </div>
      </div>
      {consensusRows.length ? (
        <div className="race-detail-consensus-list">
          {consensusRows.slice(0, 3).map((item, index) => (
            <article key={item.horseNo} className="race-detail-consensus-item">
              <span>{`0${index + 1}`.slice(-2)}</span>
              <strong>{`${MAIN_MARK}${item.horseNo}`}</strong>
              <em>{`${item.count}モデル`}</em>
            </article>
          ))}
        </div>
      ) : (
        <PanelEmpty>本命印の集計はまだありません。</PanelEmpty>
      )}
    </section>
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
      <MarkGrid marks={marks} itemKey={card?.engine || card?.label || "compare"} />
    </article>
  );
}

function ModelDetailCard({ card, highlightRoi = false }) {
  const marks = parseMarks(card?.marks_text);
  const resultText = String(card?.result_triplet_text || "結果はまだありません");
  const planCount = parsePlanCount(card?.ticket_plan_text);

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

      <MarkGrid marks={marks} itemKey={card?.engine || card?.label || "model"} />

      <div className="race-detail-model-card__stats">
        <DetailSummary label="買い目数" value={`${planCount || 0}件`} />
        <DetailSummary
          label="本命"
          value={`${MAIN_MARK}${pickHorse(marks, MAIN_MARK) || "-"}`}
          accent
        />
      </div>

      <div className="race-detail-model-card__section">
        <div className="race-detail-model-card__section-head">
          <span>購入プラン</span>
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
  const variant = String(race?.display_variant || "").trim();
  const status = race?.display_status || {};
  const resultText = String(race?.display_body?.result_text || "結果は公開後に表示されます。");
  const resultEntries = parseResultEntries(resultText);
  const consensusRows = useMemo(() => buildConsensusRows(cards), [cards]);
  const badges = formatRaceBadges(race);
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
            予測一覧へ戻る
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
          <DetailSummary label="公開モデル" value={`${cards.length}モデル`} />
          <DetailSummary
            label="買い目総数"
            value={`${totalPlanCount}件`}
          />
          <DetailSummary
            label="結果"
            value={resultEntries.length ? `${resultEntries.length}件の確定着順` : resultText}
          />
        </div>
      </div>

      <section className="race-detail-panel race-detail-panel--spotlights">
        <div className="race-detail-panel__head">
          <div>
            <span className="race-detail-panel__eyebrow">LLM Desk</span>
            <h2>各モデルの推奨サマリー</h2>
          </div>
        </div>
        {cards.length ? (
          <div className="race-detail-spotlight-grid">
            {cards.map((card) => (
              <SpotlightCard key={card?.engine || card?.label} card={card} />
            ))}
          </div>
        ) : (
          <PanelEmpty>公開モデルはまだありません。</PanelEmpty>
        )}
      </section>

      <div className="race-detail-layout">
        <section
          className="race-detail-panel race-detail-panel--compare"
          id="race-detail-compare"
        >
          <div className="race-detail-panel__head">
            <div>
              <span className="race-detail-panel__eyebrow">Compare</span>
              <h2>モデル別の本命比較</h2>
            </div>
          </div>
          <div className="race-detail-compare-list">
            {cards.length ? (
              cards.map((card) => (
                <CompareRow key={card?.engine || card?.label} card={card} />
              ))
            ) : (
              <PanelEmpty>比較対象のモデルはまだありません。</PanelEmpty>
            )}
          </div>
        </section>

        <div className="race-detail-side-stack">
          <ConsensusPanel consensusRows={consensusRows} />

          <section className="race-detail-panel race-detail-panel--result">
            <div className="race-detail-panel__head">
              <div>
                <span className="race-detail-panel__eyebrow">Result</span>
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

      <section
        className="race-detail-panel race-detail-panel--models"
        id="race-detail-models"
      >
        <div className="race-detail-panel__head">
          <div>
            <span className="race-detail-panel__eyebrow">Purchase Desk</span>
            <h2>モデル別の購入プラン</h2>
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
            <PanelEmpty>購入プランはまだ公開されていません。</PanelEmpty>
          )}
        </div>
      </section>
    </section>
  );
}
