import React, { useMemo } from "react";
import AutoFitLine from "./AutoFitLine";
import EzoicAdSlot from "./EzoicAdSlot";
import RaceDetailAffiliateCard from "./RaceDetailAffiliateCard";
import {
  APP_BASE_PATH,
  MARK_ORDER,
  formatRaceBadges,
  parseMarks,
  parseResultEntries,
  pickHorse,
} from "../lib/publicRace";

const MAIN_MARK = MARK_ORDER[0];
const MARK_WEIGHT = { "◎": 5, "○": 4, "▲": 3, "△": 2, "☆": 1 };

function buildBackHref(search) {
  const query = String(search || "").replace(/^\?/, "");
  return query ? `${APP_BASE_PATH}?${query}` : APP_BASE_PATH;
}

function buildHorseSignalRows(predictorCards = []) {
  const tally = new Map();

  for (const card of predictorCards || []) {
    const marks = parseMarks(card?.marks_text);
    if (!marks.length) continue;
    const seen = new Set();

    for (const mark of marks) {
      const horseNo = String(mark?.horseNo || "").trim();
      if (!horseNo) continue;
      const entry = tally.get(horseNo) || {
        horseNo,
        score: 0,
        supportCount: 0,
        mainCount: 0,
      };
      entry.score += MARK_WEIGHT[mark.symbol] || 0;
      if (mark.symbol === MAIN_MARK) {
        entry.mainCount += 1;
      }
      if (!seen.has(horseNo)) {
        entry.supportCount += 1;
        seen.add(horseNo);
      }
      tally.set(horseNo, entry);
    }
  }

  const rows = [...tally.values()].sort(
    (left, right) =>
      right.score - left.score ||
      right.mainCount - left.mainCount ||
      right.supportCount - left.supportCount ||
      Number(left.horseNo) - Number(right.horseNo),
  );
  const topScore = Number(rows[0]?.score || 0);

  return rows.map((row) => ({
    ...row,
    aiIndex: topScore > 0 ? Math.max(1, Math.round((row.score / topScore) * 100)) : 0,
  }));
}

function buildConfidenceMeta(signalRows, totalSources) {
  const top = signalRows[0] || null;
  const next = signalRows[1] || null;
  if (!top || totalSources <= 0) {
    return { stars: "☆☆☆☆☆", supportText: "-", modelCountText: "0モデル" };
  }

  const supportRatio = top.mainCount / totalSources;
  const marginRatio = top.score > 0
    ? Math.max(0, (top.score - Number(next?.score || 0)) / top.score)
    : 0;
  const score = Math.max(0, Math.min(1, 0.55 * supportRatio + 0.45 * marginRatio));
  const filledStars = Math.max(1, Math.min(5, Math.round(score * 4) + 1));
  return {
    stars: `${"★".repeat(filledStars)}${"☆".repeat(5 - filledStars)}`,
    supportText: `${top.mainCount}/${totalSources}モデル支持`,
    modelCountText: `${totalSources}モデル`,
  };
}

function resolveLead(variant, race, signalRows, totalSources) {
  if (variant === "placeholder") {
    return String(
      race?.display_body?.message ||
        "公開データの準備中です。更新後にこのレースの詳細が表示されます。",
    );
  }
  const top = signalRows[0] || null;
  if (!top) {
    return "このレースの定量モデル比較データを上から順に確認できます。";
  }
  return `6つの定量モデルを束ねた AI本命は ◎${top.horseNo} です。上位候補と各モデルの印をそのまま見比べられます。`;
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

function SpotlightCard({ eyebrow, title, value, meta = [], accent = false }) {
  return (
    <article className={`race-detail-spotlight-card${accent ? " race-detail-spotlight-card--accent" : ""}`}>
      <div className="race-detail-spotlight-card__head">
        <span>{eyebrow}</span>
      </div>
      <div className="race-detail-spotlight-card__main">
        <span>{title}</span>
        <strong>{value || "-"}</strong>
      </div>
      {meta.length ? (
        <div className="race-detail-spotlight-card__stats">
          {meta.map((item) => (
            <div key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value || "-"}</strong>
            </div>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function AiIndexPanel({ signalRows }) {
  return (
    <section className="race-detail-panel" id="race-detail-index">
      <div className="race-detail-panel__head">
        <div>
          <span className="race-detail-panel__eyebrow">AI指数</span>
          <h2>上位候補ランキング</h2>
        </div>
      </div>
      {signalRows.length ? (
        <div className="race-detail-index-list">
          {signalRows.slice(0, 5).map((item, index) => (
            <article key={item.horseNo} className="race-detail-index-item">
              <span>{`0${index + 1}`.slice(-2)}</span>
              <strong>{item.horseNo}</strong>
              <em>{item.aiIndex}</em>
            </article>
          ))}
        </div>
      ) : (
        <PanelEmpty>AI指数を計算できるデータがまだありません。</PanelEmpty>
      )}
    </section>
  );
}

function PredictorCompareRow({ card }) {
  const marks = parseMarks(card?.marks_text);
  const mainHorse = pickHorse(marks, MAIN_MARK) || "-";

  return (
    <article className="race-detail-compare-row">
      <div className="race-detail-compare-row__top">
        <strong>{card?.label || "-"}</strong>
        <em>{`${MAIN_MARK}${mainHorse}`}</em>
      </div>
      <MarkGrid
        marks={marks}
        itemKey={card?.predictor_id || card?.label || "predictor-compare"}
      />
    </article>
  );
}

export default function RaceDetailPage({ race, search = "", appShell = false }) {
  const predictorCompareCards = Array.isArray(race?.predictor_compare_cards)
    ? race.predictor_compare_cards.filter(Boolean)
    : [];
  const variant = String(race?.display_variant || "").trim();
  const status = race?.display_status || {};
  const resultText = String(race?.display_body?.result_text || "結果はまだありません");
  const resultEntries = parseResultEntries(resultText);
  const signalRows = useMemo(
    () => buildHorseSignalRows(predictorCompareCards),
    [predictorCompareCards],
  );
  const confidenceMeta = useMemo(
    () => buildConfidenceMeta(signalRows, predictorCompareCards.length),
    [predictorCompareCards.length, signalRows],
  );
  const badges = formatRaceBadges(race).filter(
    (item) => !["良", "稍重", "重", "不良"].includes(String(item || "").trim()),
  );
  const backHref = buildBackHref(search);
  const detailTitle = String(
    race?.display_header?.detail_title || race?.display_header?.title || "-",
  ).trim() || "-";
  const topHorse = signalRows[0] || null;

  return (
    <section className="race-detail-page">
      <div className="race-detail-hero" id="race-detail-summary">
        <div className="race-detail-hero__copy">
          <a className="race-detail-back-link" href={backHref}>
            予測一覧に戻る
          </a>
          <span className="race-detail-hero__eyebrow">レース詳細</span>
          <AutoFitLine
            as="h1"
            className="race-detail-hero__title"
            maxFontSize={44}
            minFontSize={14}
          >
            {detailTitle}
          </AutoFitLine>
          {badges.length ? (
            <div className="race-detail-hero__badges">
              {badges.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          ) : null}
          <p>{resolveLead(variant, race, signalRows, predictorCompareCards.length)}</p>
        </div>

        <div className="race-detail-hero__meta">
          <DetailSummary label="公開状態" value={status?.label || "公開中"} accent />
          <DetailSummary label="掲載モデル" value={confidenceMeta.modelCountText} />
          <DetailSummary label="本命支持" value={confidenceMeta.supportText} />
          <DetailSummary label="AI信頼度" value={confidenceMeta.stars} />
        </div>
      </div>

      <section className="race-detail-panel" id="race-detail-conclusion">
        <div className="race-detail-panel__head">
          <div>
            <span className="race-detail-panel__eyebrow">結論</span>
            <h2>AI本命</h2>
          </div>
        </div>
        <div className="race-detail-spotlight-grid">
          <SpotlightCard
            eyebrow="AI本命"
            title="本命馬"
            value={topHorse ? `◎${topHorse.horseNo}` : "-"}
            meta={[
              { label: "本命票", value: `${topHorse?.mainCount || 0}/${predictorCompareCards.length || 0}` },
              { label: "AI指数", value: `${topHorse?.aiIndex || 0}` },
            ]}
            accent
          />
          <SpotlightCard
            eyebrow="AI信頼度"
            title="信頼度"
            value={confidenceMeta.stars}
            meta={[
              { label: "支持状況", value: confidenceMeta.supportText },
            ]}
          />
          <SpotlightCard
            eyebrow="モデル数"
            title="定量モデル"
            value={confidenceMeta.modelCountText}
            meta={[
              { label: "比較対象", value: "main / v2 / v3 / v4 / v5 / v6" },
            ]}
          />
          <SpotlightCard
            eyebrow="結果"
            title="レース結果"
            value={resultEntries[0] ? resultEntries[0].body : "未確定"}
            meta={resultEntries.slice(1, 3).map((entry) => ({
              label: `${entry.rank}着`,
              value: entry.body,
            }))}
          />
        </div>
      </section>

      <AiIndexPanel signalRows={signalRows} />

      <section
        className="race-detail-panel race-detail-panel--compare"
        id="race-detail-compare"
      >
        <div className="race-detail-panel__head">
          <div>
            <span className="race-detail-panel__eyebrow">定量モデル予想</span>
            <h2>6モデルの本命比較</h2>
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
            <PanelEmpty>定量モデルの比較データはまだありません。</PanelEmpty>
          )}
        </div>
      </section>

      <section className="race-detail-panel race-detail-panel--result" id="race-detail-result">
        <div className="race-detail-panel__head">
          <div>
            <span className="race-detail-panel__eyebrow">レース結果</span>
            <h2>確定着順</h2>
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

      {!appShell ? (
        <EzoicAdSlot
          slot="raceDetailSidebar"
          wrapperClassName="ezoic-ad-slot--content"
        />
      ) : null}
      <RaceDetailAffiliateCard />
    </section>
  );
}
