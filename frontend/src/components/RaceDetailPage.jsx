import React, { useMemo } from "react";
import AutoFitLine from "./AutoFitLine";
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
const APP_DOWNLOAD_HREF = "https://x.gd/BDVgd";
const APP_BADGE_SRC = "/keiba/GetItOnGooglePlay_Badge_Web_color_Japanese.png";
const PREDICTOR_LABELS = {
  main: "ゲート",
  v2_opus: "ストライド",
  v3_premium: "伯楽",
  v4_gemini: "馬場眼",
  v5_stacking: "フュージョン",
  v6_kiwami: "極 KIWAMI",
};
const PREDICTOR_ORDER = ["main", "v2_opus", "v3_premium", "v4_gemini", "v5_stacking", "v6_kiwami"];

function buildBackHref(search) {
  const query = String(search || "").replace(/^\?/, "");
  return query ? `${APP_BASE_PATH}?${query}` : APP_BASE_PATH;
}

function extractClockText(value) {
  const matched = String(value || "").trim().match(/(\d{2}:\d{2})/);
  return matched ? matched[1] : "";
}

function buildFullReleaseText(scheduledOffTime) {
  const offClock = extractClockText(scheduledOffTime);
  if (!offClock) {
    return "完全公開準備中";
  }
  const [hourText, minuteText] = offClock.split(":");
  const totalMinutes = Math.max(
    0,
    Number(hourText || 0) * 60 + Number(minuteText || 0) - 25,
  );
  const publishHour = String(Math.floor(totalMinutes / 60)).padStart(2, "0");
  const publishMinute = String(totalMinutes % 60).padStart(2, "0");
  return `${publishHour}:${publishMinute}頃完全公開`;
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

function buildMorningIndexRows(top5 = []) {
  return (Array.isArray(top5) ? top5 : [])
    .filter(Boolean)
    .map((item) => {
      const sourceCount = Array.isArray(item?.sources) ? item.sources.length : 0;
      return {
        horseNo: String(item?.horse_no || "").trim() || "-",
        horseName: String(item?.horse_name || "").trim() || "-",
        aiIndex: Number(item?.support_score || 0) || 0,
        score: Number(item?.support || 0) || 0,
        supportCount: sourceCount,
        mainCount: sourceCount,
      };
    })
    .sort((left, right) => right.aiIndex - left.aiIndex || right.score - left.score)
    .slice(0, 5);
}

function buildMorningCompareCards(predictorTop5, scheduledOffTime = "") {
  const source = predictorTop5 && typeof predictorTop5 === "object" ? predictorTop5 : {};
  const releaseText = buildFullReleaseText(scheduledOffTime);
  return PREDICTOR_ORDER
    .map((predictorId) => {
      const ranking = Array.isArray(source?.[predictorId]) ? source[predictorId].slice(0, 5) : [];
      if (!ranking.length) {
        return {
          predictor_id: predictorId,
          label: PREDICTOR_LABELS[predictorId] || predictorId,
          is_placeholder: true,
          placeholder_text: releaseText,
        };
      }
      const marksText = ranking
        .slice(0, MARK_ORDER.length)
        .map((item, index) => {
          const horseNo = String(item?.horse_no || "").trim();
          if (!horseNo) return "";
          return `${MARK_ORDER[index]}${horseNo}`;
        })
        .filter(Boolean)
        .join(" ");
      if (!marksText) return null;
      return {
        predictor_id: predictorId,
        label: PREDICTOR_LABELS[predictorId] || predictorId,
        marks_text: marksText,
      };
    })
    .filter(Boolean);
}

function buildConfidenceMeta(signalRows, totalSources) {
  const top = signalRows[0] || null;
  const next = signalRows[1] || null;
  if (!top || totalSources <= 0) {
    return { percentText: "-", supportText: "-", modelCountText: "0モデル", score: 0 };
  }

  const supportRatio = top.mainCount / totalSources;
  const marginRatio = top.score > 0
    ? Math.max(0, (top.score - Number(next?.score || 0)) / top.score)
    : 0;
  const score = Math.max(0, Math.min(1, 0.55 * supportRatio + 0.45 * marginRatio));
  return {
    percentText: `${Math.round(score * 100)}%`,
    supportText: `${top.mainCount}/${totalSources}モデル支持`,
    modelCountText: `${totalSources}モデル`,
    score,
  };
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

function DetailAppSummary() {
  return (
    <article className="race-detail-summary-card race-detail-summary-card--app">
      <span>アプリ</span>
      <a
        className="race-detail-summary-card__app-link"
        href={APP_DOWNLOAD_HREF}
        aria-label="Androidアプリをダウンロード"
      >
        <img
          className="race-detail-summary-card__app-badge"
          src={APP_BADGE_SRC}
          alt="Google Play で手に入れよう"
        />
      </a>
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
              <span>{item.horseNo}</span>
              <strong>{item.horseName || "-"}</strong>
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

function ConditionPredictorRankingPanel({ ranking }) {
  const cards = Array.isArray(ranking?.cards) ? ranking.cards : [];

  return (
    <section className="race-detail-panel race-detail-panel--condition-ranking" id="race-detail-condition-ranking">
      <div className="race-detail-panel__head">
        <div>
          <h2>この条件の定量モデル順位</h2>
        </div>
      </div>
      {ranking?.condition_text ? (
        <p className="race-detail-condition-ranking__meta">
          {ranking.condition_text}
          {ranking?.sample_count ? ` / 対象 ${ranking.sample_count}レース` : ""}
        </p>
      ) : null}
      {cards.length ? (
        <ol className="race-detail-condition-ranking__list">
          {cards.map((item) => (
            <li key={item.predictor_id} className="race-detail-condition-ranking__item">
              <span>{`${item.rank || "-"}`}</span>
              <strong>{item.label || "-"}</strong>
              <em>{item.top5_to_top3_hit_rate_text || "-"}</em>
            </li>
          ))}
        </ol>
      ) : (
        <PanelEmpty>この条件に一致する履歴データはまだありません。</PanelEmpty>
      )}
    </section>
  );
}

function PredictorCompareRow({ card }) {
  if (card?.is_placeholder) {
    return (
      <article className="race-detail-compare-row race-detail-compare-row--placeholder">
        <div className="race-detail-compare-row__top">
          <strong>{card?.label || "-"}</strong>
          <span className="race-detail-compare-status" aria-label="計算中">
            <span />
            <span />
            <span />
          </span>
        </div>
        <div className="race-detail-compare-loading" aria-hidden="true">
          <span className="race-detail-compare-loading__bar" />
        </div>
        <p className="race-detail-compare-placeholder">{card?.placeholder_text || "完全公開準備中"}</p>
      </article>
    );
  }
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

export default function RaceDetailPage({ race, search = "", appShell = false }) {
  const predictorCompareCards = Array.isArray(race?.predictor_compare_cards)
    ? race.predictor_compare_cards.filter(Boolean)
    : [];
  const variant = String(race?.display_variant || "").trim();
  const isMorningPreview = variant === "morning_preview";
  const morningCompareCards = useMemo(
    () => buildMorningCompareCards(race?.predictor_top5, race?.scheduled_off_time),
    [race?.predictor_top5, race?.scheduled_off_time],
  );
  const compareCards = predictorCompareCards.length ? predictorCompareCards : morningCompareCards;
  const activeCompareCount = predictorCompareCards.length
    ? predictorCompareCards.length
    : morningCompareCards.filter((item) => !item?.is_placeholder).length;
  const status = race?.display_status || {};
  const isSettled = variant === "settled";
  const resultText = isSettled
    ? String(race?.display_body?.result_text || "結果はまだありません")
    : "結果は確定後に表示されます";
  const resultEntries = parseResultEntries(resultText);
  const signalRows = useMemo(
    () =>
      predictorCompareCards.length
        ? buildHorseSignalRows(predictorCompareCards)
        : buildMorningIndexRows(race?.top5),
    [predictorCompareCards, race?.top5],
  );
  const confidenceMeta = useMemo(
    () => buildConfidenceMeta(signalRows, activeCompareCount),
    [activeCompareCount, signalRows],
  );
  const badges = formatRaceBadges(race).filter(
    (item) => !["良", "稍重", "重", "不良"].includes(String(item || "").trim()),
  );
  const backHref = buildBackHref(search);
  const detailTitle = String(
    race?.display_header?.detail_title || race?.display_header?.title || "-",
  ).trim() || "-";
  const detailConfidenceText =
    Number.isFinite(Number(race?.confidence_score))
      ? `${Math.round(Number(race.confidence_score) * 100)}%`
      : confidenceMeta.percentText;
  const conditionRanking = race?.condition_predictor_ranking || {};

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
        </div>

        <div className="race-detail-hero__meta">
          <DetailSummary label="公開状態" value={status?.label || "公開中"} accent />
          <DetailSummary label="自信度" value={detailConfidenceText} />
          <DetailAppSummary />
        </div>
      </div>

      <div className="race-detail-layout">
        <AiIndexPanel signalRows={signalRows} />

        <div className="race-detail-side-stack">
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
        </div>
      </div>

      <div className="race-detail-layout race-detail-layout--compare">
        <section
          className="race-detail-panel race-detail-panel--compare"
          id="race-detail-compare"
        >
          <div className="race-detail-panel__head">
            <div>
              <h2>
                {isMorningPreview && !predictorCompareCards.length
                  ? "定量モデル比較"
                  : "6モデルの本命比較"}
              </h2>
            </div>
          </div>
          <div className="race-detail-compare-list">
            {compareCards.length ? (
              compareCards.map((card) => (
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

        <div className="race-detail-side-stack">
          <ConditionPredictorRankingPanel ranking={conditionRanking} />
        </div>
      </div>
    </section>
  );
}
