import React from "react";

function safeText(value) {
  return String(value || "").trim();
}

function formatConfidence(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${Math.round(number * 100)}%`;
}

const MARK_LABELS = ["◎", "○", "▲", "△", "☆"];

function resolveDecision(race) {
  const explicitDecision = safeText(race?.agent_prediction?.strategy?.bet_decision);
  if (explicitDecision === "BET") return { label: "高評価", tone: "bet" };
  if (explicitDecision === "SKIP") return { label: "見送り", tone: "skip" };

  const confidence = Number(race?.confidence_score);
  if (confidence >= 0.62) return { label: "高評価", tone: "bet" };
  if (confidence >= 0.45) return { label: "要確認", tone: "watch" };
  if (Number.isFinite(confidence)) return { label: "見送り", tone: "skip" };
  return { label: "確認待ち", tone: "watch" };
}

function mainHorseText(race) {
  const top5 = Array.isArray(race?.top5) ? race.top5 : [];
  const main = top5[0] || null;
  if (!main) return "-";
  return `◎ ${main.horse_no || "-"} ${main.horse_name || ""}`.trim();
}

function topMarksText(race) {
  const top5 = Array.isArray(race?.top5) ? race.top5 : [];
  const marks = top5
    .filter(Boolean)
    .slice(0, MARK_LABELS.length)
    .map((item, index) => {
      const horseNo = safeText(item?.horse_no);
      if (!horseNo) return "";
      return `${MARK_LABELS[index] || ""}${horseNo}`;
    })
    .filter(Boolean);
  return marks.length ? marks.join(" / ") : "-";
}

function resultText(race) {
  return safeText(race?.display_body?.result_text) || "結果は確定後に表示されます";
}

function pickFocusRace(races) {
  const available = (Array.isArray(races) ? races : []).filter(
    (race) => safeText(race?.display_variant) !== "placeholder",
  );
  return available.find((race) => resolveDecision(race).tone === "bet") || available[0] || null;
}

function MetricCard({ label, value, note = "", accent = false }) {
  return (
    <article className={`dashboard-metric-card${accent ? " dashboard-metric-card--accent" : ""}`}>
      <span>{label}</span>
      <strong>{value || "-"}</strong>
      {note ? <small>{note}</small> : null}
    </article>
  );
}

export function DashboardInsightPanel({ data, races }) {
  const periods = data?.history?.agent_prediction?.periods || {};
  const activePeriod = periods.days_30 || periods.all_time || {};
  const visibleRaces = (Array.isArray(races) ? races : []).filter(
    (race) => safeText(race?.display_variant) !== "placeholder",
  );
  const highEvalCount = visibleRaces.filter((race) => resolveDecision(race).tone === "bet").length;
  const skipCount = visibleRaces.filter((race) => resolveDecision(race).tone === "skip").length;
  const settledCount = Number(data?.totals?.settled_count || activePeriod?.settled_races || 0);
  const predictedCount = Number(activePeriod?.predicted_races || visibleRaces.length || 0);

  return (
    <aside className="dashboard-insight-panel" aria-label="検証成績と注意事項">
      <section className="dashboard-insight-card dashboard-insight-card--metrics">
        <div className="dashboard-insight-card__head">
          <div>
            <span>検証成績</span>
            <strong>直近30日間</strong>
          </div>
          <a href="/keiba/history">詳細</a>
        </div>
        <div className="dashboard-metric-grid">
          <MetricCard
            label="本命複勝圏率"
            value={activePeriod?.main_top3_rate_text || "-"}
            note={`${activePeriod?.main_top3_count || 0}/${activePeriod?.settled_races || 0}レース`}
            accent
          />
          <MetricCard
            label="本命1着率"
            value={activePeriod?.main_win_rate_text || "-"}
            note={`${activePeriod?.main_win_count || 0}/${activePeriod?.settled_races || 0}レース`}
          />
          <MetricCard
            label="上位5頭カバー"
            value={activePeriod?.top5_cover_rate_text || "-"}
            note={`${activePeriod?.top5_cover_hits || 0}/${(activePeriod?.settled_races || 0) * 3}頭`}
          />
          <MetricCard
            label="予測レース"
            value={`${predictedCount}レース`}
            note={`結果確定 ${settledCount}レース`}
          />
        </div>
      </section>

      <section className="dashboard-insight-card">
        <div className="dashboard-insight-card__head">
          <div>
            <span>リスク・注意事項</span>
            <strong>確認ポイント</strong>
          </div>
        </div>
        <ul className="dashboard-risk-list">
          <li>見送り判断のレースは印と評価理由を確認してください。</li>
          <li>信頼度が低いレースは結果確認を優先してください。</li>
          <li>直前オッズの変動に注意してください。</li>
        </ul>
      </section>

      <section className="dashboard-insight-card">
        <div className="dashboard-insight-card__head">
          <div>
            <span>本日の傾向メモ</span>
            <strong>AIメモ</strong>
          </div>
        </div>
        <p className="dashboard-ai-note">
          高評価 {highEvalCount}レース、見送り {skipCount}レース。予測印と結果を同じ行で確認できます。
        </p>
      </section>
    </aside>
  );
}

export function DashboardRaceMemo({ races }) {
  const focusRace = pickFocusRace(races);
  if (!focusRace) return null;

  const decision = resolveDecision(focusRace);
  const title = safeText(focusRace?.display_header?.title) || "注目レース";
  const subtitle = safeText(focusRace?.display_header?.subtitle);
  const confidence = formatConfidence(focusRace?.confidence_score);
  const result = resultText(focusRace);
  const marks = topMarksText(focusRace);

  return (
    <section className={`dashboard-race-memo dashboard-race-memo--${decision.tone}`}>
      <div className="dashboard-race-memo__bar">
        <strong>{subtitle ? `${title} ${subtitle}` : title}</strong>
        <span>{decision.label}</span>
        <span>信頼度 {confidence}</span>
      </div>
      <div className="dashboard-race-memo__grid">
        <article>
          <span>本命馬</span>
          <strong>{mainHorseText(focusRace)}</strong>
        </article>
        <article>
          <span>AIの判断理由</span>
          <p>信頼度と上位馬評価をもとに、詳細ページで予測印を確認できます。</p>
        </article>
        <article>
          <span>上位印</span>
          <strong>{marks}</strong>
        </article>
        <article>
          <span>結果</span>
          <p>{result}</p>
        </article>
      </div>
    </section>
  );
}
