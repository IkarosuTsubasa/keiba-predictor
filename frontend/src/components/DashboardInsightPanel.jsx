import { resolvePublicDecision } from "../lib/confidencePolicy";

function safeText(value) {
  return String(value || "").trim();
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
  const highEvalCount = visibleRaces.filter((race) => resolvePublicDecision(race).tone === "bet").length;
  const watchCount = visibleRaces.filter((race) => {
    const decision = resolvePublicDecision(race);
    return decision.tone === "watch" && decision.label === "注目";
  }).length;
  const skipCount = visibleRaces.filter((race) => resolvePublicDecision(race).tone === "skip").length;
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
          <li>見送り判断のレースは無理に追わず、評価理由だけ確認してください。</li>
          <li>信頼度が低いレースは購入より検証を優先してください。</li>
          <li>直前オッズと馬場変化に注意してください。</li>
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
          高評価 {highEvalCount}レース、注目 {watchCount}レース、見送り {skipCount}レース。高評価以外は直前オッズまで確認してください。
        </p>
      </section>
    </aside>
  );
}
