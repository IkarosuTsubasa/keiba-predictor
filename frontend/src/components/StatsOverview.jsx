import React from "react";
import RecoveryRateBadge from "./RecoveryRateBadge";
import RoiRankingChart from "./RoiRankingChart";
import PerformanceTrendChart from "./PerformanceTrendChart";

function StatBand({ title, kicker, summary }) {
  const cards = summary?.cards || [];
  const totals = summary?.totals || {};
  if (!cards.length && !totals?.roi_text) return null;

  return (
    <section className="stats-band">
      <div className="section-header">
        <div>
          <span className="section-kicker">{kicker}</span>
          <h2>{title}</h2>
        </div>
      </div>
      <div className="stats-band__grid">
        <RecoveryRateBadge label="全体" roiText={totals?.roi_text || "-"} emphasis />
        {cards.map((item) => (
          <RecoveryRateBadge key={`${title}-${item.engine}`} label={item.label} roiText={item.roi_text} />
        ))}
      </div>
    </section>
  );
}

function HitRatePanel({ summaryCards }) {
  return (
    <section className="chart-card">
      <div className="chart-card__head">
        <span className="section-kicker">Hit Snapshot</span>
        <h3>命中率 / 回収率摘要</h3>
      </div>
      <div className="hit-panel">
        {(summaryCards || []).map((item) => {
          const races = Number(item?.races || 0);
          const hits = Number(item?.hit_races || 0);
          const hitRate = races > 0 ? `${((hits / races) * 100).toFixed(1)}%` : "-";
          return (
            <div key={item.engine} className="hit-panel__card">
              <span className="hit-panel__label">{item.label}</span>
              <strong>{hitRate}</strong>
              <span>回収率 {item?.roi_text || "-"}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default function StatsOverview({ data }) {
  const currentSummary = {
    cards: data?.summary_cards || [],
    totals: data?.totals || {},
  };

  return (
    <section className="stats-overview">
      <StatBand title="全期間回収率" kicker="Lifetime Recovery" summary={data?.all_time_roi || {}} />
      <StatBand title="当日回収率" kicker="Daily Recovery" summary={currentSummary} />
      <div className="analytics-grid">
        <RoiRankingChart items={data?.all_time_roi?.cards || []} />
        <PerformanceTrendChart trend={data?.trend || {}} />
        <HitRatePanel summaryCards={data?.summary_cards || []} />
      </div>
    </section>
  );
}
