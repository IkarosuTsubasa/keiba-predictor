import React from "react";

function parseRoi(roiText) {
  const value = Number.parseFloat(String(roiText || "").replace("%", ""));
  return Number.isFinite(value) ? value : null;
}

function renderTrendBars(trend) {
  const labels = trend?.labels || [];
  const series = trend?.series || [];
  if (!labels.length || !series.length) {
    return <div className="secondary-stats-panel__empty">トレンド集計中</div>;
  }

  const latest = series
    .map((item) => {
      const point = (item.points || [])[item.points.length - 1] || {};
      return {
        engine: item.engine,
        label: item.label,
        roiText: point.roi_text || "-",
        roiValue: typeof point.roi_value === "number" ? point.roi_value : 0,
      };
    })
    .sort((a, b) => b.roiValue - a.roiValue);

  return (
    <div className="secondary-stats-panel__bars">
      {latest.map((item) => (
        <div key={item.engine} className="secondary-stats-panel__bar-row">
          <span>{item.label}</span>
          <div className="secondary-stats-panel__bar-track">
            <div
              className={`secondary-stats-panel__bar-fill secondary-stats-panel__bar-fill--${item.engine}`}
              style={{ width: `${Math.max(10, Math.min(100, item.roiValue || 0))}%` }}
            />
          </div>
          <strong>{item.roiText}</strong>
        </div>
      ))}
    </div>
  );
}

export default function SecondaryStatsPanel({ data }) {
  const dailyCards = data?.summary_cards || [];
  const allTimeCards = data?.all_time_roi?.cards || [];
  const topAllTime = [...allTimeCards].sort((a, b) => (parseRoi(b.roi_text) || 0) - (parseRoi(a.roi_text) || 0));

  return (
    <section className="secondary-stats-panel">
      <div className="secondary-stats-panel__header">
        <div>
          <span className="secondary-stats-panel__kicker">Secondary Insights</span>
          <h2>モデル参考データ</h2>
        </div>
        <p>回収率や最近の傾向は参考情報として配置し、主役は上部のレース一覧に置いています。</p>
      </div>

      <div className="secondary-stats-panel__grid">
        <section className="secondary-stats-panel__card">
          <h3>通算回収率</h3>
          <ul className="secondary-stats-panel__list">
            {topAllTime.map((item) => (
              <li key={item.engine}>
                <span>{item.label}</span>
                <strong>{item.roi_text || "-"}</strong>
              </li>
            ))}
          </ul>
        </section>

        <section className="secondary-stats-panel__card">
          <h3>当日回収率</h3>
          <ul className="secondary-stats-panel__list">
            {dailyCards.map((item) => (
              <li key={item.engine}>
                <span>{item.label}</span>
                <strong>{item.roi_text || "-"}</strong>
              </li>
            ))}
          </ul>
        </section>

        <section className="secondary-stats-panel__card secondary-stats-panel__card--wide">
          <h3>最近の推移</h3>
          {renderTrendBars(data?.trend || {})}
        </section>
      </div>
    </section>
  );
}
