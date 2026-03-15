import React, { useMemo } from "react";
import ModelBadge from "./ModelBadge";

function parseRoi(roiText) {
  const value = Number.parseFloat(String(roiText || "").replace("%", ""));
  return Number.isFinite(value) ? value : 0;
}

export default function RoiRankingChart({ items }) {
  const ranked = useMemo(() => {
    const list = [...(items || [])]
      .map((item) => ({ ...item, roiValue: parseRoi(item?.roi_text) }))
      .sort((a, b) => b.roiValue - a.roiValue);
    const maxValue = Math.max(110, ...list.map((item) => item.roiValue || 0));
    return { list, maxValue };
  }, [items]);

  return (
    <section className="chart-card">
      <div className="chart-card__head">
        <span className="section-kicker">ROI Ranking</span>
        <h3>AI ROI Ranking</h3>
      </div>
      <div className="ranking-chart">
        {ranked.list.map((item, index) => (
          <div key={item.engine} className="ranking-chart__row">
            <div className="ranking-chart__meta">
              <span className="ranking-chart__rank">#{index + 1}</span>
              <ModelBadge engine={item.engine} label={item.label} subtle />
            </div>
            <div className="ranking-chart__bar">
              <div
                className={`ranking-chart__fill ranking-chart__fill--${item.engine}`}
                style={{ width: `${Math.max(8, (item.roiValue / ranked.maxValue) * 100)}%` }}
              />
            </div>
            <strong className="ranking-chart__value">{item.roi_text || "-"}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}
