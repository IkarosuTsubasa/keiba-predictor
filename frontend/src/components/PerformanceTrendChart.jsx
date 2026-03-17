import React, { useMemo } from "react";
import ModelBadge from "./ModelBadge";

const SERIES_COLORS = {
  openai: "#f2b35d",
  gemini: "#7ed3a8",
  siliconflow: "#8db2ff",
  grok: "#ff7f7f",
};

function buildLine(points, width, height, minValue, maxValue) {
  const usable = points.filter((point) => point.value !== null);
  if (!usable.length) return "";
  return usable
    .map((point, index) => {
      const x = usable.length === 1 ? width / 2 : (index / (usable.length - 1)) * width;
      const ratio = maxValue === minValue ? 0.5 : (point.value - minValue) / (maxValue - minValue);
      const y = height - ratio * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

export default function PerformanceTrendChart({ trend }) {
  const chart = useMemo(() => {
    const labels = trend?.labels || [];
    const allValues = [];
    const series = (trend?.series || []).map((item) => {
      const points = (item?.points || []).map((point) => {
        const value = typeof point?.roi_value === "number" ? point.roi_value : null;
        if (value !== null) allValues.push(value);
        return { date: point?.date || "", value };
      });
      return { ...item, points };
    });
    return {
      labels,
      series,
      minValue: Math.min(80, ...(allValues.length ? allValues : [80])),
      maxValue: Math.max(130, ...(allValues.length ? allValues : [130])),
    };
  }, [trend]);

  return (
    <section className="chart-card">
      <div className="chart-card__head">
        <span className="section-kicker">Performance Trend</span>
        <h3>最近の回収率トレンド</h3>
      </div>
      <div className="trend-chart">
        <svg viewBox="0 0 520 220" className="trend-chart__svg" preserveAspectRatio="none">
          <defs>
            <linearGradient id="trendGrid" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="rgba(255,255,255,0.12)" />
              <stop offset="100%" stopColor="rgba(255,255,255,0.02)" />
            </linearGradient>
          </defs>
          {[0, 1, 2, 3].map((line) => {
            const y = 20 + line * 50;
            return <line key={line} x1="0" x2="520" y1={y} y2={y} className="trend-chart__grid-line" />;
          })}
          {(chart.series || []).map((item) => (
            <path
              key={item.engine}
              d={buildLine(item.points, 520, 180, chart.minValue, chart.maxValue)}
              className="trend-chart__path"
              stroke={SERIES_COLORS[item.engine] || "#caa96d"}
            />
          ))}
        </svg>
        <div className="trend-chart__legend">
          {(chart.series || []).map((item) => (
            <div key={item.engine} className="trend-chart__legend-item">
              <ModelBadge engine={item.engine} label={item.label} subtle />
              <span>{item.points[item.points.length - 1]?.value ? `${item.points[item.points.length - 1].value.toFixed(1)}%` : "-"}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
