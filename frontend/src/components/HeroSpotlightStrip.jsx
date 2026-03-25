import React, { useEffect, useMemo, useState } from "react";

function toNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function parsePercent(value) {
  const text = String(value || "").trim();
  const matched = text.match(/-?\d+(?:\.\d+)?/);
  if (!matched) return null;
  const number = Number(matched[0]);
  return Number.isFinite(number) ? number : null;
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value - Math.round(value)) < 0.05) {
    return `${Math.round(value)}%`;
  }
  return `${value.toFixed(1)}%`;
}

function formatYen(value) {
  const number = toNumber(value, 0);
  return `${new Intl.NumberFormat("ja-JP").format(number)}円`;
}

function useCountUp(target, duration = 900) {
  const [displayValue, setDisplayValue] = useState(target);

  useEffect(() => {
    if (!Number.isFinite(target)) {
      setDisplayValue(target);
      return undefined;
    }

    let frameId = 0;
    let startTime = 0;

    const tick = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min(1, (timestamp - startTime) / duration);
      const eased = 1 - (1 - progress) ** 3;
      setDisplayValue(target * eased);
      if (progress < 1) {
        frameId = window.requestAnimationFrame(tick);
      }
    };

    setDisplayValue(0);
    frameId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frameId);
  }, [target, duration]);

  return displayValue;
}

function AnimatedMetric({ value, formatter }) {
  const animated = useCountUp(value);
  return formatter(animated);
}

function buildRoiSpotlight(data) {
  const rows = Array.isArray(data?.summary_cards) ? data.summary_cards : [];
  const available = rows.filter((item) => toNumber(item?.races, 0) > 0);
  if (!available.length) return null;

  const roiLeader = [...available].sort((left, right) => {
    const rightPercent =
      parsePercent(right?.roi_text) ?? toNumber(right?.roi_percent, -1);
    const leftPercent =
      parsePercent(left?.roi_text) ?? toNumber(left?.roi_percent, -1);
    if (rightPercent !== leftPercent) return rightPercent - leftPercent;
    return toNumber(right?.profit_yen, 0) - toNumber(left?.profit_yen, 0);
  })[0];

  return {
    key: "roi",
    eyebrow: "最高回収率",
    badge: "ROI",
    title: String(roiLeader?.label || roiLeader?.engine || "-"),
    metricValue:
      parsePercent(roiLeader?.roi_text) ?? toNumber(roiLeader?.roi_percent, 0),
    metricFormatter: (value) => formatPercent(value),
    detail: `損益 ${formatYen(roiLeader?.profit_yen)}`,
    caption: String(roiLeader?.roi_text || "-"),
  };
}

export default function HeroSpotlightStrip({ data }) {
  const item = useMemo(() => buildRoiSpotlight(data), [data]);

  if (!item) {
    return null;
  }

  return (
    <section
      className="hero-spotlight-strip hero-spotlight-strip--single"
      aria-label="本日の注目指標"
    >
      <article className="hero-spotlight-card hero-spotlight-card--roi">
        <div className="hero-spotlight-card__top">
          <span className="hero-spotlight-card__eyebrow">{item.eyebrow}</span>
          <span className="hero-spotlight-card__badge">{item.badge}</span>
        </div>

        <div className="hero-spotlight-card__body">
          <strong className="hero-spotlight-card__title">{item.title}</strong>
          <div className="hero-spotlight-card__metric">
            <strong>
              <AnimatedMetric value={item.metricValue} formatter={item.metricFormatter} />
            </strong>
          </div>
        </div>

        <div className="hero-spotlight-card__meta">
          <span>{item.detail}</span>
          <span>{item.caption}</span>
        </div>
      </article>
    </section>
  );
}
