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

function buildSpotlights(data) {
  const rows = Array.isArray(data?.summary_cards) ? data.summary_cards : [];
  const available = rows.filter((item) => toNumber(item?.races, 0) > 0);
  if (!available.length) return [];

  const hitLeader = [...available].sort((left, right) => {
    const hitGap = toNumber(right?.hit_races, 0) - toNumber(left?.hit_races, 0);
    if (hitGap !== 0) return hitGap;
    const leftRaceCount = toNumber(left?.settled_races || left?.races, 0);
    const rightRaceCount = toNumber(right?.settled_races || right?.races, 0);
    const leftRate = leftRaceCount > 0 ? toNumber(left?.hit_races, 0) / leftRaceCount : 0;
    const rightRate = rightRaceCount > 0 ? toNumber(right?.hit_races, 0) / rightRaceCount : 0;
    return rightRate - leftRate;
  })[0];

  const roiLeader = [...available].sort((left, right) => {
    const rightPercent = parsePercent(right?.roi_text) ?? toNumber(right?.roi_percent, -1);
    const leftPercent = parsePercent(left?.roi_text) ?? toNumber(left?.roi_percent, -1);
    if (rightPercent !== leftPercent) return rightPercent - leftPercent;
    return toNumber(right?.profit_yen, 0) - toNumber(left?.profit_yen, 0);
  })[0];

  const hitRaceCount = toNumber(hitLeader?.settled_races || hitLeader?.races, 0);
  const hitValue = toNumber(hitLeader?.hit_races, 0);
  const hitRate = hitRaceCount > 0 ? (hitValue / hitRaceCount) * 100 : 0;
  const roiValue = parsePercent(roiLeader?.roi_text) ?? toNumber(roiLeader?.roi_percent, 0);

  return [
    {
      key: "hits",
      eyebrow: "本日の的中",
      badge: "的中トップ",
      title: String(hitLeader?.label || hitLeader?.engine || "-"),
      metricValue: hitValue,
      metricFormatter: (value) => `${Math.round(value)}`,
      metricSuffix: ` / ${hitRaceCount || 0}`,
      detail: `的中率 ${formatPercent(hitRate)}`,
      caption: `${hitRaceCount || 0}レース中 ${hitValue}レース的中`,
    },
    {
      key: "roi",
      eyebrow: "最高回収",
      badge: "回収トップ",
      title: String(roiLeader?.label || roiLeader?.engine || "-"),
      metricValue: roiValue,
      metricFormatter: (value) => formatPercent(value),
      metricSuffix: "",
      detail: `収益 ${formatYen(roiLeader?.profit_yen)}`,
      caption: String(roiLeader?.roi_text || "-"),
    },
  ];
}

export default function HeroSpotlightStrip({ data }) {
  const items = useMemo(() => buildSpotlights(data), [data]);

  if (!items.length) {
    return null;
  }

  return (
    <section className="hero-spotlight-strip" aria-label="本日のモデル要約">
      {items.map((item, index) => (
        <article
          key={item.key}
          className="hero-spotlight-card"
          style={{ animationDelay: `${0.18 + index * 0.08}s` }}
        >
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
              {item.metricSuffix ? <span>{item.metricSuffix}</span> : null}
            </div>
          </div>

          <div className="hero-spotlight-card__meta">
            <span>{item.detail}</span>
            <span>{item.caption}</span>
          </div>
        </article>
      ))}
    </section>
  );
}
