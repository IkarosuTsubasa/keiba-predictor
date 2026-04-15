import React, { useMemo } from "react";
import { buildTargetDateContext } from "../lib/homepage";

function toPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${Math.round(number * 100)}%`;
}

function buildSpotlightItems(data) {
  const cards = Array.isArray(data?.daily_predictor?.cards) ? data.daily_predictor.cards : [];
  const leader = data?.daily_predictor?.top5to3_leader || null;
  if (!cards.length) {
    return [];
  }

  const top1Leader = [...cards].sort(
    (left, right) => Number(right?.top1_hit_rate || 0) - Number(left?.top1_hit_rate || 0),
  )[0] || null;

  return [
    {
      key: "coverage",
      eyebrow: "対象日リーダー",
      badge: "TOP5",
      title: String(leader?.label || "-"),
      metric: toPercent(leader?.top5_to_top3_hit_rate),
      detail: `対象 ${leader?.samples || 0}レース`,
      caption: String(leader?.top5_to_top3_hit_rate_text || "-"),
    },
    {
      key: "top1",
      eyebrow: "本命精度",
      badge: "TOP1",
      title: String(top1Leader?.label || "-"),
      metric: toPercent(top1Leader?.top1_hit_rate),
      detail: `掲載 ${cards.length}モデル`,
      caption: String(top1Leader?.top1_hit_rate_text || "-"),
    },
  ];
}

export default function HeroSpotlightStrip({ data }) {
  const items = useMemo(() => buildSpotlightItems(data), [data]);
  const targetDateContext = useMemo(() => buildTargetDateContext(data), [data]);

  if (!items.length) {
    return null;
  }

  return (
    <section
      className={`hero-spotlight-strip${items.length === 1 ? " hero-spotlight-strip--single" : ""}`}
      aria-label={`${targetDateContext.targetDateLabel}の定量モデル注目指標`}
    >
      {items.map((item) => (
        <article
          key={item.key}
          className={`hero-spotlight-card${item.key === "coverage" ? " hero-spotlight-card--roi" : ""}`}
        >
          <div className="hero-spotlight-card__top">
            <span className="hero-spotlight-card__eyebrow">{item.eyebrow}</span>
            <span className="hero-spotlight-card__badge">{item.badge}</span>
          </div>

          <div className="hero-spotlight-card__body">
            <strong className="hero-spotlight-card__title">{item.title}</strong>
            <div className="hero-spotlight-card__metric">
              <strong>{item.metric}</strong>
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
