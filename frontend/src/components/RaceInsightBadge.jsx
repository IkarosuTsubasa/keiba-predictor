import React from "react";

function parseMainHorse(text) {
  const match = String(text || "").match(/◎\s*([0-9]+)/);
  return match ? match[1] : "";
}

function buildInsight(cards) {
  const mains = (cards || []).map((card) => parseMainHorse(card?.marks_text)).filter(Boolean);
  if (!mains.length) return "予想公開中";

  const counts = new Map();
  mains.forEach((horse) => counts.set(horse, (counts.get(horse) || 0) + 1));
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  const leaders = [...counts.entries()].filter((item) => item[1] === top[1]);

  if (top && top[1] >= 3 && leaders.length === 1) {
    return `本命集中 ${top[0]}号`;
  }
  if (counts.size === mains.length) {
    return "本命分散";
  }
  return `中心視 ${top[0]}号`;
}

export default function RaceInsightBadge({ cards }) {
  return <span className="race-insight-badge">{buildInsight(cards)}</span>;
}
