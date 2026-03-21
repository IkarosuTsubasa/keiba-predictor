import React from "react";

function parsePercent(value) {
  const text = String(value || "").trim();
  const matched = text.match(/-?\d+(?:\.\d+)?/);
  if (!matched) return null;
  const number = Number(matched[0]);
  return Number.isFinite(number) ? number : null;
}

function roiToneStyle(value) {
  const percent = parsePercent(value);
  if (!Number.isFinite(percent)) return null;

  const clamped = Math.max(0, Math.min(200, percent));
  const hue = (clamped / 200) * 120;
  const strong = Math.abs(clamped - 100) / 100;
  const borderAlpha = 0.18 + strong * 0.2;
  const bgAlpha = 0.06 + strong * 0.12;
  const textLightness = 76 - strong * 8;

  return {
    borderColor: `hsla(${hue}, 72%, 58%, ${borderAlpha})`,
    background: `hsla(${hue}, 72%, 58%, ${bgAlpha})`,
    color: `hsl(${hue}, 36%, ${textLightness}%)`,
  };
}

export default function ModelMetaBadge({
  label,
  value,
  tone = "neutral",
  dynamicRoi = false,
}) {
  const roiStyle = dynamicRoi ? roiToneStyle(value) : null;

  return (
    <span
      className={`model-meta-badge model-meta-badge--${tone}`}
      style={roiStyle || undefined}
    >
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </span>
  );
}
