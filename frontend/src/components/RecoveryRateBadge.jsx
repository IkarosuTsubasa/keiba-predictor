import React from "react";

function parseRoiValue(roiText) {
  const value = Number.parseFloat(String(roiText || "").replace("%", ""));
  return Number.isFinite(value) ? value : null;
}

function resolveTone(roiValue) {
  if (roiValue === null) return "muted";
  if (roiValue >= 110) return "excellent";
  if (roiValue >= 100) return "positive";
  if (roiValue >= 90) return "neutral";
  return "weak";
}

function resolveLabel(roiValue) {
  if (roiValue === null) return "集計中";
  if (roiValue >= 110) return "優秀";
  if (roiValue >= 100) return "正收益";
  return "中性";
}

export default function RecoveryRateBadge({ label, roiText, emphasis = false, compact = false }) {
  const roiValue = parseRoiValue(roiText);
  const tone = resolveTone(roiValue);

  return (
    <div className={`recovery-badge recovery-badge--${tone}${emphasis ? " recovery-badge--emphasis" : ""}${compact ? " recovery-badge--compact" : ""}`}>
      <span className="recovery-badge__label">{label}</span>
      <strong className="recovery-badge__value">{roiText || "-"}</strong>
      <span className="recovery-badge__meta">回収率 {resolveLabel(roiValue)}</span>
    </div>
  );
}
