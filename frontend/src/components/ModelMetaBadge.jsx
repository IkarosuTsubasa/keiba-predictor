import React from "react";

export default function ModelMetaBadge({ label, value, tone = "neutral" }) {
  return (
    <span className={`model-meta-badge model-meta-badge--${tone}`}>
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </span>
  );
}
