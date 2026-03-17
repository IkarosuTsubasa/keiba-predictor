import React from "react";

export default function ResultBadge({ label, tone = "pending" }) {
  return <span className={`result-badge result-badge--${tone}`}>{label}</span>;
}
