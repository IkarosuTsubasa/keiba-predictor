import React from "react";

const MODEL_ACCENTS = {
  openai: "openai",
  gemini: "gemini",
  siliconflow: "siliconflow",
  grok: "grok",
};

export default function ModelBadge({ engine, label, subtle = false }) {
  return (
    <span className={`model-badge model-badge--${MODEL_ACCENTS[engine] || "neutral"}${subtle ? " model-badge--subtle" : ""}`}>
      {label}
    </span>
  );
}
