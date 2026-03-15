import React, { useMemo } from "react";

function parseLines(text) {
  return String(text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

export default function BetPreviewList({ text, maxItems = 3, compact = false }) {
  const lines = useMemo(() => parseLines(text), [text]);
  const visible = compact ? lines.slice(0, maxItems) : lines;

  if (!visible.length) {
    return <div className="bet-preview-list bet-preview-list--empty">買い目なし</div>;
  }

  return (
    <ul className={`bet-preview-list${compact ? " bet-preview-list--compact" : ""}`}>
      {visible.map((line) => (
        <li key={line}>{line}</li>
      ))}
    </ul>
  );
}
