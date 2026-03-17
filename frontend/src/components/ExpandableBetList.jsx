import React, { useMemo, useState } from "react";

function parseLines(text) {
  return String(text || "")
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

export default function ExpandableBetList({ text }) {
  const [expanded, setExpanded] = useState(false);
  const lines = useMemo(() => parseLines(text), [text]);

  if (!lines.length) {
    return <div className="empty-chip">買い目なし</div>;
  }

  const visibleLines = expanded ? lines : lines.slice(0, 3);

  return (
    <div className="expandable-bet-list">
      <ul className="bet-list">
        {visibleLines.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
      {lines.length > 3 ? (
        <button type="button" className="expandable-bet-list__toggle" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "買い目を閉じる" : `残り ${lines.length - 3} 件を表示`}
        </button>
      ) : null}
    </div>
  );
}
