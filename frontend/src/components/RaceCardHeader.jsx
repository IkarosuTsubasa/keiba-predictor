import React from "react";

function formatOffTime(text) {
  const source = String(text || "").trim();
  if (!source) return "";
  const match = source.match(/T(\d{2}):(\d{2})/);
  if (match) return `${match[1]}:${match[2]}`;
  return source.slice(0, 5);
}

function resolveStatus(race) {
  const actual = String(race?.actual_text || "");
  if (actual && !actual.includes("未")) {
    return { label: "結果確定", tone: "settled" };
  }
  return { label: "予想公開中", tone: "open" };
}

export default function RaceCardHeader({ race }) {
  const status = resolveStatus(race);
  const badges = [formatOffTime(race?.scheduled_off_time), race?.distance_label].filter(Boolean);
  const resultText = race?.actual_text || "結果未確定";

  return (
    <header className="race-card-header">
      <div className="race-card-header__main">
        <div>
          <h3>{race?.race_title || "-"}</h3>
          {badges.length ? (
            <div className="race-card-header__badges">
              {badges.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          ) : null}
        </div>
        <span className={`race-card-header__status race-card-header__status--${status.tone}`}>{status.label}</span>
      </div>

      <div className="race-card-header__result">
        <span className="race-card-header__result-label">着順</span>
        <p>{resultText}</p>
      </div>
    </header>
  );
}
