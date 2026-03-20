import React from "react";

function parseResultEntries(text) {
  const source = String(text || "").trim();
  if (!source || source.includes("未") || source.includes("待ち")) {
    return [];
  }
  return source
    .split("/")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 3)
    .map((item, index) => {
      const match = item.match(/^([1-3])着\s*(.+)$/);
      const rank = Number(match?.[1] || index + 1);
      const body = String(match?.[2] || item).trim();
      return { key: `${rank}-${body}`, rank, body };
    })
    .filter((item) => item.body);
}

function resolveStatus(race) {
  const status = race?.display_status || {};
  return {
    label: String(status.label || "").trim() || "予測中",
    tone: String(status.tone || "").trim() || "open",
  };
}

export default function RaceCardHeader({ race, actions = null }) {
  const status = resolveStatus(race);
  const title = String(race?.display_header?.title || "-");
  const badges = Array.isArray(race?.display_header?.badges)
    ? race.display_header.badges.filter(Boolean)
    : [];
  const variant = String(race?.display_variant || "").trim();
  const isPlaceholder = variant === "placeholder";
  const resultText = String(race?.display_body?.result_text || "結果未確定");
  const resultEntries = parseResultEntries(resultText);

  return (
    <header className="race-card-header">
      <div className="race-card-header__main">
        <div>
          <h3>{title}</h3>
          {badges.length ? (
            <div className="race-card-header__badges">
              {badges.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          ) : null}
        </div>
        <span
          className={`race-card-header__status race-card-header__status--${status.tone}`}
        >
          {status.label}
        </span>
      </div>

      {!isPlaceholder ? (
        <div className="race-card-header__result-row">
          <div className="race-card-header__result">
            <span className="race-card-header__result-label">結果</span>
            <div className="race-card-header__result-body">
              {resultEntries.length ? (
                <ul className="race-card-header__result-list">
                  {resultEntries.map((entry) => (
                    <li key={entry.key}>
                      <span
                        className="race-card-header__result-medal"
                        aria-hidden="true"
                      >
                        {entry.rank}着
                      </span>
                      <span>{entry.body}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>{resultText}</p>
              )}
            </div>
          </div>
          {actions ? (
            <div className="race-card-header__actions">{actions}</div>
          ) : null}
        </div>
      ) : null}
    </header>
  );
}
