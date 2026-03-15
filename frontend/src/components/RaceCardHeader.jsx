import React from "react";

const RESULT_MEDALS = ["🥇", "🥈", "🥉"];

function formatOffTime(text) {
  const source = String(text || "").trim();
  if (!source) return "";
  const match = source.match(/T(\d{2}):(\d{2})/);
  if (match) return `${match[1]}:${match[2]}`;
  return source.slice(0, 5);
}

function parseResultEntries(text) {
  const source = String(text || "").trim();
  if (!source || source.includes("未")) return [];

  return source
    .split("/")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 3)
    .map((item, index) => {
      const match = item.match(/^([1-3])着\s*(.+)$/);
      const rank = Number(match?.[1] || index + 1);
      const body = String(match?.[2] || item).trim();
      return {
        key: `${rank}-${body}`,
        medal: RESULT_MEDALS[rank - 1] || "🏅",
        body,
      };
    })
    .filter((item) => item.body);
}

function resolveStatus(race) {
  const actual = String(race?.actual_text || "");
  if (actual && !actual.includes("未")) {
    return { label: "結果確定", tone: "settled" };
  }
  return { label: "予想公開中", tone: "open" };
}

export default function RaceCardHeader({ race, actions = null }) {
  const status = resolveStatus(race);
  const badges = [formatOffTime(race?.scheduled_off_time), race?.distance_label].filter(Boolean);
  const resultText = race?.actual_text || "結果未確定";
  const resultEntries = parseResultEntries(resultText);

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

      <div className="race-card-header__result-row">
        <div className="race-card-header__result">
          <span className="race-card-header__result-label">着順</span>
          <div className="race-card-header__result-body">
            {resultEntries.length ? (
              <ul className="race-card-header__result-list">
                {resultEntries.map((entry) => (
                  <li key={entry.key}>
                    <span className="race-card-header__result-medal" aria-hidden="true">
                      {entry.medal}
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
        {actions ? <div className="race-card-header__actions">{actions}</div> : null}
      </div>
    </header>
  );
}
