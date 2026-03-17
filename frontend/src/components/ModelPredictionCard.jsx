import React from "react";
import RecoveryRateBadge from "./RecoveryRateBadge";
import ResultBadge from "./ResultBadge";

function parseMarks(text) {
  const source = String(text || "").trim();
  if (!source || source === "-") return [];
  const matches = [...source.matchAll(/([◎○▲△☆])\s*([0-9]+)/g)];
  return matches.map((item) => ({ symbol: item[1], horseNo: item[2] }));
}

function resolveTicketLines(text) {
  const lines = String(text || "")
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
  return lines.length ? lines : [];
}

function ticketStateText(card, ticketLines) {
  const decision = String(card?.decision_text || "").trim().toLowerCase();
  if (!ticketLines.length) {
    if (decision === "no_bet") return "買い目なし";
    return "集計中";
  }
  return "";
}

function formatResultLine(card) {
  const result = String(card?.result_triplet_text || "").trim();
  return result && result !== "-" ? result : "未確定";
}

export default function ModelPredictionCard({ card }) {
  const marks = parseMarks(card?.marks_text);
  const ticketLines = resolveTicketLines(card?.ticket_plan_text);
  const ticketState = ticketStateText(card, ticketLines);

  return (
    <article className={`prediction-card prediction-card--${card.engine}`}>
      <div className="prediction-card__head">
        <div>
          <h3>{card.label}</h3>
        </div>
        <ResultBadge label={card.status_label} tone={card.status_tone} />
      </div>

      <div className="prediction-card__marks">
        {marks.length ? (
          marks.map((mark) => (
            <div key={`${card.engine}-${mark.symbol}-${mark.horseNo}`} className="mark-chip">
              <span className="mark-chip__symbol">{mark.symbol}</span>
              <strong className="mark-chip__horse">{mark.horseNo}</strong>
            </div>
          ))
        ) : (
          <div className="state-chip">未生成</div>
        )}
      </div>

      <div className="prediction-card__body">
        <section className="prediction-card__section">
          <span className="prediction-card__label">買い目</span>
          {ticketLines.length ? (
            <ul className="ticket-list">
              {ticketLines.map((line) => (
                <li key={`${card.engine}-${line}`}>{line}</li>
              ))}
            </ul>
          ) : (
            <div className="state-chip">{ticketState}</div>
          )}
        </section>

        <section className="prediction-card__side">
          <div className="prediction-card__result-box">
            <span className="prediction-card__label">結果</span>
            <strong>{formatResultLine(card)}</strong>
          </div>
          <RecoveryRateBadge label="回収率" roiText={card.roi_text || "-"} />
        </section>
      </div>
    </article>
  );
}
