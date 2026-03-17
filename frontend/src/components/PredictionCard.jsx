import React, { useMemo } from "react";
import ExpandableBetList from "./ExpandableBetList";
import ModelBadge from "./ModelBadge";
import RecoveryRateBadge from "./RecoveryRateBadge";
import ResultBadge from "./ResultBadge";

function parseMarks(text) {
  const matches = [...String(text || "").matchAll(/([◎○▲△☆])\s*([0-9]+)/g)];
  return matches.map((item) => ({ symbol: item[1], horseNo: item[2] }));
}

function markLabel(symbol) {
  if (symbol === "◎") return "本命";
  if (symbol === "○") return "対抗";
  if (symbol === "▲") return "単穴";
  return "補完";
}

function resolveResultTone(card) {
  const status = String(card?.status_tone || "").toLowerCase();
  if (status === "settled" || status === "result" || status === "positive") return "positive";
  if (status === "pending") return "pending";
  return "muted";
}

function resolveResultLabel(card) {
  const result = String(card?.result_triplet_text || "").trim();
  if (result && result !== "-" && result !== "未確定") {
    return result;
  }
  if (String(card?.status_label || "").includes("未")) {
    return "未確定";
  }
  return "集計中";
}

export default function PredictionCard({ card }) {
  const marks = useMemo(() => parseMarks(card?.marks_text), [card?.marks_text]);

  return (
    <article className={`prediction-card prediction-card--${card?.engine || "neutral"}`}>
      <div className="prediction-card__header">
        <div className="prediction-card__title">
          <ModelBadge engine={card?.engine} label={card?.label || "-"} />
          <span className="prediction-card__confidence">{card?.confidence_text || "N/A"}</span>
        </div>
        <div className="prediction-card__badges">
          <RecoveryRateBadge label="ROI" roiText={card?.roi_text || "-"} compact />
          <ResultBadge label={resolveResultLabel(card)} tone={resolveResultTone(card)} />
        </div>
      </div>

      <div className="prediction-card__marks">
        {marks.length ? (
          marks.map((mark) => (
            <div key={`${card?.engine}-${mark.symbol}-${mark.horseNo}`} className={`prediction-card__mark prediction-card__mark--${mark.symbol}`}>
              <span className="prediction-card__mark-meta">{markLabel(mark.symbol)}</span>
              <strong>{mark.symbol}{mark.horseNo}</strong>
            </div>
          ))
        ) : (
          <div className="empty-chip">印未生成</div>
        )}
      </div>

      <div className="prediction-card__bets">
        <div className="prediction-card__bets-head">
          <span className="section-kicker">Recommended Bets</span>
          <span>{card?.ticket_count ? `${card.ticket_count} 点` : "買い目なし"}</span>
        </div>
        <ExpandableBetList text={card?.ticket_plan_text || ""} />
      </div>
    </article>
  );
}
