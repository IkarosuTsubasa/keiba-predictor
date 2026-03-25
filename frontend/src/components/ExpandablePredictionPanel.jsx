import React from "react";
import BetPreviewList from "./BetPreviewList";
import ModelMetaBadge from "./ModelMetaBadge";

function parseMarks(text) {
  return [...String(text || "").matchAll(/([◎○▲△☆])\s*([0-9]+)/g)].map(
    (item) => ({
      symbol: item[1],
      horseNo: item[2],
    }),
  );
}

function resultTone(card) {
  const result = String(card?.result_triplet_text || "");
  if (result && result !== "-" && !result.includes("未")) return "positive";
  if (String(card?.status_label || "").includes("未")) return "muted";
  return "neutral";
}

export default function ExpandablePredictionPanel({
  cards,
  highlightRoi = false,
}) {
  return (
    <div className="expandable-prediction-panel">
      {(cards || []).map((card) => (
        <section key={card.engine} className="expandable-prediction-panel__model">
          <div className="expandable-prediction-panel__top">
            <strong>{card.label}</strong>
            <div className="expandable-prediction-panel__meta">
              <ModelMetaBadge
                label="ROI"
                value={card.roi_text || "-"}
                tone="subtle"
                dynamicRoi={highlightRoi}
              />
            </div>
          </div>

          <div className="expandable-prediction-panel__marks">
            {parseMarks(card?.marks_text).length ? (
              parseMarks(card?.marks_text).map((mark) => (
                <span
                  key={`${card.engine}-${mark.symbol}-${mark.horseNo}`}
                  className="expandable-prediction-panel__mark"
                >
                  {mark.symbol}
                  {mark.horseNo}
                </span>
              ))
            ) : (
              <span className="expandable-prediction-panel__empty">印なし</span>
            )}
          </div>

          <BetPreviewList text={card?.ticket_plan_text || ""} />

          <div
            className={`expandable-prediction-panel__result expandable-prediction-panel__result--${resultTone(card)}`}
          >
            <span>結果</span>
            <strong>{card?.result_triplet_text || "結果はまだありません"}</strong>
          </div>
        </section>
      ))}
    </div>
  );
}
