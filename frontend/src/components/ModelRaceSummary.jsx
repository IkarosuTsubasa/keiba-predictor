import React from "react";
import BetPreviewList from "./BetPreviewList";
import ModelMetaBadge from "./ModelMetaBadge";

const MARK_ORDER = ["◎", "○", "▲", "△", "☆"];

function parseMarks(text) {
  return [...String(text || "").matchAll(/([◎○▲△☆])\s*([0-9]+)/g)].map(
    (item) => ({
      symbol: item[1],
      horseNo: item[2],
    }),
  );
}

function pickHorse(marks, symbol) {
  const found = (marks || []).find((item) => item.symbol === symbol);
  return found ? found.horseNo : null;
}

export default function ModelRaceSummary({ card, highlightRoi = false }) {
  const marks = parseMarks(card?.marks_text);
  const markItems = MARK_ORDER.map((symbol) => ({
    symbol,
    horseNo: pickHorse(marks, symbol),
  })).filter((item) => item.horseNo);

  const main =
    markItems.find((item) => item.symbol === "◎") || { symbol: "◎", horseNo: "-" };
  const secondary = markItems.filter((item) => item.symbol !== "◎");

  return (
    <div className="model-race-summary">
      <div className="model-race-summary__top">
        <div className="model-race-summary__marks">
          <span className="model-race-summary__mark model-race-summary__mark--main">
            <em>{main.symbol}</em>
            <strong>{main.horseNo}</strong>
          </span>
          {secondary.map((item) => (
            <span
              key={`${item.symbol}-${item.horseNo}`}
              className="model-race-summary__mark"
            >
              <em>{item.symbol}</em>
              <strong>{item.horseNo}</strong>
            </span>
          ))}
        </div>
        <div className="model-race-summary__meta">
          <ModelMetaBadge
            label="ROI"
            value={card?.roi_text || "-"}
            tone="subtle"
            dynamicRoi={highlightRoi}
          />
        </div>
      </div>

      <div className="model-race-summary__tickets">
        <BetPreviewList text={card?.ticket_plan_text || ""} />
      </div>
    </div>
  );
}
