import React from "react";
import ModelMetaBadge from "./ModelMetaBadge";
import { MARK_ORDER, parseMarks, pickHorse } from "../lib/publicRace";

export default function AiPickSummary({ card, highlightRoi = false }) {
  const marks = parseMarks(card?.marks_text);
  const mainHorse = pickHorse(marks, "◎") || "-";
  const supportMarks = MARK_ORDER.filter((symbol) => symbol !== "◎")
    .map((symbol) => ({
      symbol,
      horseNo: pickHorse(marks, symbol),
    }))
    .filter((item) => item.horseNo);
  const badgeLabel = String(card?.metaLabel || "").trim() || "指標";
  const badgeValue = String(card?.metaValue || card?.roi_text || "").trim() || "-";

  return (
    <article
      className={`ai-pick-summary ai-pick-summary--${card?.engine || "generic"}`}
      data-share-text={card?.share_text || ""}
    >
      <div className="ai-pick-summary__head">
        <strong className="ai-pick-summary__model">{card?.label || "-"}</strong>
        <ModelMetaBadge
          label={badgeLabel}
          value={badgeValue}
          tone="subtle"
          dynamicRoi={highlightRoi}
        />
      </div>

      <div className="ai-pick-summary__marks">
        <div className="ai-pick-summary__main">
          <span>◎</span>
          <strong>{mainHorse}</strong>
        </div>

        {supportMarks.length ? (
          <div className="ai-pick-summary__subs">
            {supportMarks.map((item) => (
              <span
                key={`${item.symbol}-${item.horseNo}`}
                className="ai-pick-summary__submark"
              >
                <em>{item.symbol}</em>
                <strong>{item.horseNo}</strong>
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}
