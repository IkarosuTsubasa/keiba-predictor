import ModelMetaBadge from "./ModelMetaBadge";
import { buildPredictorConsensusSummary } from "../lib/publicRace";

const MARKS = ["◎", "○", "▲", "△", "☆"];

function formatConfidence(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${Math.round(number * 100)}%`;
}

export default function MorningRaceSummary({ race }) {
  const derivedSummary =
    !Array.isArray(race?.top5) || !race.top5.length
      ? buildPredictorConsensusSummary(race?.predictor_compare_cards)
      : null;
  const top5 = Array.isArray(race?.top5) && race.top5.length
    ? race.top5.slice(0, 5)
    : Array.isArray(derivedSummary?.top5)
      ? derivedSummary.top5
      : [];
  const main = top5[0] || null;
  const supportMarks = top5
    .slice(1)
    .map((item, index) => ({
      symbol: MARKS[index + 1],
      horseNo: item?.horse_no || "-",
    }))
    .filter((item) => item.horseNo && item.horseNo !== "-");

  return (
    <article className="ai-pick-summary ai-pick-summary--morning">
      <div className="ai-pick-summary__head">
        <ModelMetaBadge
          label="自信度"
          value={formatConfidence(race?.confidence_score ?? derivedSummary?.confidence_score)}
          tone="subtle"
        />
      </div>

      <div className="ai-pick-summary__marks">
        <div className="ai-pick-summary__main">
          <span>◎</span>
          <strong>{main?.horse_no || "-"}</strong>
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

      <div className="ai-pick-summary__meta-row">
        <span>{`一致度 ${formatConfidence(race?.agreement_score ?? derivedSummary?.agreement_score)}`}</span>
      </div>
    </article>
  );
}
