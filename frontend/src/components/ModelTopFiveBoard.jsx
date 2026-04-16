import React, { useEffect, useMemo, useState } from "react";
import ModelRaceSummary from "./ModelRaceSummary";
import RaceCardHeader from "./RaceCardHeader";
import { sortRacesForDisplay } from "./RaceGrid";

function raceCardKey(race, engine) {
  return `${race?.card_id || race?.run_id || race?.race_id || ""}-${engine || ""}`;
}

function buildSummaryMap(summaryCards) {
  const map = new Map();
  for (const item of summaryCards || []) {
    const predictorId = String(item?.predictor_id || "").trim();
    if (!predictorId) continue;
    map.set(predictorId, item);
  }
  return map;
}

function buildModelRows(races, summaryCards) {
  const summaryMap = buildSummaryMap(summaryCards);
  const order = [];
  const rows = new Map();

  const ensureRow = (predictorId, label = "") => {
    if (!predictorId) return null;
    if (!rows.has(predictorId)) {
      order.push(predictorId);
      rows.set(predictorId, {
        predictorId,
        label: label || summaryMap.get(predictorId)?.label || predictorId,
        summary: summaryMap.get(predictorId) || null,
        items: [],
      });
    }
    const current = rows.get(predictorId);
    if (!current.label) {
      current.label = label || summaryMap.get(predictorId)?.label || predictorId;
    }
    return current;
  };

  for (const item of summaryCards || []) {
    ensureRow(String(item?.predictor_id || "").trim(), String(item?.label || "").trim());
  }

  for (const race of sortRacesForDisplay(races || [])) {
    for (const card of race?.predictor_compare_cards || []) {
      const predictorId = String(card?.predictor_id || "").trim();
      const row = ensureRow(predictorId, String(card?.label || "").trim());
      if (!row) continue;
      row.items.push({
        key: raceCardKey(race, predictorId),
        race,
        card: {
          ...card,
          metricText:
            summaryMap.get(predictorId)?.top5_to_top3_hit_rate_text ||
            summaryMap.get(predictorId)?.top3_hit_rate_text ||
            "-",
        },
      });
    }
  }

  return order
    .map((predictorId) => rows.get(predictorId))
    .filter((item) => item && item.items.length > 0);
}

export default function ModelTopFiveBoard({ data, races }) {
  const modelRows = useMemo(
    () => buildModelRows(races, data?.daily_predictor?.cards || []),
    [data, races],
  );
  const [predictorId, setPredictorId] = useState("");

  useEffect(() => {
    if (!modelRows.length) {
      setPredictorId("");
      return;
    }
    if (!modelRows.some((item) => item.predictorId === predictorId)) {
      setPredictorId(modelRows[0].predictorId);
    }
  }, [modelRows, predictorId]);

  const active = modelRows.find((item) => item.predictorId === predictorId) || modelRows[0];

  if (!active) {
    return null;
  }

  return (
    <section className="model-top-five-board">
      <div className="model-top-five-board__tabs" role="tablist" aria-label="モデル切替">
        {modelRows.map((item) => (
          <button
            key={item.predictorId}
            type="button"
            role="tab"
            aria-selected={active.predictorId === item.predictorId}
            className={active.predictorId === item.predictorId ? "is-active" : ""}
            onClick={() => setPredictorId(item.predictorId)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="model-top-five-board__header">
        <div className="model-top-five-board__stats">
          <div className="model-top-five-board__stat">
            <span>TOP5内カバー</span>
            <strong>{active.summary?.top5_to_top3_hit_rate_text || "-"}</strong>
          </div>
          <div className="model-top-five-board__stat">
            <span>本命1着率</span>
            <strong>{active.summary?.top1_hit_rate_text || "-"}</strong>
          </div>
          <div className="model-top-five-board__stat">
            <span>掲載</span>
            <strong>{active.items.length}件</strong>
          </div>
        </div>
      </div>

      <div className="race-grid model-top-five-board__grid">
        {active.items.map((item) => (
          <article key={item.key} className="race-card model-top-five-board__item">
            <RaceCardHeader race={item.race} />
            <ModelRaceSummary
              card={item.card}
              highlightRoi={false}
            />
          </article>
        ))}
      </div>
    </section>
  );
}
