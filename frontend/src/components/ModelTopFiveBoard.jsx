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
    const engine = String(item?.engine || "").trim();
    if (!engine) continue;
    map.set(engine, item);
  }
  return map;
}

function buildModelRows(races, summaryCards) {
  const summaryMap = buildSummaryMap(summaryCards);
  const order = [];
  const rows = new Map();

  const ensureRow = (engine, label = "") => {
    if (!engine) return null;
    if (!rows.has(engine)) {
      order.push(engine);
      rows.set(engine, {
        engine,
        label: label || summaryMap.get(engine)?.label || engine,
        summary: summaryMap.get(engine) || null,
        items: [],
      });
    }
    const current = rows.get(engine);
    if (!current.label) {
      current.label = label || summaryMap.get(engine)?.label || engine;
    }
    return current;
  };

  for (const item of summaryCards || []) {
    ensureRow(String(item?.engine || "").trim(), String(item?.label || "").trim());
  }

  for (const race of sortRacesForDisplay(races || [])) {
    for (const card of race?.cards || []) {
      const engine = String(card?.engine || "").trim();
      const row = ensureRow(engine, String(card?.label || "").trim());
      if (!row) continue;
      row.items.push({
        key: raceCardKey(race, engine),
        race,
        card,
      });
    }
  }

  return order
    .map((engine) => rows.get(engine))
    .filter((item) => item && item.items.length > 0);
}

function formatHitText(summary) {
  const raceCount = Number(summary?.settled_races || summary?.races || 0);
  const hitRaces = Number(summary?.hit_races || 0);
  if (!raceCount) return "-";
  return `${hitRaces} / ${raceCount}`;
}

export default function ModelTopFiveBoard({ data, races }) {
  const modelRows = useMemo(
    () => buildModelRows(races, data?.summary_cards || []),
    [data, races],
  );
  const [engine, setEngine] = useState("");

  useEffect(() => {
    if (!modelRows.length) {
      setEngine("");
      return;
    }
    if (!modelRows.some((item) => item.engine === engine)) {
      setEngine(modelRows[0].engine);
    }
  }, [engine, modelRows]);

  const active = modelRows.find((item) => item.engine === engine) || modelRows[0];

  if (!active) {
    return null;
  }

  return (
    <section className="model-top-five-board">
      <div className="model-top-five-board__tabs" role="tablist" aria-label="モデル切替">
        {modelRows.map((item) => (
          <button
            key={item.engine}
            type="button"
            role="tab"
            aria-selected={active.engine === item.engine}
            className={active.engine === item.engine ? "is-active" : ""}
            onClick={() => setEngine(item.engine)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="model-top-five-board__header">
        <div className="model-top-five-board__stats">
          <div className="model-top-five-board__stat">
            <span>回収率</span>
            <strong>{active.summary?.roi_text || "-"}</strong>
          </div>
          <div className="model-top-five-board__stat">
            <span>的中</span>
            <strong>{formatHitText(active.summary)}</strong>
          </div>
          <div className="model-top-five-board__stat">
            <span>表示</span>
            <strong>{active.items.length}件</strong>
          </div>
        </div>
      </div>

      <div className="race-grid model-top-five-board__grid">
        {active.items.map((item) => (
          <article key={item.key} className="race-card model-top-five-board__item">
            <RaceCardHeader race={item.race} />
            <ModelRaceSummary card={item.card} />
          </article>
        ))}
      </div>
    </section>
  );
}
