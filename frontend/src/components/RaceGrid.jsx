import React, { useMemo, useState } from "react";
import RaceCard from "./RaceCard";

function displayOrderValue(race) {
  const value = Number(race?.display_order);
  return Number.isFinite(value) ? value : Number.MAX_SAFE_INTEGER;
}

export function sortRacesForDisplay(races) {
  return [...(races || [])].sort(
    (a, b) => displayOrderValue(a) - displayOrderValue(b),
  );
}

function displayVariant(race) {
  return String(race?.display_variant || "").trim();
}

function isSettledRace(race) {
  return displayVariant(race) === "settled";
}

export default function RaceGrid({ races }) {
  const [tab, setTab] = useState("all");

  const filtered = useMemo(() => {
    if (tab === "settled") {
      return sortRacesForDisplay((races || []).filter(isSettledRace));
    }
    if (tab === "open") {
      return sortRacesForDisplay(
        (races || []).filter((race) => !isSettledRace(race)),
      );
    }
    return sortRacesForDisplay(races || []);
  }, [races, tab]);

  return (
    <div className="race-grid-section">
      <div className="race-grid-tabs">
        <button
          type="button"
          className={tab === "all" ? "is-active" : ""}
          onClick={() => setTab("all")}
        >
          すべて
        </button>
        <button
          type="button"
          className={tab === "open" ? "is-active" : ""}
          onClick={() => setTab("open")}
        >
          確定待ち
        </button>
        <button
          type="button"
          className={tab === "settled" ? "is-active" : ""}
          onClick={() => setTab("settled")}
        >
          結果確定
        </button>
      </div>

      <div className="race-grid">
        {filtered.map((race) => (
          <RaceCard
            key={`${race.card_id || race.run_id || ""}-${race.display_header?.title || race.race_title || ""}`}
            race={race}
            style={{ order: displayOrderValue(race) }}
          />
        ))}
      </div>
    </div>
  );
}
