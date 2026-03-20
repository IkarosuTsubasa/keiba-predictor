import React, { useMemo, useState } from "react";
import RaceCard from "./RaceCard";

function extractRaceRank(title) {
  const match = String(title || "").match(/(\d+)R/i);
  return match ? Number(match[1]) : 0;
}

export function sortRacesForDisplay(races) {
  return [...(races || [])].sort((a, b) => {
    const rankDiff = extractRaceRank(b?.race_title) - extractRaceRank(a?.race_title);
    if (rankDiff !== 0) return rankDiff;
    return String(a?.race_title || "").localeCompare(String(b?.race_title || ""), "ja");
  });
}

function isSettledRace(race) {
  const actual = String(race?.actual_text || "");
  return Boolean(actual && !actual.includes("未"));
}

export default function RaceGrid({ races }) {
  const [tab, setTab] = useState("all");

  const filtered = useMemo(() => {
    if (tab === "settled") {
      return (races || []).filter((race) => isSettledRace(race));
    }
    if (tab === "open") {
      return (races || []).filter((race) => !isSettledRace(race));
    }
    return races || [];
  }, [races, tab]);

  return (
    <div className="race-grid-section">
      <div className="race-grid-tabs">
        <button type="button" className={tab === "all" ? "is-active" : ""} onClick={() => setTab("all")}>
          すべて
        </button>
        <button type="button" className={tab === "open" ? "is-active" : ""} onClick={() => setTab("open")}>
          出走待ち
        </button>
        <button type="button" className={tab === "settled" ? "is-active" : ""} onClick={() => setTab("settled")}>
          結果確定
        </button>
      </div>

      <div className="race-grid">
        {filtered.map((race) => (
          <RaceCard key={`${race.run_id}-${race.race_title}`} race={race} />
        ))}
      </div>
    </div>
  );
}
