import React, { useMemo, useState } from "react";
import RaceCard from "./RaceCard";

function extractRaceRank(title) {
  const match = String(title || "").match(/(\d+)R/i);
  return match ? Number(match[1]) : 0;
}

function extractOffTimeValue(text) {
  const source = String(text || "").trim();
  const match = source.match(/T?(\d{2}):(\d{2})/);
  if (!match) return Number.POSITIVE_INFINITY;
  return Number(match[1]) * 60 + Number(match[2]);
}

export function sortRacesForDisplay(races) {
  return [...(races || [])].sort((a, b) => {
    const aPlaceholder = Boolean(a?.is_placeholder);
    const bPlaceholder = Boolean(b?.is_placeholder);
    if (aPlaceholder !== bPlaceholder) return aPlaceholder ? 1 : -1;

    if (aPlaceholder && bPlaceholder) {
      const timeDiff = extractOffTimeValue(a?.scheduled_off_time) - extractOffTimeValue(b?.scheduled_off_time);
      if (timeDiff !== 0) return timeDiff;
      return String(a?.race_title || "").localeCompare(String(b?.race_title || ""), "ja");
    }

    const rankDiff = extractRaceRank(b?.race_title) - extractRaceRank(a?.race_title);
    if (rankDiff !== 0) return rankDiff;
    return String(a?.race_title || "").localeCompare(String(b?.race_title || ""), "ja");
  });
}

function isSettledRace(race) {
  if (race?.is_placeholder) return false;
  const actual = String(race?.actual_text || "");
  return Boolean(actual && !actual.includes("未"));
}

export default function RaceGrid({ races }) {
  const [tab, setTab] = useState("all");

  const filtered = useMemo(() => {
    if (tab === "settled") {
      return sortRacesForDisplay((races || []).filter((race) => isSettledRace(race)));
    }
    if (tab === "open") {
      return sortRacesForDisplay((races || []).filter((race) => !isSettledRace(race)));
    }
    return sortRacesForDisplay(races || []);
  }, [races, tab]);

  return (
    <div className="race-grid-section">
      <div className="race-grid-tabs">
        <button type="button" className={tab === "all" ? "is-active" : ""} onClick={() => setTab("all")}>
          すべて
        </button>
        <button type="button" className={tab === "open" ? "is-active" : ""} onClick={() => setTab("open")}>
          確定待ち
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
