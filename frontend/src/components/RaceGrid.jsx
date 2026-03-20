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

function toFiniteNumber(value, fallback) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function displayOrderValue(race) {
  const index = toFiniteNumber(race?.display_sort_index, Number.NaN);
  if (Number.isFinite(index)) return index;

  const isPlaceholder = Boolean(race?.is_placeholder);
  if (isPlaceholder) {
    return 10000 + extractOffTimeValue(race?.scheduled_off_time);
  }

  return -extractRaceRank(race?.race_title);
}

export function sortRacesForDisplay(races) {
  return [...(races || [])].sort((a, b) => {
    const aIndex = toFiniteNumber(a?.display_sort_index, Number.NaN);
    const bIndex = toFiniteNumber(b?.display_sort_index, Number.NaN);
    if (Number.isFinite(aIndex) && Number.isFinite(bIndex) && aIndex !== bIndex) {
      return aIndex - bIndex;
    }

    const aGroup = toFiniteNumber(a?.display_sort_group, Number.NaN);
    const bGroup = toFiniteNumber(b?.display_sort_group, Number.NaN);
    const aValue = toFiniteNumber(a?.display_sort_value, Number.NaN);
    const bValue = toFiniteNumber(b?.display_sort_value, Number.NaN);
    const aLabel = String(a?.display_sort_label || "");
    const bLabel = String(b?.display_sort_label || "");
    if (Number.isFinite(aGroup) && Number.isFinite(bGroup)) {
      if (aGroup !== bGroup) return aGroup - bGroup;
      if (Number.isFinite(aValue) && Number.isFinite(bValue) && aValue !== bValue) return aValue - bValue;
      if (aLabel || bLabel) return aLabel.localeCompare(bLabel, "ja");
    }

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
          <RaceCard
            key={`${race.run_id}-${race.race_title}`}
            race={race}
            style={{ order: displayOrderValue(race) }}
          />
        ))}
      </div>
    </div>
  );
}
