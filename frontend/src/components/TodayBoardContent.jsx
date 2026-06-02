import React, { useEffect, useState } from "react";
import EmptyRaceState from "./EmptyRaceState";
import ModelTopFiveBoard from "./ModelTopFiveBoard";
import RaceGrid from "./RaceGrid";

function hasAgentPredictionRows(races) {
  return (Array.isArray(races) ? races : []).some(
    (race) => String(race?.source_type || "").trim() === "agent_prediction",
  );
}

export default function TodayBoardContent({ data, races, appShell = false }) {
  const [mode, setMode] = useState("race");
  const isAgentPredictionBoard = hasAgentPredictionRows(races);
  const showViewSwitch = !isAgentPredictionBoard;

  useEffect(() => {
    if (isAgentPredictionBoard && mode !== "race") {
      setMode("race");
    }
  }, [isAgentPredictionBoard, mode]);

  return (
    <>
      {showViewSwitch ? (
        <div className="board-view-switch" role="tablist" aria-label="表示切替">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "race"}
          className={mode === "race" ? "is-active" : ""}
          onClick={() => setMode("race")}
        >
          レース別
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "model"}
          className={mode === "model" ? "is-active" : ""}
          onClick={() => setMode("model")}
        >
          モデル別
        </button>
      </div>
      ) : null}

      {data?.fallback_notice ? (
        <section className="notice-strip">{data.fallback_notice}</section>
      ) : null}

      {races.length ? (
        mode === "race" ? (
          <RaceGrid races={races} appShell={appShell} />
        ) : (
          <ModelTopFiveBoard data={data} races={races} />
        )
      ) : (
        <EmptyRaceState />
      )}
    </>
  );
}
