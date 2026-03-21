import React, { useState } from "react";
import EmptyRaceState from "./EmptyRaceState";
import ModelTopFiveBoard from "./ModelTopFiveBoard";
import RaceGrid from "./RaceGrid";

export default function TodayBoardContent({ data, races }) {
  const [mode, setMode] = useState("race");

  return (
    <>
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

      {data?.fallback_notice ? (
        <section className="notice-strip">{data.fallback_notice}</section>
      ) : null}

      {races.length ? (
        mode === "race" ? (
          <RaceGrid races={races} />
        ) : (
          <ModelTopFiveBoard data={data} races={races} />
        )
      ) : (
        <EmptyRaceState />
      )}
    </>
  );
}
