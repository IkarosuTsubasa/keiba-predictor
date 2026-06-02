import React from "react";
import EmptyRaceState from "./EmptyRaceState";
import RaceGrid from "./RaceGrid";

function hasAgentPredictionRows(races) {
  return (Array.isArray(races) ? races : []).some(
    (race) => String(race?.source_type || "").trim() === "agent_prediction",
  );
}

export default function TodayBoardContent({ data, races, appShell = false }) {
  const isAgentPredictionBoard = Boolean(data?.agent_mode) || hasAgentPredictionRows(races);

  return (
    <>
      {data?.fallback_notice ? (
        <section className="notice-strip">{data.fallback_notice}</section>
      ) : null}

      {races.length ? (
        <RaceGrid races={races} appShell={appShell} />
      ) : (
        <EmptyRaceState agentMode={isAgentPredictionBoard} />
      )}
    </>
  );
}
