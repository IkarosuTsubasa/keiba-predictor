import React, { useCallback, useEffect, useState } from "react";
import { DashboardInsightPanel } from "./DashboardInsightPanel";
import EmptyRaceState from "./EmptyRaceState";
import RaceGrid from "./RaceGrid";

function hasAgentPredictionRows(races) {
  return (Array.isArray(races) ? races : []).some(
    (race) => String(race?.source_type || "").trim() === "agent_prediction",
  );
}

export default function TodayBoardContent({ data, races, appShell = false }) {
  const [visibleRaces, setVisibleRaces] = useState(races);
  const isAgentPredictionBoard = Boolean(data?.agent_mode) || hasAgentPredictionRows(races);
  const handleVisibleRacesChange = useCallback((nextRaces) => {
    setVisibleRaces(nextRaces);
  }, []);

  useEffect(() => {
    setVisibleRaces(races);
  }, [races]);

  return (
    <>
      {data?.fallback_notice ? (
        <section className="notice-strip">{data.fallback_notice}</section>
      ) : null}

      {races.length ? (
        appShell ? (
          <RaceGrid races={races} appShell={appShell} />
        ) : (
          <div className="dashboard-board">
            <div className="dashboard-board__table">
              <RaceGrid
                races={races}
                appShell={appShell}
                onVisibleRacesChange={handleVisibleRacesChange}
              />
            </div>
            <DashboardInsightPanel data={data} races={visibleRaces} />
          </div>
        )
      ) : (
        <EmptyRaceState agentMode={isAgentPredictionBoard} />
      )}
    </>
  );
}
