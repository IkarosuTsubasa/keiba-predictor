import React from "react";
import RaceCourseView from "./RaceCourseView";
import RacePackView from "./RacePackView";
import CourseElevation from "./CourseElevation";
import { formatTime } from "../lib/raceSimulator";

export default function RaceSimulationRenderer({ simulation, frame, time, selected, onSelect, view = "course", companion = null, presentation = false }) {
  const leader = frame.order[0];
  const progress = leader.distance / simulation.distance;
  const phase = time === 0 ? "ゲートイン" : leader.finishTime !== null ? "ゴール" : progress < .12 ? "スタート" : progress < .4 ? "先行争い" : progress < .67 ? "道中" : "ラストスパート";
  const remaining = Math.max(0, Math.ceil((simulation.distance - leader.distance) / 10) * 10);
  const View = view === "pack" ? RacePackView : RaceCourseView;
  return (
    <section className="sim-stage-panel" aria-label="展開シミュレーション">
      <div className="sim-stage-visual">
        <div className="sim-course-panel">
          <div className="sim-stage-heading">
            <strong>{phase}</strong>
            <span className="sim-stage-clock">経過 {formatTime(time)}<b>残り {remaining}m</b></span>
          </div>
          <div className="sim-track-area">
            <View frame={frame} simulation={simulation} selected={selected} onSelect={onSelect} phase={phase} numbered={presentation} />
          </div>
        </div>
        {companion}
      </div>
      <CourseElevation simulation={simulation} frame={frame} selected={selected} compact />
    </section>
  );
}
