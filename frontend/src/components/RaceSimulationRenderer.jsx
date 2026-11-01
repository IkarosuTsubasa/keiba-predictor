import React from "react";
import RaceCourseView from "./RaceCourseView";
import RacePackView from "./RacePackView";
import CourseElevation from "./CourseElevation";
import { formatTime } from "../lib/raceSimulator";

export default function RaceSimulationRenderer({ simulation, frame, time, selected, onSelect, view = "course" }) {
  const leader = frame.order[0];
  const progress = leader.distance / simulation.distance;
  const phase = time === 0 ? "ゲートイン" : leader.finishTime !== null ? "ゴール" : progress < .12 ? "スタート" : progress < .4 ? "先行争い" : progress < .67 ? "道中" : "ラストスパート";
  const View = view === "pack" ? RacePackView : RaceCourseView;
  return <section className="sim-stage-panel" aria-label="展開シミュレーション">
    <div className="sim-stage-heading"><strong>{phase}</strong><span>経過 {formatTime(time)} · 残り {Math.max(0, Math.ceil((simulation.distance - leader.distance) / 10) * 10)}m</span></div>
    <div className="sim-track-area"><View frame={frame} simulation={simulation} selected={selected} onSelect={onSelect} phase={phase} /></div>
    <CourseElevation simulation={simulation} frame={frame} selected={selected} />
  </section>;
}
