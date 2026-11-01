import React, { useEffect, useMemo, useRef, useState } from "react";
import { FRAME_COLORS } from "../lib/raceSimulator";
import { layoutRacePack, PACK_ROW_HEIGHT } from "../lib/racePackLayout";
import { terrainAt } from "../lib/raceCourses";

export default function RacePackView({ frame, simulation, selected, onSelect, phase }) {
  const host = useRef(null);
  const [width, setWidth] = useState(820);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(320, entry.contentRect.width)));
    observer.observe(host.current);
    return () => observer.disconnect();
  }, []);
  const profiles = useMemo(() => new Map(simulation.profiles.map((h) => [h.horse_no, h])), [simulation]);
  const layout = layoutRacePack(frame.order, { width });
  const leaderDistance = frame.order[0].distance;
  const selectedProfile = profiles.get(selected);
  const selectedState = frame.horses.find(h=>h.horse_no===selected)||frame.order[0];
  const selectedTerrain = terrainAt(simulation.course, selectedState.distance, simulation.distance);
  const sorted = [...layout.horses].sort((a, b) => Number(a.horse_no === selected) - Number(b.horse_no === selected));
  return <div className="sim-pack-host" ref={host}>
    <svg className="sim-pack" viewBox={`0 0 ${width} ${layout.height}`} style={{ height: layout.height }} role="group" aria-label="馬群の拡大図。見やすい間隔に広げて表示し、各馬の番号を選ぶと詳細を表示します。">
      <defs>
        <linearGradient id="sim-pack-surface" x2="0" y2="1"><stop stopColor="#2d4735" /><stop offset="1" stopColor="#21392c" /></linearGradient>
        <pattern id="sim-pack-grain" width="36" height="36" patternUnits="userSpaceOnUse"><path d="m4 8 3-3m17 24 3-3" stroke="#bdcba1" strokeOpacity=".055" /></pattern>
      </defs>
      <text x="20" y="26" className="sim-pack-phase">{phase}</text>
      <text x={width/2} y="25" textAnchor="middle" className={`sim-pack-terrain${selectedTerrain.grade>.0015?' is-uphill':selectedTerrain.grade<-.0015?' is-downhill':''}`}>{selectedTerrain.label}</text>
      <text x={width - 20} y="25" textAnchor="end" className="sim-pack-direction">進行方向 →</text>
      <rect x="12" y="43" width={width - 24} height={layout.rows * PACK_ROW_HEIGHT + 10} rx="10" fill="url(#sim-pack-surface)" />
      <rect x="12" y="43" width={width - 24} height={layout.rows * PACK_ROW_HEIGHT + 10} rx="10" fill="url(#sim-pack-grain)" />
      {Array.from({ length: layout.rows + 1 }, (_, row) => <path key={row} d={`M18 ${48 + row * PACK_ROW_HEIGHT}H${width - 18}`} stroke="#b7c69c" strokeOpacity={row === 0 || row === layout.rows ? ".45" : ".1"} strokeWidth={row === 0 || row === layout.rows ? "2" : "1"} strokeDasharray={row === 0 || row === layout.rows ? undefined : "6 9"} />)}
      {sorted.map((state) => {
        const horse = profiles.get(state.horse_no);
        const active = horse.horse_no === selected;
        const color = FRAME_COLORS[horse.frame - 1];
        const ink = horse.frame === 1 || horse.frame === 5 ? "#243027" : "#fff";
        const gap = leaderDistance - state.distance;
        const stride = Math.sin(frame.time * 10 + horse.horse_no) * 4;
        const terrain = terrainAt(simulation.course, state.distance, simulation.distance);
        const pitch = Math.max(-8,Math.min(8,-Math.atan(terrain.grade)*180/Math.PI*4));
        return <g key={horse.horse_no} className={`sim-pack-runner${active ? " is-selected" : ""}`} transform={`translate(${state.x} ${state.y})`} role="button" tabIndex="0" aria-label={`${horse.horse_no}番 ${horse.horse_name}の詳細`} aria-pressed={active} data-horse={horse.horse_no} data-row={state.row} data-x={state.x} onClick={() => onSelect(horse.horse_no)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(horse.horse_no); } }}>
          <title>{`${horse.horse_no} ${horse.horse_name}`}</title>
          <rect x="-29" y="-26" width="58" height="58" rx="10" fill="transparent" pointerEvents="all" />
          {active && <rect x="-29" y="-26" width="58" height="50" rx="10" fill="#dfc182" fillOpacity=".1" stroke="#e5c88d" strokeWidth="1.3" />}
          <g transform={`rotate(${pitch})`}>
          <ellipse cx="-2" cy="14" rx="24" ry="3" fill="#091e13" opacity=".4" />
          <path d={`M-14 5l${-4 - stride} 9m9-9 3 8M9 5l${4 + stride} 9M13 4l7 6`} stroke={active ? "#f0dec1" : "#c6ad88"} strokeWidth="2.2" fill="none" strokeLinecap="round" />
          <path d="M-17 3q-9-7-12-2m11-2q3-8 12-7L9-6l6-10 8 2 3 7-8 2-2 9-8 2h-19Z" fill={active ? "#c6a67a" : "#a98a61"} stroke="#e1c9a0" strokeWidth="1" strokeLinejoin="round" />
          <path d="m-3-8 4-7 9 4-4 6" fill={color} stroke="#22352a" strokeWidth="1.5" />
          </g>
          <rect x="-14" y="-24" width="28" height="19" rx="5" fill={color} stroke={active ? "#f6dfac" : "#20382a"} strokeWidth="1.2" />
          <text y="-10" textAnchor="middle" fontSize="12" fontWeight="800" fill={ink}>{horse.horse_no}</text>
          <text y="28" textAnchor="middle" className="sim-pack-gap">{state.finishTime !== null ? "入線" : frame.time === 0 ? "発走前" : gap < 0.05 ? "先頭" : `${gap.toFixed(1)}m`}</text>
        </g>;
      })}
      <text x="20" y={layout.height - 8} className="sim-pack-caption">{selectedProfile ? `${selectedProfile.horse_no}番 ${selectedProfile.horse_name}` : ""}</text>
    </svg>
  </div>;
}
