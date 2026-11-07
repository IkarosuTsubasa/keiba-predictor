import React, { useMemo } from "react";
import { FRAME_COLORS } from "../lib/raceSimulator";
import { courseGeometryPath, coursePoint, courseSection, terrainAt } from "../lib/raceCourses";

export default function RaceCourseView({ frame, simulation, selected, onSelect, phase, numbered = false }) {
  const course=simulation.course;
  const profiles=useMemo(()=>new Map(simulation.profiles.map(h=>[h.horse_no,h])),[simulation]);
  // Keep the display schematic compact without changing race distance or terrain calculations.
  const yScale = numbered ? .78 : 1;
  const yOffset = numbered ? 10 : 0;
  const projectPoint = (distance, raceDistance, lane) => {
    const point = coursePoint(course, distance, raceDistance, lane);
    return { ...point, y: point.y * yScale + yOffset };
  };
  const path=useMemo(()=>{
    if (!course) return "";
    if (!numbered) return courseGeometryPath(course,simulation.distance);
    const length=course.direction==="straight"?1000:course.lap_m;
    return Array.from({ length: 181 }, (_, i) => {
      const point = coursePoint(course, length * i / 180, length, 3);
      return `${i ? "L" : "M"}${point.x.toFixed(2)},${(point.y * yScale + yOffset).toFixed(2)}`;
    }).join(" ");
  }, [course, simulation.distance, numbered, yScale, yOffset]);
  if (!course) return <div className="sim-course-empty">この走路の公式データは未登録です。走路条件からコースを選択できます。</div>;
  const trackColor=course.surface==="芝"?"#6f8a57":"#ad8653";
  const samples=Array.from({length:120},(_,i)=>{
    const distance=(course.lap_m||1000)*i/120;
    return { point:projectPoint(distance,course.lap_m||1000,3),terrain:terrainAt(course,distance,course.lap_m||1000) };
  });
  const start=projectPoint(0,simulation.distance,3);
  const finish=projectPoint(simulation.distance,simulation.distance,3);
  const selectedState=frame.horses.find(h=>h.horse_no===selected)||frame.order[0];
  const section=courseSection(course,selectedState.distance,simulation.distance);
  const currentTerrain=terrainAt(course,selectedState.distance,simulation.distance);
  return <svg className="sim-real-course" viewBox={`0 0 860 ${numbered ? 370 : 455}`} role="group" aria-label={`${course.label}のコース図。公式平面図を模式化し、高低差を強調しています。`}>
    <defs><pattern id="sim-real-grass" width="18" height="18" patternUnits="userSpaceOnUse"><path d="M0 18 18 0" stroke="#abc49c" strokeOpacity=".04" /></pattern></defs>
    <rect width="860" height="455" fill="url(#sim-real-grass)" />
    <path d={path} transform="translate(0 14)" fill="none" stroke="#091710" strokeWidth="57" strokeLinecap="round" strokeLinejoin="round" />
    <path d={path} fill="none" stroke="#b4b89a" strokeOpacity=".55" strokeWidth="46" strokeLinejoin="round" strokeLinecap="round" />
    <path d={path} fill="none" stroke={trackColor} strokeWidth="40" strokeLinejoin="round" strokeLinecap="round" />
    <path d={path} fill="none" stroke="#c7d0ad" strokeOpacity=".3" strokeWidth="1" strokeDasharray="5 10" />
    {samples.map((item,i)=>{
      const next=samples[(i+1)%samples.length];
      if (course.direction==="straight" && i===samples.length-1) return null;
      if(!item.terrain.known||Math.abs(item.terrain.grade)<.0015)return null;
      return <path key={i} d={`M${item.point.x},${item.point.y}L${next.point.x},${next.point.y}`} stroke={item.terrain.grade>0?"#dfaa62":"#74b7c5"} strokeWidth="7" strokeOpacity=".8" fill="none" />;
    })}
    {[.12,.48,.78].map(f=>{
      const point=projectPoint((course.lap_m||1000)*f,course.lap_m||1000,3);
      return <path key={f} d="m-5-5 7 5-7 5" transform={`translate(${point.x} ${point.y}) rotate(${point.angle*180/Math.PI})`} stroke="#f0e5bc" strokeWidth="2" fill="none" />;
    })}
    <g transform={`translate(${finish.x} ${finish.y}) rotate(${finish.angle*180/Math.PI})`}><path d="M0-23V23" stroke="#fff2d2" strokeWidth="5" />{Array.from({length:8},(_,i)=><rect key={i} x={i%2?-2.5:0} y={-22+i*5.5} width="2.5" height="5.5" fill="#183023" />)}</g>
    <text x={finish.x} y={finish.y+44} textAnchor="middle" className="sim-map-label">ゴール</text>
    <g transform={`translate(${start.x} ${start.y})`}><circle r="9" fill="#e0c184" stroke="#182b21" strokeWidth="2" /><text y="-18" textAnchor="middle" className="sim-map-label">発走</text></g>
    <text x="435" y={numbered?165:195} textAnchor="middle" className="sim-map-kicker">{course.label}</text>
    {!numbered && <text x="435" y="230" textAnchor="middle" className="sim-real-phase">{phase}</text>}
    <text x="435" y={numbered?195:257} textAnchor="middle" className="sim-map-sub">{section.label} · {currentTerrain.label}</text>
    <text x="435" y={numbered?221:285} textAnchor="middle" className="sim-map-small">{course.elevation_points?'平面形状・発走位置は近似／高さは強調':'平面形状・発走位置は近似／坂位置未取得'}</text>
    {frame.horses.filter(h=>h.horse_no!==selectedState.horse_no).concat(selectedState).map(state=>{
      const horse=profiles.get(state.horse_no),active=state.horse_no===selectedState.horse_no;
      const point=projectPoint(state.distance,simulation.distance,state.lane);
      const selectionStroke = numbered ? "#fff" : "#f3d897";
      return <g key={state.horse_no} className="sim-runner" transform={`translate(${point.x} ${point.y})`} role="button" tabIndex="0" aria-label={`${horse.horse_no}番 ${horse.horse_name}の詳細`} aria-pressed={active} onClick={()=>onSelect(state.horse_no)} onKeyDown={e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();onSelect(state.horse_no);}}}>
        <title>{horse.horse_name}</title><circle r={numbered ? 12 : active?9:5} fill={FRAME_COLORS[horse.frame-1]} stroke={active ? selectionStroke : "#162a20"} strokeWidth={active?2:1} />
        {numbered ? <><text y="4" textAnchor="middle" fontSize="12" fontWeight="700" fill={horse.frame === 2 ? "#fff" : "#101e30"}>{horse.horse_no}</text>{active && <circle r="17" fill="none" stroke="#fff" strokeWidth="2" />}</> : active&&<><circle r="15" fill="none" stroke="#f3d897" strokeOpacity=".65" /><rect x="-12" y="-38" width="24" height="22" rx="5" fill={FRAME_COLORS[horse.frame-1]} stroke="#f3d897" /><text y="-23" textAnchor="middle" fontSize="12" fontWeight="700" fill={horse.frame===1||horse.frame===5?"#203126":"#fff"}>{horse.horse_no}</text><path d="M0-16V-10" stroke="#f3d897" /></>}
      </g>;
    })}
    {course.elevation_points&&<g transform={`translate(28 ${numbered?350:425})`} className="sim-terrain-legend"><path d="M0 0h22" stroke="#dfaa62" strokeWidth="4" /><text x="28" y="4">上り</text><path d="M80 0h22" stroke="#74b7c5" strokeWidth="4" /><text x="108" y="4">下り</text></g>}
  </svg>;
}
