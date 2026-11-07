import React, { useEffect, useMemo, useRef, useState } from "react";
import { raceElevationSamples, terrainAt } from "../lib/raceCourses";

export default function CourseElevation({ simulation, frame, selected, compact = false }) {
  const host=useRef(null);
  const [width,setWidth]=useState(850);
  useEffect(()=>{
    if(!host.current)return undefined;
    const observer=new ResizeObserver(([entry])=>setWidth(Math.max(300,entry.contentRect.width)));
    observer.observe(host.current);
    return ()=>observer.disconnect();
  },[simulation.course]);
  const course=simulation.course;
  const samples=useMemo(()=>raceElevationSamples(course,simulation.distance),[course,simulation.distance]);
  const state=frame.horses.find(h=>h.horse_no===selected)||frame.order[0];
  const current=terrainAt(course,state.distance,simulation.distance);
  if(!course)return null;
  const heights=samples.filter(s=>s.known).map(s=>s.height);
  const min=Math.floor(Math.min(0,...heights)*2)/2;
  const max=Math.max(min+1,Math.ceil(Math.max(0,...heights)*2)/2);
  const end=width-32;
  const x=d=>46+d/simulation.distance*(end-46);
  const baseline=compact?58:105;
  const y=h=>(compact?54:99)-(h-min)/(max-min)*(compact?30:67);
  const path=samples.map((s,i)=>`${i?"L":"M"}${x(s.distance).toFixed(1)},${y(s.height||0).toFixed(1)}`).join(" ");
  return <section ref={host} className="sim-elevation" aria-label="試走区間の高低断面">
    <div className="sim-elevation-heading"><strong>高低断面</strong><b className="sim-elevation-horse">{state.horse_no}番</b><span className={current.grade>.0015?"is-uphill":current.grade<-.0015?"is-downhill":""}>{current.label}{current.known&&Math.abs(current.grade)>=.0015?` ${current.grade>0?"+":""}${(current.grade*100).toFixed(1)}%`:""}</span><small>{course.elevation_accuracy==='official_flat'?"公式公表の平坦走路":current.known?"公式断面図を近似／縦倍率を拡大":"高低差のみ確認・坂位置は未取得"}</small></div>
    {course.elevation_points?<svg viewBox={`0 0 ${width} ${compact?85:140}`} style={{height:compact?85:140}} role="img" aria-label={`${course.label}、試走距離${simulation.distance}メートルの高低断面。勾配は図の読み取りによる近似値です。`}>
      <path d={`${path}L${end},${baseline}H46Z`} fill="#c5ac7220" />
      {[min,max].map(h=><g key={h}><path d={`M46 ${y(h)}H${end}`} stroke="#81946a" strokeOpacity=".2" /><text x="37" y={y(h)+4} textAnchor="end">{h.toFixed(1)}m</text></g>)}
      <path d={path} fill="none" stroke="#cbb37a" strokeWidth="2" />
      <path d={`M${x(state.distance)} 19V${baseline}`} stroke="#e9d7aa" strokeWidth="1" strokeDasharray="3 4" />
      <circle cx={x(state.distance)} cy={y(current.height)} r="4" fill="#ead6a2" />
      {(width<500?[0,.5,1]:[0,.25,.5,.75,1]).map(f=><g key={f}><path d={`M${x(f*simulation.distance)} ${baseline}v4`} stroke="#8b9d79" /><text x={x(f*simulation.distance)} y={compact?78:126} textAnchor="middle">{Math.round(f*simulation.distance).toLocaleString()}m</text></g>)}
      <text x={end} y="17" textAnchor="end">ゴール基準の相対高低</text>
    </svg>:<div className="sim-elevation-unknown">公式公表の高低差は{course.elevation_m}m。坂の位置・勾配を確認できていないため、起伏の描画と速度・消耗への反映を保留しています。</div>}
    <p>{simulation.terrainEnabled&&current.known?"上りは減速・消耗増、下りは加速・消耗軽減。反応係数は展示用の仮定です。":"坂の速度・消耗効果は適用していません。"}</p>
  </section>;
}
