import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { COURSE_PROFILES, resolveRaceCourse, terrainAt, terrainResponse, coursePoint, courseGeometryPath } from "../src/lib/raceCourses.js";
import { simulateRace } from "../src/lib/raceSimulator.js";

const profile=id=>COURSE_PROFILES.find(p=>p.id===id);
const race=JSON.parse(readFileSync(new URL("../src/fixtures/raceSimulatorDemo.json",import.meta.url),"utf8")).races[0];

test("official catalog separates 44 layouts across 10 central and 14 local venues",()=>{
  assert.equal(new Set(COURSE_PROFILES.map(p=>p.id)).size,44);
  for(const [group,count] of [['central',10],['local',14]])assert.equal(new Set(COURSE_PROFILES.filter(p=>p.group===group).map(p=>p.course)).size,count);
  assert.equal(profile('niigata-turf-outer').straight_m,658.7);
  assert.equal(profile('niigata-turf-outer').elevation_m,2.2);
  assert.equal(profile('nakayama-turf-inner').elevation_m,5.3);
  assert.equal(profile('kyoto-turf-outer').elevation_m,4.3);
  assert.equal(profile('morioka-dirt-standard').elevation_m,4.4);
  for(const p of COURSE_PROFILES){
    assert.ok(p.source_url.startsWith('https://'));
    assert.ok(p.plan_source_url.startsWith('https://'));
    assert.ok(p.plan_points.length>=2);
    assert.ok(p.distances_m.length>0);
    if(p.direction!=='straight'){
      const area=p.plan_points.reduce((sum,a,i)=>{const b=p.plan_points[(i+1)%p.plan_points.length];return sum+a[0]*b[1]-b[0]*a[1];},0);
      assert.equal(Math.sign(area),p.direction==='right'?1:-1,`${p.id} plan must follow its official direction`);
    }
    if(p.elevation_points){
      const heights=p.elevation_points.map(v=>v[1]);
      assert.equal(p.elevation_points[0][0],0);
      assert.equal(p.elevation_points.at(-1)[0],p.lap_m||1000);
      assert.equal(p.elevation_points[0][1],0);
      for(let i=1;i<p.elevation_points.length;i++)assert.ok(p.elevation_points[i][0]>p.elevation_points[i-1][0]);
      if(p.elevation_m>0)assert.ok(Math.abs(Math.max(...heights)-Math.min(...heights)-p.elevation_m)<.08);
    }
  }
});

test("ambiguous inner/outer layouts stay explicit instead of averaging tracks",()=>{
  const info={course:'新潟',surface:'芝',distance:2000};
  const unresolved=resolveRaceCourse(info);
  assert.equal(unresolved.ambiguous,true);
  assert.equal(unresolved.course.id,'niigata-turf-outer');
  assert.equal(resolveRaceCourse({...info,course_layout:'inner'}).course.id,'niigata-turf-inner');
  assert.equal(resolveRaceCourse({course:'未知',surface:'芝',distance:2000}).course,null);
  assert.throws(()=>resolveRaceCourse(info,'missing-profile'));
});

test("digitized hills occur in their documented sections and have the correct sign",()=>{
  assert.ok(terrainAt(profile('nakayama-turf-inner'),1850,2000).grade>.01,'Nakayama finish climb');
  assert.ok(terrainAt(profile('kyoto-turf-outer'),950,2000).grade>0,'Kyoto approach to third-corner hill');
  assert.ok(terrainAt(profile('kyoto-turf-outer'),1350,2000).grade<0,'Kyoto descent after hill');
  assert.ok(terrainAt(profile('morioka-dirt-standard'),1450,1600).grade>0,'Morioka finish climb');
  const lap=profile('nakayama-turf-inner').lap_m;
  const a=terrainAt(profile('nakayama-turf-inner'),100,3600),b=terrainAt(profile('nakayama-turf-inner'),100+lap,3600);
  assert.ok(Math.abs(a.height-b.height)<1e-8);
});

test("unknown hill locations remain unknown, while official flat courses stay neutral",()=>{
  const unknown=terrainAt(profile('kochi-dirt-standard'),500,1400);
  assert.equal(unknown.known,false);
  assert.equal(unknown.height,null);
  assert.deepEqual(terrainResponse(unknown),{speed:0,energy:1});
  const flat=terrainAt(profile('kawasaki-dirt-standard'),500,1400);
  assert.equal(flat.known,true);
  assert.equal(flat.grade,0);
  const climb=terrainResponse({known:true,grade:.02});
  assert.ok(climb.speed<0&&climb.energy>1);
  const descent=terrainResponse({known:true,grade:-.02});
  assert.ok(descent.speed>0&&descent.energy<1);
});

test("physical course progress closes at the goal and repeats on additional laps",()=>{
  for(const p of COURSE_PROFILES){
    const distance=p.direction==='straight'?1000:p.distances_m[0];
    const goal=coursePoint(p,distance,distance,3);
    const expected=p.direction==='straight'?p.plan_points.at(-1):p.plan_points[0];
    assert.ok(Math.hypot(goal.x-expected[0],goal.y-expected[1])<.001);
    assert.ok(!courseGeometryPath(p,distance).includes('NaN'));
    if(p.lap_m){
      const a=coursePoint(p,40,distance+p.lap_m,3),b=coursePoint(p,40+p.lap_m,distance+p.lap_m,3);
      assert.ok(Math.hypot(a.x-b.x,a.y-b.y)<1e-6);
    }
  }
});

test("hill response changes a run without changing its source inputs; flat and unknown courses remain neutral",()=>{
  for(const id of ['nakayama-turf-inner','kawasaki-dirt-standard','kochi-dirt-standard']){
    const course=profile(id);
    const input={...race,race_info:{...race.race_info,course:course.course,surface:course.surface,course_layout:course.layout,distance:course.distances_m[0]}};
    const original=structuredClone(input);
    const on=simulateRace(input,{courseProfileId:id,terrainEnabled:true,seed:7});
    const off=simulateRace(input,{courseProfileId:id,terrainEnabled:false,seed:7});
    assert.deepEqual(input,original);
    assert.ok(on.finishOrder.every(h=>Number.isFinite(h.finishTime)&&h.energy>=0&&h.energy<=1));
    if(id==='nakayama-turf-inner')assert.notDeepEqual(on.finishOrder,off.finishOrder);
    else assert.deepEqual(on.finishOrder,off.finishOrder);
  }
});
