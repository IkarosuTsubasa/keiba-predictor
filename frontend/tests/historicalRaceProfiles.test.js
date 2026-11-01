import test from "node:test";
import assert from "node:assert/strict";
import {buildHistoricalProfiles} from "../src/lib/historicalRaceProfiles.js";
import {simulateRace,BASELINE_PARAMETERS} from "../src/lib/raceSimulator.js";

const model={version:"test-historical",features:{priorCount:3,recencyDays:120,distanceScale:.35},references:{speed:{courses:[],coefficients:[16,0,0,0,0]},closing:{courses:[],coefficients:[38,0,0,0,0]}},parameters:{...BASELINE_PARAMETERS,historicalSpeedWeight:.7,staminaGainWeight:.4,staminaDecayWeight:.4},randomness:{dayVarianceFloor:.01,performanceVarianceWeight:.1,estimateVarianceWeight:.01,paceSigma:.1,startSigma:.12,rhythmSigma:.04,formScale:1}};
const race={race_info:{race_id:"202601010101",race_date:"2026-11-01",course:"東京",surface:"ダート",source:"central",distance:1600},horses:Array.from({length:6},(_,i)=>({horse_no:i+1,horse_name:`馬${i+1}`,recent_runs:[{race_id:"202601010001",date:"2026-10-01",course:"東京",surface:"ダート",distance:1600,track_condition:"良",finish:3,field_size:10,corner_positions:[7,7],time_seconds:100,final_3f:38+i}]}))};

test("持久力与历史名次分离，实测用时与末段分别产生特征",()=>{
  const p=buildHistoricalProfiles(race,model);
  assert.equal(p[0].ability,p[5].ability);
  assert.notEqual(p[0].closing,p[5].closing);
  assert.notEqual(p[0].stamina,p[5].stamina);
  const changed=structuredClone(race);changed.horses[0].recent_runs[0].time_seconds=105;
  assert.notEqual(buildHistoricalProfiles(changed,model)[0].historicalSpeed,p[0].historicalSpeed);
});
test("目标赛果、行情、当天或未来记录不会改变优化模拟",()=>{
  const dirty=structuredClone(race);dirty.result={winner:6};
  for(const h of dirty.horses){h.odds=1;h.popularity=1;h.running_style="逃げ";for(const run of h.recent_runs){run.odds=1;run.popularity=1;}h.recent_runs.push({...h.recent_runs[0],date:race.race_info.race_date,race_id:race.race_info.race_id,time_seconds:70});h.recent_runs.push({...h.recent_runs[0],date:"2099-01-01",time_seconds:70});}
  assert.deepEqual(simulateRace(dirty,{model,seed:42}).snapshots,simulateRace(race,{model,seed:42}).snapshots);
});
test("优化模型种子可重放，摘要与播放相同，体力与距离保持有效",()=>{
  const sim=simulateRace(race,{model,seed:9});assert.deepEqual(simulateRace(race,{model,seed:9}),sim);
  assert.deepEqual(simulateRace(race,{model,seed:9,recordSnapshots:false}).finishOrder,sim.finishOrder);
  assert.notDeepEqual(simulateRace(race,{model,seed:10}).finishOrder,sim.finishOrder);
  for(const frame of sim.snapshots)for(const h of frame.horses){assert.ok(h.energy>=0&&h.energy<=1);assert.ok(h.distance>=0&&h.distance<=1600);}
});
