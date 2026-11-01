import {readFileSync,writeFileSync} from "node:fs";
import {buildHistoricalProfiles} from "../frontend/src/lib/historicalRaceProfiles.js";
import {buildProfiles,simulateRace} from "../frontend/src/lib/raceSimulator.js";
import {resolveRaceCourse,terrainAt,courseSection} from "../frontend/src/lib/raceCourses.js";
const [mode,casesFile,output,modelFile,seedsFile]=process.argv.slice(2);
const cases=JSON.parse(readFileSync(casesFile,"utf8"));
const model=modelFile?JSON.parse(readFileSync(modelFile,"utf8")):null;
if(mode==="profiles") {
  const fields=["horse_no","ability","early","closing","stamina","style","uncertainty","positionGain","enduranceSpeed","historicalSpeed","performanceVariance","estimationVariance","speedCount"];
  writeFileSync(output,JSON.stringify(cases.map(c=>(model?buildHistoricalProfiles(c.race,model):buildProfiles(c.race)).map(h=>Object.fromEntries(fields.map(k=>[k,h[k]??0]))))));
} else if(mode==="terrain") {
  const terrain={},keys=[];
  for(const c of cases){const info=c.race.race_info,{course}=resolveRaceCourse(info),key=`${course?.id||"unknown"}:${info.distance}`;keys.push(key);if(terrain[key])continue;terrain[key]=Array.from({length:info.distance+2},(_,d)=>{const p=d/info.distance,t=terrainAt(course,d,info.distance);return[t.grade,+(course?courseSection(course,d,info.distance).isBend:(p>.2&&p<.43)||(p>.6&&p<.8))];});}
  writeFileSync(output,JSON.stringify({terrain,keys}));
} else if(mode==="evaluate") {
  const seeds=seedsFile?JSON.parse(readFileSync(seedsFile,"utf8")):[null];
  const rows=cases.map((c,i)=>{
    const horses=new Map(c.race.horses.map(h=>[h.horse_no,{horse_no:h.horse_no,time:0,final_3f:0,win_count:0,top3_count:0}]));
    const runs=[];
    for(const seed of seeds) {
      const sim=simulateRace(c.race,{model:model?.references?model:null,parameters:model?.references?null:model?.parameters,variation:seed===null?0:1,seed:(seed??20261101)^Number(c.race.race_info.race_id),recordSnapshots:false});
      sim.finishOrder.forEach((h,rank)=>{const v=horses.get(h.horse_no);v.time+=h.finishTime/seeds.length;v.final_3f+=(h.finishTime-h.closingStartTime)/seeds.length;v.win_count+=rank===0?1:0;v.top3_count+=rank<3?1:0;});
      runs.push(sim.finishOrder.map(h=>({horse_no:h.horse_no,time:h.finishTime,final_3f:h.finishTime-h.closingStartTime})));
    }
    if(i%500===0)process.stderr.write(`原版模拟 ${i}/${cases.length} × ${seeds.length}\n`);
    return {race_id:c.race.race_info.race_id,predictions:[...horses.values()],runs,seeds:seeds.length};
  });writeFileSync(output,JSON.stringify(rows));
} else throw new Error("Unknown mode");
