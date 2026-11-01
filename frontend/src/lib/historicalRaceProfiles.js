// Historical features use strictly prior observations. References are fitted
// on development races only and travel with the frozen simulation model.
const clamp = (x,a=0,b=1) => Math.max(a,Math.min(b,x));
const number = x => typeof x === "number" && Number.isFinite(x);
const wet = condition => ({"良":0,"稍":1,"稍重":1,"重":2,"不良":3}[condition] ?? 0);
export function referenceFeatures(info, courses=[]) {
  const local=info.source === "local" || String(info.scope_key || "").startsWith("local") || !["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"].includes(info.course);
  return [1, +(info.surface === "芝"), +local, Math.log(info.distance/1600), wet(info.track_condition), ...courses.map(c => +(info.course===c))];
}
function reference(info, spec, fallback) {
  if (!spec) return fallback;
  return referenceFeatures(info,spec.courses).reduce((sum,v,i)=>sum+v*spec.coefficients[i],0);
}
export function buildHistoricalProfiles(race, model) {
  const info=race.race_info, features=model.features || {}, prior=features.priorCount ?? 3;
  const decay=features.recencyDays ?? 120, distanceScale=features.distanceScale ?? .35;
  const coeff=model.parameters || {};
  return race.horses.map((horse,index)=>{
    const runs=(horse.recent_runs || []).filter(r=>r.date && r.date<info.race_date && r.race_id!==info.race_id);
    const values={form:[],early:[],closing:[],gain:[],endurance:[],speed:[]};
    const add=(key,value,weight)=>{if(number(value)&&weight>0)values[key].push([value,weight]);};
    for (const run of runs) {
      const days=(Date.parse(info.race_date)-Date.parse(run.date))/86400000;
      if (!number(days) || days<=0 || !number(run.distance) || run.distance<=0)continue;
      const same=run.surface===info.surface;
      const weight=Math.exp(-days/decay)*Math.exp(-Math.abs(Math.log(run.distance/info.distance))/distanceScale)*(same?1:.2);
      const validFinish=number(run.finish)&&number(run.field_size)&&run.field_size>1&&run.finish>=1&&run.finish<=run.field_size;
      if (validFinish)add("form",1-(run.finish-1)/(run.field_size-1),weight);
      const corners=(run.corner_positions || []).filter(p=>number(p)&&p>=1&&p<=run.field_size);
      if (corners.length&&run.field_size>1)add("early",1-(corners[0]-1)/(run.field_size-1),weight);
      const closing=number(run.final_3f)&&run.final_3f>25&&run.final_3f<65;
      const context={...run};
      const refSpeed=clamp(reference(context,model.references?.speed,run.surface==="芝"?16.2:15.2),10,20);
      const refClosing=clamp(reference(context,model.references?.closing,run.surface==="芝"?35.5:39.5),28,55);
      if (same&&closing)add("closing",clamp(.5+(refClosing-run.final_3f)/8),weight);
      if (same&&validFinish&&corners.length&&run.distance>=info.distance*.85)add("gain",(corners.at(-1)-run.finish)/(run.field_size-1),weight);
      if (same&&number(run.time_seconds)&&run.time_seconds>0&&run.distance/run.time_seconds>7&&run.distance/run.time_seconds<22) {
        const residual=clamp(run.distance/run.time_seconds-refSpeed,-2,2);
        add("speed",residual,weight);
        if (closing&&run.distance>800&&run.time_seconds>run.final_3f+10&&run.distance>=info.distance*.85) {
          const expectedTime=run.distance/refSpeed;
          if (expectedTime>refClosing+10) {
            const observed=600/run.final_3f-(run.distance-600)/(run.time_seconds-run.final_3f);
            const expected=600/refClosing-(run.distance-600)/(expectedTime-refClosing);
            add("endurance",clamp((observed-expected)/2,-1,1),weight);
          }
        }
      }
    }
    const estimate=(key,fallback=0)=>{
      const rows=values[key],weight=rows.reduce((s,r)=>s+r[1],0);
      return {value:(rows.reduce((s,r)=>s+r[0]*r[1],0)+fallback*prior)/(weight+prior),weight,count:rows.length};
    };
    const form=estimate("form",.5),early=estimate("early",.5),closing=estimate("closing",.5),gain=estimate("gain"),endurance=estimate("endurance"),speed=estimate("speed");
    const speedMean=values.speed.reduce((s,r)=>s+r[0]*r[1],0)/Math.max(speed.weight,1e-9);
    const variance=values.speed.reduce((s,r)=>s+r[1]*(r[0]-speedMean)**2,0)/Math.max(speed.weight,1e-9);
    const measuredVariance=speed.count>=2?variance*speed.weight/(speed.weight+prior):0;
    const stamina=clamp(.5+(coeff.staminaGainWeight ?? .4)*gain.value+(coeff.staminaDecayWeight ?? .4)*endurance.value);
    const style=early.count?early.value>.8?"逃げ":early.value>.6?"先行":early.value>.35?"差し":"追込":"不明";
    const assumptions=[];
    if(!form.count)assumptions.push("近走成績がなく、基礎能力は中間値");
    if(!early.count)assumptions.push("通過順がなく、序盤の位置取りは中間値");
    if(!closing.count)assumptions.push("同条件の上がりが不足し、末脚は中間値");
    if(!gain.count&&!endurance.count)assumptions.push("持続力の資料が不足し、中間値");
    return {...horse,ability:form.value,early:early.value,closing:closing.value,stamina,positionGain:gain.value,enduranceSpeed:endurance.value,historicalSpeed:speed.value,performanceVariance:measuredVariance,estimationVariance:1/(speed.weight+prior),style,uncertainty:1/(form.weight+1),consistency:clamp(1-Math.sqrt(measuredVariance)),assumptions,runCount:runs.length,closingCount:closing.count,speedCount:speed.count,frame:number(horse.frame_no)&&horse.frame_no>=1&&horse.frame_no<=8?horse.frame_no:Math.min(8,Math.floor(index*8/race.horses.length)+1)};
  });
}

export function raceNoise(profiles, model, random, variation=1) {
  const spec=model.randomness;
  const triangular=()=> (random()+random()-1)*Math.sqrt(6);
  const paceOffset=triangular()*(spec.paceSigma ?? 0)*variation;
  const horses=profiles.map(h=>{
    const sigma=Math.sqrt(Math.max(.0001,spec.dayVarianceFloor+spec.performanceVarianceWeight*h.performanceVariance+spec.estimateVarianceWeight*h.estimationVariance))*(spec.formScale ?? 1);
    return {form:triangular()*sigma*variation,startDelay:.15+random()*(spec.startSigma ?? .12)*variation,rhythm:random()*Math.PI*2,rhythmAmplitude:(spec.rhythmSigma ?? .04)*variation};
  });
  return {paceOffset,horses};
}
