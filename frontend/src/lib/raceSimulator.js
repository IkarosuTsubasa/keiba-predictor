// An illustrative, seeded race model. No results, market features, or fixed finish order.
import { courseSection, resolveRaceCourse, terrainAt, terrainResponse } from "./raceCourses.js";
import { buildHistoricalProfiles, raceNoise } from "./historicalRaceProfiles.js";

export const MODEL_VERSION = "illustrative-terrain-v2";
export const STEP = 0.2;
// Frozen prototype baseline. Calibration passes explicit coefficients; the
// production Agent scorer and saved predictions are independent of this model.
export const BASELINE_PARAMETERS = Object.freeze({
  turfSpeed: 16.8, dirtSpeed: 15.8, localOffset: 0, distanceSlope: 0,
  abilitySpeed: 1.2, earlySpeed: 1.5, kickBase: 0.55, kickClosing: 1.9,
  kickStyle: 0.45, energyDrain: 0.67, staminaSaving: 0.3, fatigueScale: 5.2,
});
export const PACES = {
  slow: { label: "スロー", offset: -0.75 },
  average: { label: "ミドル", offset: 0 },
  fast: { label: "ハイ", offset: 0.85 },
};
export const FRAME_COLORS = ["#f2ede3", "#344052", "#ed6865", "#5797e6", "#edc85a", "#58b385", "#e9954f", "#e58fb4"];
const clamp = (x, min = 0, max = 1) => Math.min(max, Math.max(min, x));
const mean = (items, fallback = 0.5) => items.length ? items.reduce((a, b) => a + b, 0) / items.length : fallback;
const valid = (x) => typeof x === "number" && Number.isFinite(x);

export function seededRandom(seed) {
  let state = Number(seed) >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildProfiles(race) {
  const info = race.race_info;
  // Restrict raw closing-time comparisons to similar distance AND the same surface.
  const comparable = (run) => run.surface === info.surface && valid(run.distance)
    && Math.abs(run.distance - info.distance) <= Math.max(200, info.distance * 0.2);
  const eligibleRuns = (h) => (h.recent_runs || []).filter((r) => r.date && r.date < info.race_date);
  const closingPool = race.horses.flatMap((h) => eligibleRuns(h).filter(comparable)
    .filter((r) => valid(r.final_3f) && r.final_3f > 25 && r.final_3f < 65).map((r) => r.final_3f));
  return race.horses.map((horse, index) => {
    const runs = eligibleRuns(horse);
    const results = runs.filter((r) => valid(r.finish) && valid(r.field_size) && r.finish >= 1 && r.finish <= r.field_size && r.field_size > 1);
    const ability = mean(results.map((r) => 1 - (r.finish - 1) / (r.field_size - 1)));
    const starts = runs.filter((r) => r.corner_positions?.length && valid(r.field_size) && r.field_size > 1);
    const early = mean(starts.map((r) => clamp(1 - (r.corner_positions[0] - 1) / (r.field_size - 1))));
    const closeRuns = runs.filter(comparable).filter((r) => valid(r.final_3f) && r.final_3f > 25 && r.final_3f < 65);
    const closing = closeRuns.length && closingPool.length >= 4
      ? mean(closeRuns.map((r) => (closingPool.filter((v) => v > r.final_3f).length + closingPool.filter((v) => v === r.final_3f).length / 2) / closingPool.length))
      : 0.5;
    const distanceRuns = results.filter((r) => r.surface === info.surface && r.distance >= info.distance * 0.9);
    const stamina = mean(distanceRuns.map((r) => 1 - (r.finish - 1) / (r.field_size - 1)));
    const measuredStyle = early > 0.8 ? "逃げ" : early > 0.6 ? "先行" : early > 0.35 ? "差し" : "追込";
    const style = ["逃げ", "先行", "差し", "追込"].includes(horse.running_style) ? horse.running_style : starts.length ? measuredStyle : "不明";
    const variance = results.length > 1 ? Math.sqrt(mean(results.map((r) => (1 - (r.finish - 1) / (r.field_size - 1) - ability) ** 2), 0)) : 0.3;
    const assumptions = [];
    if (!results.length) assumptions.push("近走成績がなく、基礎能力は中間値");
    if (!starts.length) assumptions.push("通過順がなく、序盤の位置取りは中間値");
    if (!closeRuns.length || closingPool.length < 4) assumptions.push("同条件の上がりが不足し、末脚は中間値");
    if (!distanceRuns.length) assumptions.push("同馬場・近い距離の成績がなく、持続力は中間値");
    return {
      ...horse, style, ability, early, closing, stamina,
      consistency: clamp(1 - variance * 2), uncertainty: clamp(0.18 + assumptions.length * 0.14 + 0.35 / (runs.length + 1)),
      assumptions, runCount: runs.length, closingCount: closeRuns.length,
      frame: valid(horse.frame_no) && horse.frame_no >= 1 && horse.frame_no <= 8 ? horse.frame_no : Math.min(8, Math.floor(index * 8 / race.horses.length) + 1),
    };
  });
}

function orderHorses(horses) {
  return [...horses].sort((a, b) => {
    if (a.finishTime !== null && b.finishTime !== null) return a.finishTime - b.finishTime || a.horse_no - b.horse_no;
    if (a.finishTime !== null) return -1;
    if (b.finishTime !== null) return 1;
    return b.distance - a.distance || a.horse_no - b.horse_no;
  });
}

export function simulateRace(race, { seed = 20261101, pace = "average", variation = 1, courseProfileId = null, terrainEnabled = true, parameters = null, model = null, recordSnapshots = true, preparedProfiles = null } = {}) {
  if (!race?.horses?.length || !valid(race.race_info?.distance) || race.race_info.distance <= 0) throw new Error("Invalid race input");
  const coefficients = { ...BASELINE_PARAMETERS, ...model?.parameters, ...parameters };
  const profiles = preparedProfiles || (model ? buildHistoricalProfiles(race,{...model,parameters:coefficients}) : buildProfiles(race));
  const random = seededRandom(seed);
  const length = race.race_info.distance;
  const courseResolution = resolveRaceCourse(race.race_info, courseProfileId);
  const course = courseResolution.course;
  const local = race.race_info.source === "local" || String(race.race_info.scope_key || "").startsWith("local");
  const baseSpeed = (race.race_info.surface === "芝" ? coefficients.turfSpeed : coefficients.dirtSpeed)
    + (local ? coefficients.localOffset : 0) + coefficients.distanceSlope * Math.log(length / 1600);
  const nominalDuration = length / baseSpeed;
  const noise=model ? raceNoise(profiles,model,random,variation) : null;
  const paceOffset = (PACES[pace] || PACES.average).offset+(noise?.paceOffset || 0);
  const frontCount = profiles.filter((h) => h.style === "逃げ" || h.style === "先行").length;
  const pressure = clamp((frontCount - 2) / 6, 0, 0.8);
  let states = profiles.map((h, index) => ({
    horse_no: h.horse_no, distance: 0, speed: 0, energy: 1,
    lane: 0.7 + index * 6.5 / Math.max(1, profiles.length - 1), finishTime: null, closingStartTime: null,
    form: noise?noise.horses[index].form:(random() + random() - 1) * (0.25 + h.uncertainty * 1.5) * variation,
    startDelay: noise?noise.horses[index].startDelay:0.15 + random() * (0.3 + h.uncertainty * 1.1) * variation,
    rhythm: noise?noise.horses[index].rhythm:random() * Math.PI * 2,
    rhythmAmplitude: noise?noise.horses[index].rhythmAmplitude:.14*variation,
  }));
  const snapshots = [];
  const snapshot = (time) => ({ time, horses: states.map(({ form, startDelay, rhythm, ...s }) => ({ ...s })) });
  if (recordSnapshots) snapshots.push(snapshot(0));
  let time = 0;
  for (let tick = 1; tick < 6000; tick += 1) {
    time = tick * STEP;
    const leaderDistance = Math.max(...states.map((h) => h.distance));
    states = states.map((state, index) => {
      if (state.finishTime !== null) return state;
      const h = profiles[index];
      if (time < state.startDelay) return state;
      const progress = state.distance / length;
      const terrain = terrainAt(course, state.distance, length);
      const terrainEffect = terrainResponse(terrain, terrainEnabled);
      const late = clamp((progress - 0.66) / 0.25);
      const earlyPhase = 1 - clamp(progress / 0.42);
      const frontStyle = h.style === "逃げ" ? 1.15 : h.style === "先行" ? 0.7 : h.style === "追込" ? -0.6 : -0.15;
      const earlyDrive = (frontStyle + (h.early - 0.5) * coefficients.earlySpeed) * earlyPhase;
      const raceFlow = (paceOffset + pressure * 0.55) * (1 - late * 0.75);
      const kick = late * (coefficients.kickBase + h.closing * coefficients.kickClosing + (h.style === "追込" || h.style === "差し" ? coefficients.kickStyle : 0));
      const fatigue = Math.max(0, 0.36 - state.energy) * (coefficients.fatigueScale - h.stamina * 2);
      const laneTarget = progress < 0.12 ? state.lane : (h.style === "逃げ" ? 0.85 : 1.3 + (index % 4) * 0.65) + late * (h.style === "追込" || h.style === "差し" ? 1.2 : 0.15);
      const lane = state.lane + clamp(laneTarget - state.lane, -0.35 * STEP, 0.35 * STEP);
      const onBend = course ? courseSection(course, state.distance, length).isBend : (progress > 0.2 && progress < 0.43) || (progress > 0.6 && progress < 0.8);
      const outerCost = onBend ? lane * 0.055 : 0;
      const rhythm = Math.sin(time * 0.32 + state.rhythm) * state.rhythmAmplitude;
      const historySpeed=model?(h.historicalSpeed || 0)*(coefficients.historicalSpeedWeight ?? 0):0;
      const target = baseSpeed + (h.ability - 0.5) * coefficients.abilitySpeed + historySpeed + state.form + earlyDrive + raceFlow + kick - fatigue - outerCost + rhythm + terrainEffect.speed;
      const acceleration = time < 6 ? 2.8 + h.early : 0.85;
      const speed = clamp(state.speed + clamp(target - state.speed, -1.5 * STEP, acceleration * STEP), 0, 23);
      const effort = Math.max(0.6, speed / baseSpeed);
      const leadEffort = state.distance >= leaderDistance - 4 ? pressure * 0.15 : 0;
      const drain = (coefficients.energyDrain / nominalDuration) * effort ** 3.2 * (1.14 - h.stamina * coefficients.staminaSaving + leadEffort) * terrainEffect.energy;
      const energy = clamp(state.energy - drain * STEP);
      const distance = state.distance + speed * STEP;
      const finishTime = distance >= length ? time - STEP + (length - state.distance) / Math.max(speed, 0.01) : null;
      const closingStartTime = state.closingStartTime ?? (distance >= length - 600 ? time - STEP + (length - 600 - state.distance) / Math.max(speed, 0.01) : null);
      return { ...state, lane, speed, energy, distance: Math.min(length, distance), finishTime, closingStartTime };
    });
    if (recordSnapshots && (tick % 2 === 0 || states.every((h) => h.finishTime !== null))) snapshots.push(snapshot(time));
    if (states.every((h) => h.finishTime !== null)) break;
  }
  const finishOrder = orderHorses(states);
  if (!recordSnapshots) return { finishOrder, profiles, duration: time, coefficients, version: model?.version || MODEL_VERSION };
  const events = [{ time: 0, text: `${profiles.length}頭がゲートイン。${PACES[pace]?.label || "ミドル"}の展開を試走します。`, kind: "start" }];
  let lastLeader = null;
  let lastLeadEvent = 0;
  let closingEvent = false;
  for (const frame of snapshots) {
    const leader = orderHorses(frame.horses)[0];
    if (frame.time > 8 && leader.horse_no !== lastLeader && frame.time - lastLeadEvent > 12 && leader.finishTime === null) {
      events.push({ time: frame.time, text: `${leader.horse_no}番が先頭へ。${profiles.find((h) => h.horse_no === leader.horse_no).horse_name}がレースを引っ張ります。`, kind: "lead" });
      lastLeadEvent = frame.time;
    }
    lastLeader = leader.horse_no;
    if (!closingEvent && leader.distance >= Math.max(length * 0.5, length - 600)) {
      events.push({ time: frame.time, text: `先頭は残り${Math.round(length - leader.distance)}m。各馬の末脚と前半の消耗が効く局面へ。`, kind: "closing" });
      closingEvent = true;
    }
  }
  const winner = finishOrder[0];
  events.push({ time: winner.finishTime, text: `${winner.horse_no}番 ${profiles.find((h) => h.horse_no === winner.horse_no).horse_name}が先着。この条件での模擬結果です。`, kind: "finish" });
  events.sort((a, b) => a.time - b.time);
  return { seed, pace, variation, profiles, snapshots, finishOrder, events, duration: time, distance: length, version: model?.version || MODEL_VERSION, course, courseAmbiguous: courseResolution.ambiguous, terrainEnabled };
}

export function sampleSimulation(simulation, time) {
  const frames = simulation.snapshots;
  const t = clamp(time, 0, simulation.duration);
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].time <= t) lo = mid; else hi = mid - 1;
  }
  const a = frames[lo];
  const b = frames[Math.min(lo + 1, frames.length - 1)];
  const mix = b.time === a.time ? 0 : (t - a.time) / (b.time - a.time);
  const horses = a.horses.map((h, i) => ({
    ...h, distance: h.distance + (b.horses[i].distance - h.distance) * mix,
    lane: h.lane + (b.horses[i].lane - h.lane) * mix,
    speed: h.speed + (b.horses[i].speed - h.speed) * mix,
    energy: h.energy + (b.horses[i].energy - h.energy) * mix,
    finishTime: simulation.finishOrder.find((f) => f.horse_no === h.horse_no).finishTime <= t
      ? simulation.finishOrder.find((f) => f.horse_no === h.horse_no).finishTime : null,
  }));
  return { time: t, horses, order: orderHorses(horses) };
}

export function formatTime(time) {
  const minutes = Math.floor(time / 60);
  return `${minutes}:${(time % 60).toFixed(1).padStart(4, "0")}`;
}
