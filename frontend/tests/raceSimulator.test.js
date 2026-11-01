import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { BASELINE_PARAMETERS, buildProfiles, sampleSimulation, simulateRace } from "../src/lib/raceSimulator.js";

const demo = JSON.parse(readFileSync(new URL("../src/fixtures/raceSimulatorDemo.json", import.meta.url), "utf8"));
const race = demo.races[0];

test("all horses finish in every demo race; distance and energy stay valid", () => {
  for (const r of demo.races) {
    for (const pace of ["slow", "average", "fast"]) {
      const sim = simulateRace(r, { seed: 42, pace });
      assert.equal(sim.finishOrder.length, r.horses.length);
      assert.ok(sim.finishOrder.every((h) => Number.isFinite(h.finishTime) && h.finishTime > 0));
      for (let i = 0; i < sim.snapshots.length; i += 1) {
        const snapshot = sim.snapshots[i];
        for (let j = 0; j < snapshot.horses.length; j += 1) {
          const h = snapshot.horses[j];
          assert.ok(h.distance >= 0 && h.distance <= r.race_info.distance);
          assert.ok(h.energy >= 0 && h.energy <= 1);
          if (i) assert.ok(h.distance >= sim.snapshots[i - 1].horses[j].distance);
        }
      }
      assert.equal(sampleSimulation(sim, sim.duration).order[0].horse_no, sim.finishOrder[0].horse_no);
    }
  }
});

test("replaying the same seed is exact; a new seed or pace changes the trajectory", () => {
  const first = simulateRace(race, { seed: 88 });
  assert.deepEqual(simulateRace(race, { seed: 88 }), first);
  assert.notDeepEqual(simulateRace(race, { seed: 89 }).snapshots[80], first.snapshots[80]);
  assert.notDeepEqual(simulateRace(race, { seed: 88, pace: "fast" }).snapshots[80], first.snapshots[80]);
});

test("winner and final order are not fixed across seeds", () => {
  const winners = new Set(Array.from({ length: 24 }, (_, seed) => simulateRace(race, { seed, variation: 1.7 }).finishOrder[0].horse_no));
  assert.ok(winners.size > 1);
});

test("results, odds and popularity cannot affect profiles or simulations", () => {
  const polluted = structuredClone(race);
  polluted.actual_result = { winner: 16 };
  polluted.result = { top3: [16, 15, 14] };
  for (const h of polluted.horses) {
    h.odds = h.horse_no === 16 ? 1.01 : 999;
    h.popularity = 17 - h.horse_no;
    for (const run of h.recent_runs) { run.odds = 1; run.popularity = 1; }
  }
  const a = simulateRace(race, { seed: 7 });
  const b = simulateRace(polluted, { seed: 7 });
  assert.deepEqual(b.snapshots, a.snapshots);
  assert.deepEqual(b.finishOrder, a.finishOrder);
});

test("missing history uses stated defaults; same-day/future runs are excluded", () => {
  const copy = structuredClone(race);
  const noHistory = copy.horses[0];
  noHistory.recent_runs = [];
  noHistory.running_style = "不明";
  const profile = buildProfiles(copy)[0];
  assert.equal(profile.ability, 0.5);
  assert.equal(profile.early, 0.5);
  assert.equal(profile.closing, 0.5);
  assert.equal(profile.stamina, 0.5);
  assert.equal(profile.assumptions.length, 4);
  noHistory.recent_runs.push({ date: copy.race_info.race_date, finish: 1, field_size: 16, final_3f: 30 });
  noHistory.recent_runs.push({ date: "2099-01-01", finish: 1, field_size: 16, final_3f: 30 });
  assert.equal(buildProfiles(copy)[0].runCount, 0);
});

test("seeking clamps bounds and does not reveal a finish before its timestamp", () => {
  const sim = simulateRace(race);
  assert.equal(sampleSimulation(sim, -10).time, 0);
  assert.equal(sampleSimulation(sim, 999).time, sim.duration);
  assert.ok(sampleSimulation(sim, sim.finishOrder[0].finishTime - 0.01).horses.every((h) => h.finishTime === null));
  assert.equal(sampleSimulation(sim, sim.finishOrder[0].finishTime + 0.01).order[0].horse_no, sim.finishOrder[0].horse_no);
});

test("fixture contains no market features, outcomes, or runs on/after target day", () => {
  assert.doesNotMatch(JSON.stringify(demo), /"(odds|popularity|result|finish_order|horse_scores)"/);
  for (const r of demo.races) {
    for (const h of r.horses) assert.ok(h.recent_runs.every((run) => run.date < r.race_info.race_date));
  }
});

test("calibration summary and playback use identical dynamics and last600 timestamps", () => {
  for (const r of demo.races) {
    for (const parameters of [BASELINE_PARAMETERS, { ...BASELINE_PARAMETERS, dirtSpeed: 14.8, kickClosing: 0.4, energyDrain: 0.85 }]) {
      const playback = simulateRace(r, { variation: 0, parameters });
      const summary = simulateRace(r, { variation: 0, parameters, recordSnapshots: false });
      assert.deepEqual(summary.finishOrder, playback.finishOrder);
      for (const h of summary.finishOrder) {
        assert.ok(h.closingStartTime > 0 && h.closingStartTime < h.finishTime);
        assert.ok(h.finishTime - h.closingStartTime > 20);
      }
    }
  }
});
