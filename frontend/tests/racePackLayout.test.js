import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { layoutRacePack } from "../src/lib/racePackLayout.js";
import { simulateRace, sampleSimulation } from "../src/lib/raceSimulator.js";

function assertReadable(order, width) {
  const before = structuredClone(order);
  const { horses, height } = layoutRacePack(order, { width });
  assert.deepEqual(order, before, "layout must not change model positions");
  assert.equal(horses.length, order.length);
  assert.deepEqual(horses.map((h) => h.horse_no), order.map((h) => h.horse_no));
  for (let i = 0; i < horses.length; i += 1) {
    assert.ok(horses[i].x >= 29 && horses[i].x <= width - 29, `horse stays in view: ${horses[i].x}`);
    assert.ok(horses[i].y >= 26 && horses[i].y <= height - 36);
    if (i) assert.ok(horses[i].x <= horses[i - 1].x + 0.001, "display preserves race order");
    for (let j = i + 1; j < horses.length; j += 1) {
      assert.ok(Math.abs(horses[i].x - horses[j].x) >= 69.9 || Math.abs(horses[i].y - horses[j].y) >= 58, "horses and number badges do not overlap, including the selected outline");
    }
  }
}

test("a completely packed starting gate and finish remain readable", () => {
  for (const width of [320, 390, 560, 820]) {
    for (const count of [14, 16, 18]) {
      assertReadable(Array.from({ length: count }, (_, i) => ({ horse_no: i + 1, distance: 0 })), width);
    }
  }
});

test("all stages of the demo races fit desktop and mobile without overlaps", () => {
  const demo = JSON.parse(readFileSync(new URL("../src/fixtures/raceSimulatorDemo.json", import.meta.url), "utf8"));
  for (const race of demo.races) {
    const simulation = simulateRace(race);
    for (let time = 0; time <= simulation.duration; time += 4) {
      for (const width of [320, 390, 560, 820]) assertReadable(sampleSimulation(simulation, time).order, width);
    }
  }
});

test("selection-independent layout is deterministic and supports an empty field", () => {
  const order = [{ horse_no: 4, distance: 100 }, { horse_no: 1, distance: 100 }, { horse_no: 7, distance: 90 }];
  assert.deepEqual(layoutRacePack(order), layoutRacePack(order));
  assert.deepEqual(layoutRacePack([]).horses, []);
});
