// Reuses the actual frontend engine. No labels enter profiles or simulations.
import { readFileSync, writeFileSync } from "node:fs";
import { buildProfiles, simulateRace, BASELINE_PARAMETERS } from "../frontend/src/lib/raceSimulator.js";
import { resolveRaceCourse, terrainAt, courseSection } from "../frontend/src/lib/raceCourses.js";

const [mode, casesFile, outputFile, parameterFile] = process.argv.slice(2);
const cases = JSON.parse(readFileSync(casesFile, "utf8"));
if (mode === "prepare") {
  const terrainCache = new Map();
  const prepared = cases.map((c) => {
    const race = c.race, info = race.race_info;
    // The calibration input deliberately has no running_style: derive it from
    // strictly prior corners instead of relying on saved LLM analyses.
    const profiles = buildProfiles(race);
    const { course, ambiguous } = resolveRaceCourse(info);
    const key = `${course?.id || "unknown"}:${info.distance}`;
    if (!terrainCache.has(key)) {
      const curve = [];
      for (let d = 0; d <= info.distance; d += 1) {
        const t = terrainAt(course, d, info.distance);
        const p = d / info.distance;
        const bend = course ? courseSection(course, d, info.distance).isBend : (p > 0.2 && p < 0.43) || (p > 0.6 && p < 0.8);
        curve.push([t.grade, +bend]);
      }
      terrainCache.set(key, curve);
    }
    return { ...c, profiles: profiles.map((h) => ({ horse_no: h.horse_no, ability: h.ability, early: h.early, closing: h.closing, stamina: h.stamina, style: h.style, runCount: h.runCount, uncertainty: h.uncertainty })), terrainKey: key, courseKnown: !!course, courseAmbiguous: ambiguous };
  });
  writeFileSync(outputFile, JSON.stringify({ parameters: BASELINE_PARAMETERS, cases: prepared, terrain: Object.fromEntries(terrainCache) }));
} else if (mode === "evaluate") {
  const bundle = parameterFile ? JSON.parse(readFileSync(parameterFile, "utf8")) : {};
  const output = cases.map((c, i) => {
    const parameters = bundle.byRace?.[c.race.race_info.race_id] || bundle.parameters || BASELINE_PARAMETERS;
    const sim = simulateRace(c.race, { variation: bundle.variation ?? 0, seed: (bundle.seed ?? 20261101) ^ Number(c.race.race_info.race_id), parameters, recordSnapshots: false });
    if (i % 1000 === 0) process.stderr.write(`Exact engine ${i}/${cases.length}\n`);
    return { race_id: c.race.race_info.race_id, predictions: sim.finishOrder.map((h) => ({ horse_no: h.horse_no, time: h.finishTime, final_3f: h.finishTime - h.closingStartTime })) };
  });
  writeFileSync(outputFile, JSON.stringify(output));
} else throw new Error("Expected prepare or evaluate");
