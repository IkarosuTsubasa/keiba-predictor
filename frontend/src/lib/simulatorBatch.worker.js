import { simulateRace } from "./raceSimulator.js";
self.onmessage = ({ data }) => {
  const { race, settings, taskId } = data;
  const tallies = Object.fromEntries(race.horses.map(h => [h.horse_no, { wins: 0, top3: 0 }]));
  for (let i = 0; i < 200; i++) {
    const result = simulateRace(race, { ...settings, seed: (settings.seed + i * 7919) >>> 0, recordSnapshots: false });
    tallies[result.finishOrder[0].horse_no].wins++;
    result.finishOrder.slice(0, 3).forEach(h => tallies[h.horse_no].top3++);
    if ((i + 1) % 10 === 0) self.postMessage({ taskId, type: "progress", completed: i + 1 });
  }
  self.postMessage({ taskId, type: "complete", completed: 200, tallies });
};
