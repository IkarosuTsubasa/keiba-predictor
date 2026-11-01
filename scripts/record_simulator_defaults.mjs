// Reproducible default records. Reads published inputs only; never edits sources.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { simulateRace, MODEL_VERSION } from '../frontend/src/lib/raceSimulator.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const day = process.argv[process.argv.indexOf('--date') + 1];
if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error('--date YYYY-MM-DD required');
const inputs = path.join(root, 'pipeline/data/simulator_inputs', day);
const revision = execFileSync('git', ['rev-parse', 'HEAD'], {cwd: root, encoding: 'utf8'}).trim();
const engineHash = crypto.createHash('sha256');
for (const file of ['raceSimulator.js', 'historicalRaceProfiles.js', 'raceCourses.js']) engineHash.update(fs.readFileSync(path.join(root, 'frontend/src/lib', file)));
engineHash.update(fs.readFileSync(path.join(root, 'frontend/src/fixtures/raceCourses.json')));
const engineHashText = engineHash.digest('hex');
let count = 0;
for (const rid of fs.existsSync(inputs) ? fs.readdirSync(inputs) : []) {
  if (!/^\d{12}$/.test(rid)) continue;
  const directory = path.join(inputs, rid), latest = JSON.parse(fs.readFileSync(path.join(directory, 'latest.json'), 'utf8'));
  if (!['ready', 'partial'].includes(latest.availability)) continue;
  const input = JSON.parse(fs.readFileSync(path.join(directory, `${latest.input_version}.json`), 'utf8'));
  const race = {...input, horses: input.horses.filter(h => h.entry_status === 'active')};
  const settings = {seed: input.default_seed, pace: 'average', variation: 1, recordSnapshots: false};
  const output = path.join(root, 'pipeline/data/simulator_runs', day, rid, input.input_version, `${MODEL_VERSION}-${input.default_seed}.json`);
  if (fs.existsSync(output)) continue;
  const single = simulateRace(race, settings), tally = Object.fromEntries(race.horses.map(h => [h.horse_no, {wins: 0, top3: 0}]));
  for (let i = 0; i < 200; i++) {
    const result = simulateRace(race, {...settings, seed: (input.default_seed + i * 7919) >>> 0});
    tally[result.finishOrder[0].horse_no].wins++;
    result.finishOrder.slice(0, 3).forEach(h => tally[h.horse_no].top3++);
  }
  const record = {schema_version: 1, race_id: rid, race_date: day, input_version: input.input_version, input_hash: input.input_hash, model_version: MODEL_VERSION, code_revision: revision, engine_hash: engineHashText, settings, generated_at: new Date().toISOString(), source_as_of: input.source_as_of, scheduled_off_at: input.scheduled_off_at, freeze_evidence: input.freeze_evidence, finish_order: single.finishOrder, batch_count: 200, tally};
  fs.mkdirSync(path.dirname(output), {recursive: true});
  fs.writeFileSync(output, JSON.stringify(record, null, 2) + '\n', {flag: 'wx'});
  const url = process.env.SIMULATOR_COLLECTION_CALLBACK_URL, secret = process.env.PIPELINE_CALLBACK_SECRET;
  if (url && secret) {
    const body = JSON.stringify({default_run: record});
    const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Hub-Signature-256': 'sha256=' + crypto.createHmac('sha256', secret).update(body).digest('hex')}, body});
    if (!response.ok) throw new Error(`record callback failed (${response.status})`);
  }
  count++;
}
console.log(JSON.stringify({race_date: day, recorded: count, model_version: MODEL_VERSION}));
