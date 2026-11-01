import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { appLink, contextFromUrl, validContext, engineRace, validateSettings } from './simulatorClient.js';
import { simulateRace } from './raceSimulator.js';

test('同场跳转保留版本种子，拒绝非法上下文', () => {
  const context = { raceId: '202605050101', date: '2026-11-01', inputVersion: 'abcdef123456abcdef123456', seed: 42, modelVersion: 'illustrative-terrain-v2', source: 'web_detail' };
  assert.deepEqual(contextFromUrl(`https://www.ikaimo-ai.com${appLink(context)}`), context);
  assert.equal(validContext({...context, inputVersion: '../../secret'}), false);
  assert.equal(validContext({...context, modelVersion: 'unknown'}), false);
  assert.equal(validContext({...context, seed: -1}), false);
  assert.throws(() => validateSettings({pace: 'unknown'}));
  assert.throws(() => validateSettings({distance: 999999}));
  assert.deepEqual(validateSettings({distance: null, courseProfileId: null}), {distance: null, courseProfileId: null});
});
test('取消马不参赛，两端固定输入与种子得到相同轨迹', () => {
  const fixture = JSON.parse(fs.readFileSync(new URL('../fixtures/raceSimulatorDemo.json', import.meta.url)));
  const race = structuredClone(fixture.races[0]);
  race.horses.forEach((h, i) => h.entry_status = i === 0 ? 'scratched' : 'active');
  const ready = engineRace(race), one = simulateRace(ready, {seed: 42}), two = simulateRace(ready, {seed: 42});
  assert.equal(ready.horses.length, race.horses.length - 1);
  assert.deepEqual(one.finishOrder, two.finishOrder);
  assert.deepEqual(one.snapshots, two.snapshots);
  const fast = simulateRace(ready, {seed: 42, recordSnapshots: false});
  assert.deepEqual(fast.finishOrder, one.finishOrder);
});
