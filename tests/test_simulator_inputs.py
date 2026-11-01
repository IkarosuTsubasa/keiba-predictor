import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipeline.simulator_input_store import publish_race, read_race, list_races, sanitize_race
from pipeline.web_data.simulator_inputs import register_routes


def example():
    return {"race_info": {"race_id": "202605050101", "race_date": "2026-11-01", "course": "東京", "surface": "芝", "distance": 1600, "source": "central", "scope_key": "central_turf", "scheduled_start_at": "2026-11-01T15:00:00+09:00"}, "horses": [
        {"horse_no": no, "horse_name": f"テスト{no}", "horse_id": str(no), "odds": 1.1, "popularity": 1, "total_score": 99, "recent_runs": [
            {"race_id": "202605040101", "date": "2026-10-01", "distance": 1600, "surface": "芝", "finish": no, "field_size": 10, "odds": 1.1, "popularity": 1, "final_3f": 35.5},
            {"date": "2026-11-01", "distance": 1600, "finish": 1}, {"date": "2099-01-01", "distance": 1600},
            {"race_id": "202605050101", "date": "2026-10-01", "distance": 1600, "finish": 1}]} for no in range(1, 5)]}


class SimulatorInputTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "pipeline"
        self.root.mkdir()
        self.raw = example()
    def tearDown(self):
        self.temp.cleanup()
    def test_whitelist_and_strict_history_cutoff(self):
        self.raw['private_memory'] = {'secret': True}
        data = publish_race(self.root, self.raw)
        text = json.dumps(data)
        for forbidden in ('odds', 'popularity', 'total_score', 'private_memory', 'secret', 'running_style'):
            self.assertNotIn(forbidden, text)
        self.assertEqual(len(data['horses'][0]['recent_runs']), 1)
        self.assertNotIn(str(self.root), text)
        self.assertEqual(data['freeze_evidence'], 'source_timestamp_unverified')
    def test_versions_are_immutable_and_repeated_collection_is_idempotent(self):
        one = publish_race(self.root, self.raw, source_as_of='2026-11-01T01:00:00Z')
        same = publish_race(self.root, self.raw, source_as_of='2026-11-01T02:00:00Z')
        self.assertEqual(one, same)
        self.raw['horses'][0]['entry_status'] = 'scratched'
        two = publish_race(self.root, self.raw, source_as_of='2026-11-01T03:00:00Z')
        self.assertNotEqual(one['input_version'], two['input_version'])
        self.assertEqual(read_race(self.root, one['race_id'], one['race_date'], one['input_version']), one)
        self.assertEqual(read_race(self.root, one['race_id'], one['race_date']), two)
        self.assertEqual(list_races(self.root, one['race_date'])[0]['horse_count'], 3)
    def test_concurrent_identical_publications_remain_complete(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(lambda _: publish_race(self.root, self.raw), range(16)))
        self.assertTrue(all(r['input_version'] == rows[0]['input_version'] for r in rows))
        self.assertEqual(len(list(self.root.glob('data/simulator_inputs/*/*/*.json'))), 2)
    def test_invalid_identity_and_entries_rejected(self):
        for rid, day in (('../abc', '2026-11-01'), ('202605050101', '2026-02-30')):
            with self.assertRaises(ValueError): read_race(self.root, rid, day)
        self.raw['horses'].append(copy.deepcopy(self.raw['horses'][0]))
        with self.assertRaises(ValueError): sanitize_race(self.raw)
    def test_old_audit_cannot_replace_collected_cancellation(self):
        older = copy.deepcopy(self.raw)
        self.raw['horses'][0]['entry_status'] = 'scratched'
        live = publish_race(self.root, self.raw, source_as_of='2026-11-01T06:00:00Z')
        stale = publish_race(self.root, older, source='full_agent_audit', source_as_of=None)
        self.assertNotEqual(stale['input_version'], live['input_version'])
        self.assertEqual(read_race(self.root, live['race_id'], live['race_date'])['input_version'], live['input_version'])
    def test_partial_unsupported_and_no_active_field(self):
        self.assertEqual(sanitize_race(self.raw)['availability'], 'partial')
        self.raw['race_info']['surface'] = '障害'
        self.assertEqual(sanitize_race(self.raw)['availability'], 'unsupported')
        self.raw['race_info']['surface'] = '芝'
        for h in self.raw['horses']: h['entry_status'] = 'excluded'
        self.assertEqual(sanitize_race(self.raw)['availability'], 'unavailable')
    def test_schedule_keeps_pending_and_unknown_time_separate(self):
        schedule = [{'race_id': self.raw['race_info']['race_id'], 'race_date': '2026-11-01', 'scope_key': 'central_turf'}]
        all_rows = list_races(self.root, '2026-11-01', schedule)
        self.assertEqual(all_rows[0]['availability'], 'preparing')
        self.assertEqual(all_rows[0]['race_state'], 'unknown')
        self.assertEqual(list_races(self.root, '2026-11-01', schedule, state='upcoming'), [])
    def test_historical_result_requires_date_distance_and_identity(self):
        target = self.raw['horses'][0]['recent_runs'][0]
        hist = copy.deepcopy(self.raw)
        hist['race_info'].update(race_id=target['race_id'], race_date=target['date'])
        saved = self.root.parent / 'keiba_llm_agent/data/race_data' / f"{target['race_id']}.json"
        saved.parent.mkdir(parents=True)
        saved.write_text(json.dumps(hist), encoding='utf-8')
        result = self.root / 'data/agent_results' / f"{target['race_id']}.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({'race_id': target['race_id'], 'finish_order': [{'horse_no': 1, 'horse_name': 'テスト1', 'finish': 1, 'time': '1:35.0'}]}), encoding='utf-8')
        clean = sanitize_race(self.raw, base_dir=self.root)
        self.assertEqual(clean['horses'][0]['recent_runs'][0]['time_seconds'], 95)
        hist['race_info']['distance'] = 1800
        saved.write_text(json.dumps(hist), encoding='utf-8')
        self.assertIsNone(sanitize_race(self.raw, base_dir=self.root)['horses'][0]['recent_runs'][0]['time_seconds'])
    def test_http_dates_versions_and_etag(self):
        app = FastAPI()
        schedule = [{'race_id': self.raw['race_info']['race_id'], 'race_date': '2026-11-01'}]
        register_routes(app, self.root, lambda: schedule)
        client = TestClient(app)
        route = '/keiba/api/public/simulator/races/202605050101?date=2026-11-01'
        self.assertEqual(client.get(route).status_code, 409)
        saved = publish_race(self.root, self.raw)
        response = client.get(route)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.get(route, headers={'If-None-Match': response.headers['etag']}).status_code, 304)
        version = client.get(route + '&input_version=' + saved['input_version'])
        self.assertIn('immutable', version.headers['cache-control'])
        self.assertEqual(client.get(route + '&input_version=' + 'a' * 24).json()['error'], 'version_not_found')
        self.assertEqual(client.get(route.replace('2026-11-01', '2026-11-02')).status_code, 404)
        self.assertEqual(client.get(route.replace('2026-11-01', '2026-02-30')).status_code, 400)
        self.assertEqual(client.get('/api/public/simulator/races?date=2026-11-01').status_code, 200)


if __name__ == '__main__': unittest.main()
