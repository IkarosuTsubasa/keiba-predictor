from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from bs4 import BeautifulSoup

from keiba_llm_agent.parsers.netkeiba_performance_fields import (
    explicit_race_class, parse_body_weight, parse_pace, parse_sex_age, parse_time_seconds,
)
from keiba_llm_agent.parsers.netkeiba_horse_parser import parse_horse_recent_runs
from keiba_llm_agent.parsers.netkeiba_race_parser import parse_netkeiba_shutuba_html
from keiba_llm_agent.fetchers.netkeiba_horse_fetcher import enrich_race_data_with_recent_runs
from keiba_llm_agent.schemas.race_data import RaceData, RaceInfo, HorseEntry, RecentRun
from keiba_llm_agent.scoring.turf_place_features import build_supplementary_features
from keiba_llm_agent.scoring.recent_run_scorer import score_horse_by_recent_runs
from pipeline.prepare_turf_supplements import enrich_frozen_race
from pipeline.collect_turf_supplements import verify_existing_response
from keiba_llm_agent.agents.race_analysis_agent import RaceAnalysisAgent


class SupplementTests(unittest.TestCase):
    def test_collector_resume_stops_at_previous_restrictions(self):
        for status in (401, 403, 429, 500, None):
            with self.assertRaises(RuntimeError):
                verify_existing_response({'status': status}, Path('absent.html.gz'))
        with self.assertRaises(RuntimeError):
            verify_existing_response({'status': 200, 'error': 'Missing table'}, Path('absent.html.gz'))

    def test_collector_resume_checks_saved_response_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'response.html.gz'
            path.write_bytes(b'preserved response')
            meta = {'status': 200, 'stored_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            verify_existing_response(meta, path)
            meta['stored_sha256'] = 'wrong'
            with self.assertRaises(RuntimeError):
                verify_existing_response(meta, path)

    def race(self):
        return RaceData(race_info=RaceInfo(race_id='202605040911', race_date='2026-10-25',
            course='東京', surface='芝', distance=1600, source='central', scope_key='central_turf'),
            horses=[HorseEntry(horse_no=1, horse_id='2021104073', horse_name='a', carried_weight=56,
                recent_runs=[RecentRun(race_id='202605020811', date='2026-05-17', course='東京',
                    surface='芝', distance=1600, finish=2, field_size=16, final_3f=34.0)]),
                HorseEntry(horse_no=2, horse_id='other', horse_name='b', recent_runs=[])])

    def sources(self):
        race = self.race()
        card = race.model_dump()
        card['race_info'].update(race_class='2勝クラス', race_class_source='RaceData02')
        card['horses'][0].update(age=5, sex='牝', body_weight=500, odds=999)
        row = race.horses[0].recent_runs[0].model_dump()
        row.update(time_seconds=95.2, carried_weight=55, body_weight=480, body_weight_change=-2,
                   pace_first_3f=35.0, pace_last_3f=35.5)
        return race, card, {'2021104073': [row]}

    def test_time_parsing_does_not_join_minutes_and_seconds(self):
        self.assertEqual(parse_time_seconds('1:34.5'), 94.5)
        self.assertEqual(parse_time_seconds('59.9'), 59.9)
        for value in ('', '--', '1:61.0', '中止', 'NaN', '0.0'):
            self.assertIsNone(parse_time_seconds(value))

    def test_weight_sex_and_pace(self):
        self.assertEqual(parse_body_weight('482 (+6)'), (482, 6))
        self.assertEqual(parse_body_weight('482'), (482, None))
        self.assertEqual(parse_body_weight('57.0'), (None, None))
        self.assertEqual(parse_sex_age('牝３'), ('牝', 3))
        self.assertEqual(parse_sex_age('セン5'), ('セ', 5))
        self.assertEqual(parse_pace('35.2-34.7'), (35.2, 34.7))
        self.assertEqual(parse_pace('--'), (None, None))

    def test_class_is_explicit_not_inferred_from_special_race_name(self):
        soup = BeautifulSoup('<h1 class="RaceName">東京特別</h1><div class="RaceData02">２勝クラス</div>', 'html.parser')
        self.assertEqual(explicit_race_class(soup), ('2勝クラス', 'RaceData02'))
        self.assertEqual(explicit_race_class(BeautifulSoup('<h1 class="RaceName">東京特別</h1>', 'html.parser')), (None, None))
        soup = BeautifulSoup('<h1 class="RaceName"><span class="Icon_GradeType1"></span></h1><div class="RaceData02">オープン</div>', 'html.parser')
        self.assertEqual(explicit_race_class(soup)[0], 'G1')

    def test_history_parser_details_are_opt_in(self):
        html = '<table><tr><th>日付</th><th>着順</th><th>距離</th><th>タイム</th><th>斤量</th><th>馬体重</th><th>ペース</th></tr><tr><td>2026/05/01</td><td>2</td><td>芝1600</td><td>1:34.5</td><td>55</td><td>482(-2)</td><td>35.2-34.5</td></tr></table>'
        self.assertIsNone(parse_horse_recent_runs(html)[0].time_seconds)
        run = parse_horse_recent_runs(html, include_performance_details=True)[0]
        self.assertEqual((run.time_seconds, run.carried_weight, run.body_weight), (94.5, 55, 482))

    def test_card_parsing_does_not_enable_other_scopes(self):
        html = (Path(__file__).parent/'fixtures/netkeiba_shutuba_sample.html').read_text(encoding='utf-8')
        html = html.replace('<td class="Weight">57.0</td>', '<td class="Barei">牝3</td><td class="Weight">57.0</td>')
        central = parse_netkeiba_shutuba_html(html, '202605180511')
        self.assertEqual(central.horses[0].age, 3)
        self.assertIsNone(parse_netkeiba_shutuba_html(html.replace('芝1600', 'ダ1600'), '202605180511').horses[0].age)
        self.assertIsNone(parse_netkeiba_shutuba_html(html.replace('東京', '大井'), '202644031102').horses[0].age)

    def test_backfill_keeps_frozen_core_and_does_not_add_runs(self):
        race, card, histories = self.sources(); before = race.model_dump()
        histories['2021104073'].append({**histories['2021104073'][0], 'race_id': 'new', 'date': '2026-10-01'})
        out, counts = enrich_frozen_race(race, card, histories)
        self.assertEqual(race.model_dump(), before)
        self.assertEqual(len(out.horses[0].recent_runs), 1)
        self.assertEqual(out.horses[0].age, 5)
        self.assertEqual(out.horses[0].recent_runs[0].time_seconds, 95.2)
        self.assertIsNone(out.horses[0].body_weight)
        self.assertIsNone(out.horses[0].odds)
        self.assertEqual(counts['history_matched'], 1)

    def test_mismatched_card_and_horse_identity_not_merged(self):
        race, card, histories = self.sources()
        card['horses'][0]['horse_id'] = 'wrong'
        out, _ = enrich_frozen_race(race, card, histories)
        self.assertIsNone(out.horses[0].age)
        card['race_info']['race_date'] = '2026-10-26'
        out, _ = enrich_frozen_race(race, card, histories)
        self.assertIsNone(out.race_info.race_class)

    def test_existing_frozen_optional_facts_are_not_replaced(self):
        race, card, histories = self.sources()
        race.race_info.race_class = '1勝クラス'
        race.horses[0].age = 4
        race.horses[0].sex = '牡'
        race.horses[0].recent_runs[0].time_seconds = 96.0
        out, counts = enrich_frozen_race(race, card, histories)
        self.assertEqual(out.race_info.race_class, '1勝クラス')
        self.assertEqual((out.horses[0].age, out.horses[0].sex), (4, '牡'))
        self.assertEqual(out.horses[0].recent_runs[0].time_seconds, 96.0)
        self.assertEqual(counts['existing_age_conflict_preserved'], 1)

    def test_changed_historical_outcome_is_not_joined(self):
        race, card, histories = self.sources()
        histories['2021104073'][0]['finish'] = 1
        out, _ = enrich_frozen_race(race, card, histories)
        self.assertIsNone(out.horses[0].recent_runs[0].time_seconds)

    def test_ambiguous_history_not_joined(self):
        race, card, histories = self.sources()
        histories['2021104073'] *= 2
        out, _ = enrich_frozen_race(race, card, histories)
        self.assertIsNone(out.horses[0].recent_runs[0].time_seconds)

    def test_future_history_does_not_change_features(self):
        race, card, histories = self.sources()
        out, _ = enrich_frozen_race(race, card, histories)
        before = build_supplementary_features(out)
        for when, rid in [('2026-10-26', 'future'), ('2026-10-25', 'same_day'), ('2026-09-01', out.race_info.race_id)]:
            out.horses[0].recent_runs.append(out.horses[0].recent_runs[0].model_copy(update={'date': when, 'race_id': rid, 'time_seconds': 10}))
        after = build_supplementary_features(out)
        for key in before:
            np.testing.assert_allclose(before[key], after[key], equal_nan=True)

    def test_market_identity_and_target_body_weight_are_not_features(self):
        race, card, histories = self.sources()
        out, _ = enrich_frozen_race(race, card, histories)
        before = build_supplementary_features(out)
        for h in out.horses:
            h.odds = 1.1; h.popularity = 1; h.body_weight = 800; h.body_weight_change = 99
            h.horse_name = 'changed'; h.horse_id = 'changed'
            for r in h.recent_runs:
                r.odds = 999; r.popularity = 99
        for key, value in build_supplementary_features(out).items():
            np.testing.assert_allclose(before[key], value, equal_nan=True)

    def test_local_and_dirt_features_rejected(self):
        race = self.race()
        race.race_info.source = 'local'
        with self.assertRaises(ValueError):
            build_supplementary_features(race)
        race.race_info.source = 'central'; race.race_info.surface = 'ダート'
        with self.assertRaises(ValueError):
            build_supplementary_features(race)

    def test_history_fetch_failure_is_recorded_only_for_turf(self):
        race = self.race()
        with patch('keiba_llm_agent.fetchers.netkeiba_horse_fetcher.fetch_horse_html', side_effect=RuntimeError('failed')):
            with self.assertWarns(UserWarning):
                out = enrich_race_data_with_recent_runs(race)
        self.assertEqual(out.horses[0].history_status, 'fetch_failed')
        race.race_info.scope_key = 'local'
        with patch('keiba_llm_agent.fetchers.netkeiba_horse_fetcher.fetch_horse_html', side_effect=RuntimeError('failed')):
            with self.assertWarns(UserWarning):
                out = enrich_race_data_with_recent_runs(race)
        self.assertIsNone(out.horses[0].history_status)

    def test_missing_table_is_not_treated_as_no_prior_runs(self):
        with patch('keiba_llm_agent.fetchers.netkeiba_horse_fetcher.fetch_horse_html', return_value='<html></html>'):
            with self.assertWarns(UserWarning):
                out = enrich_race_data_with_recent_runs(self.race())
        self.assertEqual(out.horses[0].history_status, 'missing_results_table')

    def test_new_optional_fields_do_not_change_existing_scores(self):
        for scope, surface, source, course in [('central_turf', '芝', 'central', '東京'),
                                              ('central_dirt', 'ダート', 'central', '東京'),
                                              ('local', 'ダート', 'local', '大井')]:
            race = self.race()
            race.race_info.scope_key = scope; race.race_info.surface = surface
            race.race_info.source = source; race.race_info.course = course
            horse = race.horses[0]
            before = score_horse_by_recent_runs(horse, race.race_info).model_dump()
            horse.age = 3; horse.sex = '牝'; horse.body_weight = 480
            race.race_info.race_class = 'G1'
            horse.recent_runs[0].time_seconds = 94.0
            horse.recent_runs[0].carried_weight = 55.0
            horse.recent_runs[0].body_weight = 478
            self.assertEqual(before, score_horse_by_recent_runs(horse, race.race_info).model_dump())

    def test_supplements_do_not_enter_unvalidated_llm_ranking(self):
        race, card, histories = self.sources()
        enriched, _ = enrich_frozen_race(race, card, histories)
        for scope in ('central_turf', 'central_dirt', 'local'):
            enriched.race_info.scope_key = scope
            client = Mock()
            with patch('keiba_llm_agent.agents.race_analysis_agent.has_recent_runs_data', return_value=False), \
                 patch('keiba_llm_agent.agents.race_analysis_agent.Prediction.model_validate'):
                RaceAnalysisAgent(client).run(enriched, [], scope_reranker_enabled=False)
            payload = client.generate_analysis.call_args.args[1]['race_data']
            self.assertNotIn('race_class', payload['race_info'])
            self.assertNotIn('age', payload['horses'][0])
            self.assertNotIn('time_seconds', payload['horses'][0]['recent_runs'][0])
            self.assertEqual(payload['horses'][0]['recent_runs'][0]['finish'], 2)


if __name__ == '__main__':
    unittest.main()
