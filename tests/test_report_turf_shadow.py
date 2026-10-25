import gzip
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.report_turf_shadow import MODEL_VERSION, eligible_pair, paired_metrics, report


def audit_fixture():
    rid = '202605050101'
    return {'race_id': rid, 'audit_kind': 'full', 'prediction_stage': 'final',
            'race_info': {'race_date': '2026-11-01', 'surface': '芝', 'scope_key': 'central_turf'},
            'outputs': {'prediction': {'race_id': rid, 'prediction_stage': 'final',
                'prediction_received_at': '2026-11-01T05:59:00+00:00',
                'horse_scores': [{'horse_no': n, 'total_score': 10-n} for n in range(1, 7)]}},
            'shadow_predictions': {'central_turf_history': {'model_version': MODEL_VERSION,
                'model_sha256': 'test-hash', 'status': 'predicted', 'authoritative': False,
                'race_id': rid, 'training_end': '2026-10-25', 'generated_at': '2026-11-01T05:58:00+00:00',
                'timing': {'eligible_at_generation': True, 'scheduled_start_at': '2026-11-01T15:00:00+09:00'},
                'candidate_ranking': [6, 5, 4, 3, 2, 1], 'baseline_ranking': [1, 2, 3, 4, 5, 6],
                'horse_numbers': [1, 2, 3, 4, 5, 6]}}}


def result_fixture():
    return {'race_id': '202605050101', 'result': {'1st': 6, '2nd': 5, '3rd': 2}}


class ShadowReportTests(unittest.TestCase):
    def test_paired_metrics_are_podium_slots_not_winner_in_top5(self):
        pair, reason = eligible_pair(audit_fixture(), result_fixture(), MODEL_VERSION)
        self.assertIsNone(reason)
        self.assertEqual(paired_metrics([pair], 'baseline')['covered_slots'], 2)
        self.assertEqual(paired_metrics([pair], 'candidate')['covered_slots'], 3)
        self.assertEqual(paired_metrics([pair], 'candidate')['all3'], 1)

    def test_late_or_missing_receipt_cannot_be_prospective(self):
        for receipt in (None, '2026-11-01T06:00:00+00:00', '2026-11-01T05:59:00'):
            audit = audit_fixture()
            audit['outputs']['prediction']['prediction_received_at'] = receipt
            self.assertIsNone(eligible_pair(audit, result_fixture(), MODEL_VERSION)[0])

    def test_training_period_failures_and_wrong_versions_are_excluded(self):
        for field, value in [('training_end', '2026-11-01'), ('status', 'failed'),
                             ('model_version', 'other'), ('authoritative', True)]:
            audit = audit_fixture(); audit['shadow_predictions']['central_turf_history'][field] = value
            self.assertIsNone(eligible_pair(audit, result_fixture(), MODEL_VERSION)[0])

    def test_changed_roster_or_baseline_cannot_be_compared(self):
        audit = audit_fixture()
        audit['shadow_predictions']['central_turf_history']['baseline_ranking'].reverse()
        self.assertEqual(eligible_pair(audit, result_fixture(), MODEL_VERSION)[1], 'ranking_or_roster_mismatch')
        audit = audit_fixture()
        audit['shadow_predictions']['central_turf_history']['candidate_ranking'] = [6]*6
        self.assertEqual(eligible_pair(audit, result_fixture(), MODEL_VERSION)[1], 'ranking_or_roster_mismatch')

    def test_results_must_match_race_and_have_three_distinct_finishers(self):
        for result in ({'race_id': 'wrong', 'result': {'1st': 6, '2nd': 5, '3rd': 2}},
                       {'race_id': '202605050101', 'result': {'1st': 6, '2nd': 6, '3rd': 2}}, None):
            self.assertIsNone(eligible_pair(audit_fixture(), result, MODEL_VERSION)[0])

    def test_report_is_read_only_and_deduplicates_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rid = '202605050101'; audit = audit_fixture()
            path = root/'agent_backtest_audit'/f'{rid}.json.gz'
            path.parent.mkdir()
            with gzip.open(path, 'wt', encoding='utf-8') as handle:
                json.dump(audit, handle)
            result_path = root/'agent_results'/f'{rid}.json'
            result_path.parent.mkdir()
            result_path.write_text(json.dumps(result_fixture()), encoding='utf-8')
            version = path.parent/'_versions'/rid/'final'/'task.json.gz'
            version.parent.mkdir(parents=True)
            version.write_bytes(path.read_bytes())
            before = {p: p.read_bytes() for p in (path, version, result_path)}
            data = report(root)
            self.assertEqual(data['baseline']['races'], 1)
            self.assertEqual(data['delta']['covered_slots'], 1)
            for p, contents in before.items():
                self.assertEqual(p.read_bytes(), contents)

    def test_empty_cohort_is_not_reported_as_zero_percent_accuracy(self):
        self.assertIsNone(paired_metrics([], 'candidate')['recall_pct'])


if __name__ == '__main__':
    unittest.main()
