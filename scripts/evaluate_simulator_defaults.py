"""Evaluate frozen default records against separate Agent result files; no fitting."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.simulator_input_store import _json, read_race, _seconds


def evaluate(base, day):
    groups = {}
    for path in (base / 'data/simulator_runs' / day).glob('*/*/*.json'):
        record = _json(path) or {}
        outcome = _json(base / 'data/agent_results' / f"{record.get('race_id')}.json") or {}
        rows = outcome.get('finish_order') or []
        actual = [int(r['horse_no']) for r in rows if str(r.get('finish')) in ('1', '2', '3')]
        if len(actual) != 3:
            continue
        race = read_race(base, record['race_id'], record['race_date'], record['input_version'])
        if not race or race['input_hash'] != record['input_hash']:
            continue
        scope = 'local' if str(race['race_info'].get('scope_key') or '').startswith('local') else 'central'
        g = groups.setdefault(scope, {'races': 0, 'covered_seats': 0, 'all_three': 0, 'at_least_two': 0, 'single_covered_seats': 0, 'winner_top3': 0, 'winner_top5': 0, 'winner_rank_sum': 0, 'brier_sum': 0, 'brier_horses': 0, 'winner_time_errors': [], 'freeze_evidence': {}, 'closing_labels': 0})
        ranked = sorted(record['tally'], key=lambda no: (-record['tally'][no]['top3'], -record['tally'][no]['wins'], int(no)))
        top5 = set(map(int, ranked[:5])); covered = len(top5.intersection(actual))
        winner = next(int(r['horse_no']) for r in rows if str(r.get('finish')) == '1')
        single = [int(h['horse_no']) for h in record['finish_order']]
        g['races'] += 1; g['covered_seats'] += covered; g['all_three'] += covered == 3; g['at_least_two'] += covered >= 2
        g['single_covered_seats'] += len(set(single[:5]).intersection(actual)); g['winner_top3'] += str(winner) in ranked[:3]; g['winner_top5'] += winner in top5
        g['winner_rank_sum'] += ranked.index(str(winner)) + 1 if str(winner) in ranked else len(ranked) + 1
        winner_row = next(r for r in rows if str(r.get('finish')) == '1')
        actual_time = _seconds(winner_row.get('time_seconds') or winner_row.get('time'))
        modeled = next((r for r in record['finish_order'] if int(r['horse_no']) == winner), None)
        if actual_time and modeled and modeled.get('finishTime'):
            g['winner_time_errors'].append(abs(modeled['finishTime'] - actual_time))
        evidence = record.get('freeze_evidence', 'unknown'); g['freeze_evidence'][evidence] = g['freeze_evidence'].get(evidence, 0) + 1
        for no, tally in record['tally'].items():
            g['brier_sum'] += (tally['top3'] / record['batch_count'] - int(int(no) in actual)) ** 2
            g['brier_horses'] += 1
    for g in groups.values():
        g['top5_top3_recall'] = g['covered_seats'] / (3 * g['races'])
        g['winner_average_rank'] = g.pop('winner_rank_sum') / g['races']
        g['top3_brier'] = g.pop('brier_sum') / g.pop('brier_horses')
        g['closing_evaluation'] = 'unavailable_without_measured_labels'
        errors = g.pop('winner_time_errors')
        g['winner_time_label_count'] = len(errors)
        g['winner_time_mae_seconds'] = sum(errors) / len(errors) if errors else None
    return {'race_date': day, 'groups': groups, 'method': 'fixed_model_no_fitting', 'note': 'Unverified source timestamps cannot establish a pre-off forward test.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--date', required=True); args = parser.parse_args()
    print(json.dumps(evaluate(ROOT / 'pipeline', args.date), ensure_ascii=False, indent=2))
