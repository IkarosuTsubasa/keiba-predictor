"""Read-only prospective paired evaluation of version-selected Agent shadow audits."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keiba_llm_agent.backtest.input_selection import select_effective_backtest_audit_paths
from keiba_llm_agent.scoring.turf_shadow import MODEL_VERSION


def read_json(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def paired_metrics(pairs, key):
    n = len(pairs)
    hits = [len(set(p[key][:5]) & set(p['top3'])) for p in pairs]
    return {'races': n, 'covered_slots': sum(hits),
            'recall_pct': 100*sum(hits)/(3*n) if n else None,
            'all3': sum(h == 3 for h in hits), 'atleast2': sum(h >= 2 for h in hits)}


def eligible_pair(audit, result, model_version):
    shadow = audit.get('shadow_predictions', {}).get('central_turf_history')
    if not shadow:
        return None, 'shadow_missing'
    if shadow.get('model_version') != model_version:
        return None, 'other_model_version'
    if shadow.get('status') != 'predicted':
        return None, 'shadow_'+str(shadow.get('status', 'unknown'))
    if audit.get('audit_kind') != 'full' or shadow.get('authoritative') is not False:
        return None, 'invalid_audit_contract'
    info = audit.get('race_info') or {}
    rid = str(audit.get('race_id', ''))
    if shadow.get('race_id') != rid or info.get('surface') != '芝' or str(info.get('scope_key', '')).startswith('local'):
        return None, 'scope_or_identity_mismatch'
    prediction = audit.get('outputs', {}).get('prediction') or {}
    timing = shadow.get('timing') or {}
    if not timing.get('eligible_at_generation') or str(info.get('race_date', '')) <= str(shadow.get('training_end', '')):
        return None, 'not_forward_eligible'
    try:
        start = datetime.fromisoformat(timing['scheduled_start_at'].replace('Z', '+00:00'))
        generated = datetime.fromisoformat(shadow['generated_at'].replace('Z', '+00:00'))
        received = datetime.fromisoformat(prediction['prediction_received_at'].replace('Z', '+00:00'))
        if any(t.tzinfo is None for t in (start, generated, received)):
            raise ValueError('Timezone unknown')
    except (KeyError, TypeError, ValueError, AttributeError):
        return None, 'timestamp_missing_or_invalid'
    if generated >= start or received >= start:
        return None, 'generated_or_received_after_start'
    if not result:
        return None, 'unsettled'
    if str(result.get('race_id', '')) != rid:
        return None, 'result_identity_mismatch'
    top3 = [result.get('result', {}).get(key) for key in ('1st', '2nd', '3rd')]
    if not all(type(n) is int for n in top3) or len(set(top3)) != 3:
        return None, 'invalid_podium'
    candidate = shadow.get('candidate_ranking') or []
    baseline = shadow.get('baseline_ranking') or []
    roster = shadow.get('horse_numbers') or []
    actual_baseline = [s['horse_no'] for s in sorted(prediction.get('horse_scores', []),
                        key=lambda s: (-float(s['total_score']), s['horse_no']))]
    if (len(set(roster)) != len(roster) or not roster or sorted(candidate) != sorted(roster)
            or sorted(baseline) != sorted(roster) or baseline != actual_baseline
            or not set(top3).issubset(roster)):
        return None, 'ranking_or_roster_mismatch'
    return {'race_id': rid, 'date': info['race_date'], 'model_sha256': shadow.get('model_sha256'),
            'baseline': baseline, 'candidate': candidate, 'top3': top3}, None


def report(snapshot, model_version=MODEL_VERSION):
    counts = Counter(); pairs = []
    audits = select_effective_backtest_audit_paths(snapshot/'agent_backtest_audit')
    for rid, path in sorted(audits.items()):
        try:
            audit = read_json(path)
            result_path = snapshot/'agent_results'/f'{rid}.json'
            result = read_json(result_path) if result_path.exists() else audit.get('outcomes', {}).get('result')
            pair, reason = eligible_pair(audit, result, model_version)
            if reason:
                counts[reason] += 1
            else:
                pairs.append(pair)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            counts['unreadable_or_invalid_audit'] += 1
    hashes = {p['model_sha256'] for p in pairs}
    if len(hashes) > 1 or (pairs and (None in hashes or '' in hashes)):
        raise ValueError('Mixed or missing model hashes under one version; do not pool these results')
    baseline = paired_metrics(pairs, 'baseline'); candidate = paired_metrics(pairs, 'candidate')
    return {'model_version': model_version, 'selected_audits': len(audits),
            'excluded_or_pending': dict(counts), 'baseline': baseline, 'candidate': candidate,
            'delta': {k: candidate[k]-baseline[k] if candidate[k] is not None else None
                      for k in ('covered_slots', 'recall_pct', 'all3', 'atleast2')},
            'pairs': pairs, 'note': 'One effective version per race; generated and received strictly before scheduled start; after training cutoff. Descriptive paired metrics, not a significance claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--model-version', default=MODEL_VERSION)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = report(args.snapshot.resolve(), args.model_version)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        path = args.output.resolve()
        if not path.is_relative_to(ROOT/'analysis_outputs'):
            parser.error('Reports must be written inside analysis_outputs')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            handle.write(text+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}, ensure_ascii=False, indent=2))
