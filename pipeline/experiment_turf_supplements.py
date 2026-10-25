"""Fixed turf-only ablations of retrospective netkeiba supplementary inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import experiment_agent_architecture as a
from keiba_llm_agent.scoring.turf_place_features import build_supplementary_features, META_NAMES, HISTORY_NAMES
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData

e = a.e
VARIANTS = ('current', 'rich_direct', 'metadata_direct', 'history_direct', 'enriched_direct')


def sanitize(race):
    race = race.model_copy(deep=True)
    for horse in race.horses:
        horse.odds = horse.popularity = None
        horse.recent_runs = [r for r in horse.recent_runs if r.date and r.date < race.race_info.race_date
                            and r.race_id != race.race_info.race_id]
        for run in horse.recent_runs:
            run.odds = run.popularity = None
    return race


def source_hashes():
    names = [Path(__file__), ROOT/'keiba_llm_agent/scoring/turf_place_features.py',
             ROOT/'keiba_llm_agent/schemas/race_data.py',
             ROOT/'pipeline/experiment_agent_architecture.py', ROOT/'pipeline/experiment_central_evidence.py',
             ROOT/'pipeline/prepare_turf_supplements.py']
    return {str(p): e.digest(p) for p in names}


def prepare(prepared, supplements, out):
    old = joblib.load(prepared)
    assert e.production_hashes() == old['manifest']['production_hashes']
    supplement = e.read_json(supplements/'supplement_manifest.json')
    assert supplement['source_prepared_sha256'] == e.digest(prepared)
    sources = {r['race_id']: r for r in old['manifest']['source_files']}
    augmented = {r['race_id']: r for r in supplement['races']}
    rows = [r for r in old['rows'] if r['scope'] == 'central_turf']
    assert set(augmented) == {r['race_id'] for r in rows}
    for index, row in enumerate(rows):
        source = sources[row['race_id']]
        audit = e.read_json(Path(source['audit']))
        race = sanitize(RaceData.model_validate(audit['inputs']['race_data']))
        saved = Prediction.model_validate(audit['outputs']['prediction'])
        refreshed = e.refreshed_prediction(race, saved, 'current')
        record = e.feature_record(race, refreshed, 'current')
        np.testing.assert_allclose(record['matrix'], row['variants']['current']['matrix'], equal_nan=True)
        assert record['raw'] == row['variants']['current']['raw']
        path = Path(augmented[row['race_id']]['enriched'])
        assert e.digest(path) == augmented[row['race_id']]['enriched_sha256']
        enriched = sanitize(RaceData.model_validate(e.read_json(path)))
        assert [h.horse_no for h in enriched.horses] == row['numbers']
        np.testing.assert_allclose(a.raw_features(enriched), row['raw_features'], equal_nan=True)
        new_prediction = e.refreshed_prediction(enriched, saved, 'current')
        new_record = e.feature_record(enriched, new_prediction, 'current')
        np.testing.assert_allclose(new_record['matrix'], record['matrix'], equal_nan=True)
        assert new_record['raw'] == record['raw']
        blocks = build_supplementary_features(enriched)
        row['experiment_matrices'] = {
            'rich_direct': row['rich_features'],
            'metadata_direct': np.column_stack((row['rich_features'], blocks['metadata'])),
            'history_direct': np.column_stack((row['rich_features'], blocks['history'])),
            'enriched_direct': np.column_stack((row['rich_features'], blocks['metadata'], blocks['history'])),
        }
        if (index + 1) % 100 == 0:
            print(f'Baseline replay and feature parity verified: {index + 1}', flush=True)
    manifest = {'source_prepared': str(prepared.resolve()), 'source_prepared_sha256': e.digest(prepared),
                'supplement_manifest': str((supplements/'supplement_manifest.json').resolve()),
                'supplement_manifest_sha256': e.digest(supplements/'supplement_manifest.json'),
                'source_hashes': source_hashes(), 'production_hashes': e.production_hashes(),
                'numpy_version': np.__version__, 'sklearn_version': e.sklearn.__version__,
                'races': len(rows), 'qualification': old['manifest']['counts']['central_turf'],
                'metadata_features': list(META_NAMES), 'historical_features': list(HISTORY_NAMES),
                'feature_widths': {k: v.shape[1] for k, v in rows[0]['experiment_matrices'].items()},
                'source_files': [sources[r['race_id']] for r in rows],
                'baseline_parity': 'All eligible races replayed; new optional fields do not change existing scores/features.',
                'scope': 'central_turf only; all eligible full retained regardless of supplement availability',
                'caveat': 'New fields retrospectively acquired; this is not a frozen historical pre-off acquisition or prospective test.'}
    joblib.dump({'rows': rows, 'manifest': manifest}, out/'prepared.joblib', compress=3)
    e.save_json(out/'manifest.json', manifest)
    return rows, manifest


def evaluate_fold(training, testing):
    place_template, winner_template = e.templates('central_turf')
    place = e.fit_model(training, 'current', 'central_turf', place_template)
    winner = e.fit_model(training, 'current', 'central_turf', winner_template, winner=True)
    current_p = place.predict_proba(np.concatenate([r['variants']['current']['matrix'] for r in testing]))[:, 1]
    winner_p = winner.predict_proba(np.concatenate([r['variants']['current']['winner_matrix'] for r in testing]))[:, 1]
    y, weights = a.labels_weights(training)
    signals = {}
    for name in VARIANTS[1:]:
        model = clone(place_template)
        model.fit(np.concatenate([r['experiment_matrices'][name] for r in training]), y, sample_weight=weights)
        signals[name] = model.predict_proba(np.concatenate([r['experiment_matrices'][name] for r in testing]))[:, 1]
    vectors = {v: [] for v in VARIANTS}; records = []; offset = 0
    for row in testing:
        section = slice(offset, offset+len(row['numbers'])); offset += len(row['numbers'])
        rankings = {'current': e.runtime_ranking(row, 'current', current_p[section], winner_p[section])}
        rankings.update({v: a.direct_ranking(row['numbers'], p[section]) for v, p in signals.items()})
        for name, ranking in rankings.items():
            vectors[name].append(e.metric_vector(row, ranking))
        records.append({'race_id': row['race_id'], 'date': row['date'], 'rankings': rankings, 'top3': row['top3']})
    return {k: np.asarray(v) for k, v in vectors.items()}, records


def deployed_metrics(rows):
    place, winner = e.templates('central_turf')
    p = place.predict_proba(np.concatenate([r['variants']['current']['matrix'] for r in rows]))[:, 1]
    w = winner.predict_proba(np.concatenate([r['variants']['current']['winner_matrix'] for r in rows]))[:, 1]
    offset = 0; vectors = []
    for row in rows:
        section = slice(offset, offset+len(row['numbers'])); offset += len(row['numbers'])
        vectors.append(e.metric_vector(row, e.runtime_ranking(row, 'current', p[section], w[section])))
    return np.asarray(vectors)


def evaluate(rows, manifest, out):
    arrays = {v: [] for v in VARIANTS}; records = []
    for seed in e.DEFAULT_SEEDS:
        values = {v: np.full((len(rows), 8), np.nan) for v in VARIANTS}
        seen = np.zeros(len(rows), dtype=int)
        for fold, (train, test) in enumerate(e.date_splits(rows, seed)):
            vectors, rankings = evaluate_fold([rows[i] for i in train], [rows[i] for i in test])
            seen[test] += 1
            for v in VARIANTS:
                values[v][test] = vectors[v]
            records.extend({'validation': 'repeated_oof', 'seed': seed, 'fold': fold, **r} for r in rankings)
            print(f'turf seed={seed} fold={fold} test={len(test)}', flush=True)
        assert (seen == 1).all() and all(np.isfinite(v).all() for v in values.values())
        for v in VARIANTS:
            arrays[v].append(values[v])
    arrays = {v: np.asarray(x) for v, x in arrays.items()}
    previous = np.load(Path(manifest['source_prepared']).parent/'central_turf_architecture_oof.npz')
    for name in ('current', 'rich_direct'):
        np.testing.assert_allclose(arrays[name], previous[name])
    results = {'races': len(rows), 'previous_experiment_parity': True,
               'baseline': e.summary(arrays['current']),
               'versus_current': {v: e.comparison(rows, arrays['current'], arrays[v]) for v in VARIANTS[1:]},
               'supplement_increment': {v: e.comparison(rows, arrays['rich_direct'], arrays[v]) for v in VARIANTS[2:]}}
    tests = []
    for reference, names in (('current', VARIANTS[1:]), ('rich_direct', VARIANTS[2:])):
        for v in names:
            tests.append({'reference': reference, 'candidate': v,
                          'p': e.signflip(rows, (arrays[v][:, :, 0]-arrays[reference][:, :, 0]).mean(axis=0))})
    maximum = 0
    for index, test in enumerate(sorted(tests, key=lambda t: t['p'])):
        maximum = min(1, max(maximum, (len(tests)-index)*test['p']))
        test['holm_p'] = maximum
    time_rows = []; time_vectors = {v: [] for v in VARIANTS}; blocks = []
    for block in np.array_split(sorted({r['date'] for r in rows}), 5)[1:]:
        train = [r for r in rows if r['date'] < block[0]]
        test_rows = [r for r in rows if r['date'] in set(block)]
        vectors, rankings = evaluate_fold(train, test_rows)
        time_rows.extend(test_rows)
        for v in VARIANTS:
            time_vectors[v].extend(vectors[v])
        blocks.append({'train': len(train), 'test': len(test_rows), 'training_through': max(r['date'] for r in train),
                       'test_from': str(block[0]), 'test_through': str(block[-1]),
                       'slot_delta': {v: float(vectors[v][:, 0].sum()-vectors['current'][:, 0].sum()) for v in VARIANTS[1:]}})
        records.extend({'validation': 'chronological', **r} for r in rankings)
        print(f'turf chronological test_from={block[0]} races={len(test_rows)}', flush=True)
    results['chronological'] = {v: e.comparison(time_rows, np.asarray([time_vectors['current']]), np.asarray([time_vectors[v]])) for v in VARIANTS[1:]}
    results['time_blocks'] = blocks
    model_metadata = e.read_json(e.reranker._model_directory()/'metadata.json')['models']['central_turf']
    valid_dates = {r['date'] for b in blocks if b['training_through'] >= model_metadata['training_end']
                   for r in time_rows if b['test_from'] <= r['date'] <= b['test_through']}
    ids = [i for i, r in enumerate(time_rows) if r['date'] in valid_dates]
    selected = [time_rows[i] for i in ids]
    fixed = deployed_metrics(selected)
    results['versus_deployed_time_separated'] = {v: e.comparison(selected, np.asarray([fixed]), np.asarray([[time_vectors[v][i] for i in ids]])) for v in VARIANTS}
    old = [r for r in rows if r['date'] <= a.CUTOFF]
    new = [r for r in rows if r['date'] > a.CUTOFF]
    vectors, rankings = evaluate_fold(old, new)
    results['fixed_oct22_cutoff'] = {v: e.comparison(new, np.asarray([deployed_metrics(new)]), np.asarray([vectors[v]])) for v in VARIANTS}
    records.extend({'validation': 'fixed_oct22_cutoff', **r} for r in rankings)
    assert source_hashes() == manifest['source_hashes']
    assert e.production_hashes() == manifest['production_hashes']
    for record in manifest['source_files']:
        for name in ('audit', 'prediction', 'result'):
            assert e.digest(Path(record[name])) == record[name+'_sha256']
    np.savez_compressed(out/'oof.npz', **arrays)
    e.save_json(out/'results.json', results)
    e.save_json(out/'rankings.json', records)
    e.save_json(out/'multiple_comparisons.json', tests)
    print('All turf comparisons complete; baseline parity and input hashes verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--supplements', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'analysis_outputs') or out.exists():
        parser.error('Use a new output directory inside analysis_outputs')
    out.mkdir(parents=True)
    with threadpool_limits(limits=2):
        rows, manifest = prepare(args.prepared.resolve(), args.supplements.resolve(), out)
        evaluate(rows, manifest, out)
