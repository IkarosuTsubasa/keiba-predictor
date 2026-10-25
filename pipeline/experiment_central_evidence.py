"""Central-only, offline experiments; never writes production models or inputs."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from keiba_llm_agent.backtest.input_selection import (
    select_effective_backtest_audit_paths, select_effective_prediction_paths)
from keiba_llm_agent.config.scoring_config import resolve_scoring_profile_config
from keiba_llm_agent.schemas.prediction import Prediction, ScoringConfigSnapshot
from keiba_llm_agent.schemas.race_data import RaceData, RaceInfo, RecentRun
from keiba_llm_agent.scoring import recent_run_scorer as scorer
from keiba_llm_agent.scoring import scope_reranker as reranker

SCOPES = ('central_turf', 'central_dirt')
VARIANTS = ('current', 'context_form', 'relevant_evidence', 'unknown_neutral', 'surface_guard')
CANDIDATES = VARIANTS[1:]
DEFAULT_SEEDS = (42, 137, 2026)
EVIDENCE_NAMES = (
    'relevant_mean', 'relevant_best', 'relevant_worst', 'relevant_effective_count',
    'relevant_weight_mass', 'relevant_quality_minus_recent', 'same_context_good_mass',
    'same_context_bad_mass', 'cross_surface_recent_fraction', 'relevant_class_mean',
    'relevant_class_known_fraction', 'recent_relevance', 'head_finish_advantage',
    'head_margin_advantage', 'head_positive_fraction', 'head_weight_mass',
    'head_opponents', 'head_strong_opponent_advantage', 'head_positive_mass', 'head_negative_mass',
)
UNKNOWN_NAMES = ('distance_unverified', 'course_unverified', 'going_unverified',
                 'valid_history_count', 'empty_history', 'distance_evidence_count',
                 'course_evidence_count', 'going_evidence_count')
ORIGINAL_FORM = scorer.score_recent_form
ORIGINAL_TREND = scorer.score_trend
ORIGINAL_RISK = scorer.score_risk
ORIGINAL_DISTANCE = scorer.score_distance_fit
ORIGINAL_COURSE = scorer.score_course_fit
ORIGINAL_GOING = scorer.score_track_condition_fit


def read_json(path: Path):
    if path.name.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_files():
    directory = reranker._model_directory()
    names = {str(v[k]) for s, v in reranker.MODEL_SPECS.items() if s in SCOPES
             for k in ('filename', 'baseline_filename') if v.get(k)}
    return sorted(directory / n for n in names)


def production_hashes():
    files = [ROOT / 'keiba_llm_agent' / p for p in (
        'scoring/recent_run_scorer.py', 'scoring/scope_reranker.py', 'scoring/place_context.py',
        'config/scoring_config.py', 'analysis/race_level_analyzer.py', 'analysis/pace_analyzer.py',
        'analysis/race_deep_analyzer.py')]
    files += model_files()
    return {str(p): digest(p) for p in files}


def scope_of(info):
    if info.get('source') == 'local' or str(info.get('scope_key') or '').startswith('local'):
        return 'local'
    return {'\u829d':'central_turf', '\u30c0\u30fc\u30c8':'central_dirt'}.get(info.get('surface'), 'other')


def podium(result):
    values = [result.get('result', {}).get(k) for k in ('1st', '2nd', '3rd')]
    if all(isinstance(n, int) for n in values) and len(set(values)) == 3:
        return values
    rows = sorted([r for r in result.get('finish_order', []) if r.get('finish') in (1, 2, 3)],
                  key=lambda r:r['finish'])
    return [r['horse_no'] for r in rows] if len(rows) == 3 else []


def prediction_order(pred):
    return [h['horse_no'] for h in sorted(pred.get('horse_scores', []),
            key=lambda h:(-float(h['total_score']), h['horse_no']))]


def context_similarity(run: RecentRun, info: RaceInfo):
    surface = 1.0 if run.surface == info.surface and info.surface else 0.25 if run.surface and info.surface else 0.5
    distance = math.exp(-abs(run.distance-info.distance)/400.0) if run.distance and info.distance else 0.5
    return surface * distance


def contextual_mean(runs, info, *, recency_weights=True):
    if not runs:
        return 5.0
    weights = [scorer.get_weight(i) if recency_weights else 1.0 for i in range(len(runs))]
    # Unrelated evidence shrinks toward neutral, rather than becoming an excuse to add points.
    return 5.0 + sum(w*context_similarity(r, info)*(scorer.score_run_performance(r, info)-5.0)
                     for r, w in zip(runs, weights, strict=True)) / sum(weights)


def context_form(runs, info=None):
    if not runs:
        return ORIGINAL_FORM(runs, info)
    if info is None:
        return ORIGINAL_FORM(runs, info)
    valid = [r for r in runs[:scorer.RECENT_QUALITY_RUN_LIMIT] if r.finish is not None]
    return scorer.clamp_score(contextual_mean(valid, info)) if valid else 0


def context_trend(runs, info):
    valid = [r for r in runs[:scorer.RECENT_FORM_RUN_LIMIT] if r.finish is not None]
    if len(valid) < 4:
        return ORIGINAL_TREND(runs, info)
    recent = contextual_mean(valid[:2], info, recency_weights=False)
    previous = contextual_mean(valid[2:5], info, recency_weights=False)
    value = scorer.clamp_score(5.0 + (recent-previous)*0.8)
    return max(value, 4) if recent >= 6 and previous >= 7 else value


def context_risk(runs, quality, distance_fit, info):
    original = ORIGINAL_RISK(runs, quality, distance_fit)
    valid = [r for r in runs[:5] if r.finish is not None and r.field_size]
    if original == -5 and len(valid) >= 2 and all(r.finish > r.field_size/2 for r in valid[:2]):
        relevance = sum(context_similarity(r, info) for r in valid[:2])/2
        return max(-5, min(-3, round(-3.0-2.0*relevance)))
    return original


def surface_guard_course(runs, course, surface=None):
    has_data = any(r.course or r.surface for r in runs)
    same_course = []; same_surface = []
    for i, run in enumerate(runs):
        if course and run.course == course and (not surface or run.surface == surface):
            same_course.append((scorer._condition_fit_score(run), scorer.get_weight(i)))
        elif surface and run.surface == surface:
            same_surface.append((scorer._surface_fit_score(run), scorer.get_weight(i)))
    a = scorer.weighted_score_from_runs(same_course)
    b = scorer.weighted_score_from_runs(same_surface)
    if a and b:
        return max(a, scorer.clamp_score(a*.7+b*.3))
    return a or b or (1 if has_data else 0)


def distance_evidence(runs, distance):
    return [r for r in runs if r.finish is not None and distance and r.distance and abs(r.distance-distance) <= 400]


def unknown_distance(runs, distance):
    return ORIGINAL_DISTANCE(runs, distance) if distance_evidence(runs, distance) else 5


def unknown_course(runs, course, surface=None):
    available = any((course and r.course == course) or (surface and r.surface == surface) for r in runs)
    return ORIGINAL_COURSE(runs, course, surface) if available else 5


def unknown_going(runs, condition):
    available = any(scorer._track_condition_similarity(condition, r.track_condition) is not None for r in runs)
    return ORIGINAL_GOING(runs, condition) if available else 5


@contextmanager
def variant_context(name, race, saved):
    with ExitStack() as stack:
        stack.enter_context(patch.object(scorer, 'build_pedigree_analyses_for_race', return_value=saved.pedigree_analyses))
        if name == 'context_form':
            stack.enter_context(patch.object(scorer, 'score_recent_form', context_form))
            stack.enter_context(patch.object(scorer, 'score_trend', context_trend))
            stack.enter_context(patch.object(scorer, 'score_risk',
                lambda runs, quality, distance:context_risk(runs, quality, distance, race.race_info)))
        elif name == 'unknown_neutral':
            stack.enter_context(patch.object(scorer, 'score_distance_fit', unknown_distance))
            stack.enter_context(patch.object(scorer, 'score_course_fit', unknown_course))
            stack.enter_context(patch.object(scorer, 'score_track_condition_fit', unknown_going))
        elif name == 'surface_guard':
            stack.enter_context(patch.object(scorer, 'score_course_fit', surface_guard_course))
        yield


def refreshed_prediction(race, saved, name):
    profile, _ = resolve_scoring_profile_config(scoring_profile='accuracy_default')
    config = ScoringConfigSnapshot.model_validate(profile.scoring_config.model_dump())
    config = config.model_copy(update={'use_market_score_in_ranking':False, 'market_signal_weight':0.0})
    with variant_context(name, race, saved):
        return scorer.build_prediction_from_recent_runs_with_scoring_config(
            race, saved.used_lessons, scoring_profile=profile.scoring_profile, scoring_config=config,
            borderline_recovery_enabled=False, scope_reranker_enabled=False)


def evidence_matrix(race, scores):
    from keiba_llm_agent.analysis.race_class import infer_run_race_class_level
    target = date.fromisoformat(race.race_info.race_date)
    info = race.race_info
    rank = {s.horse_no:i for i, s in enumerate(scores)}
    n = len(scores)
    historical = defaultdict(dict)
    for h in race.horses:
        for run in h.recent_runs:
            if run.race_id and run.finish is not None and run.date and run.date < info.race_date:
                historical[run.race_id].setdefault(h.horse_no, run)

    def relevance(run):
        age = max(0, (target-date.fromisoformat(run.date)).days)
        course = 1.0 if run.course == info.course and info.course else .75
        going = scorer._track_condition_similarity(info.track_condition, run.track_condition)
        return context_similarity(run, info)*course*(going if going is not None else .5)*math.exp(-age/365)

    rows = []
    for horse in race.horses:
        unique = {}
        for run in horse.recent_runs:
            if run.finish is not None and run.date and run.date < info.race_date:
                unique.setdefault(run.race_id or (run.date,run.course,run.distance), run)
        valid = list(unique.values())
        related = sorted(valid, key=lambda r:(-relevance(r), r.race_id or ''))[:5]
        weighted = [(r, relevance(r)) for r in related]
        mass = sum(w for r, w in weighted)
        performances = [scorer.score_run_performance(r, info) for r in related]
        mean = sum(s*w for s, (r, w) in zip(performances, weighted, strict=True))/mass if mass else math.nan
        classes = [(infer_run_race_class_level(r), w) for r, w in weighted]
        known = [(level, w) for level, w in classes if level is not None]
        good = sum(w for r, w in weighted if r.finish <= 3 and r.surface == info.surface)
        bad = sum(w for r, w in weighted if r.field_size and r.finish > r.field_size/2 and r.surface == info.surface)
        recent = valid[:3]
        raw_quality = ORIGINAL_FORM(valid, info)
        advantages = []
        for run in valid:
            if not run.race_id:
                continue
            for opponent, other in historical[run.race_id].items():
                if opponent == horse.horse_no:
                    continue
                field = max(run.field_size or 0, other.field_size or 0, 2)
                diff = (other.finish-run.finish)/max(field-1, 1)
                own_margin = comparable_margin(run)
                other_margin = comparable_margin(other)
                margin_diff = other_margin-own_margin if own_margin is not None and other_margin is not None else math.nan
                strength = (n-rank[opponent]-1)/max(n-1, 1)
                advantages.append((diff, relevance(run), opponent, margin_diff, strength))
        head_mass = sum(w for diff, w, opponent, margin, strength in advantages)
        margins = [(m, w) for diff, w, opponent, m, strength in advantages if math.isfinite(m)]
        rows.append([
            mean, max(performances, default=math.nan), min(performances, default=math.nan),
            mass**2/sum(w*w for r, w in weighted) if mass else 0, mass,
            mean-raw_quality if math.isfinite(mean) else math.nan, good, bad,
            sum(bool(r.surface and info.surface and r.surface != info.surface) for r in recent)/max(len(recent), 1),
            sum(level*w for level, w in known)/sum(w for level, w in known) if known else math.nan,
            sum(w for level, w in known)/mass if mass else 0,
            sum(context_similarity(r, info) for r in recent)/len(recent) if recent else 0,
            sum(d*w for d, w, o, m, strength in advantages)/head_mass if head_mass else math.nan,
            sum(m*w for m, w in margins)/sum(w for m, w in margins) if margins else math.nan,
            sum(w for d, w, o, m, strength in advantages if d > 0)/head_mass if head_mass else math.nan,
            head_mass, len({o for d, w, o, m, strength in advantages}),
            sum(d*w*strength for d, w, o, m, strength in advantages)/head_mass if head_mass else math.nan,
            sum(w*max(d, 0) for d, w, o, m, strength in advantages),
            sum(w*max(-d, 0) for d, w, o, m, strength in advantages),
        ])
    matrix = np.asarray(rows, dtype=float)
    assert matrix.shape == (len(race.horses), len(EVIDENCE_NAMES))
    return matrix


def comparable_margin(run):
    if run.finish == 1:
        return 0.0
    value = str(run.margin or '').strip()
    return abs(float(value)) if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', value) else None


def unknown_matrix(race):
    info = race.race_info; rows = []
    for horse in race.horses:
        valid = [r for r in horse.recent_runs if r.finish is not None]
        ds = distance_evidence(valid, info.distance)
        cs = [r for r in valid if r.course == info.course and r.surface == info.surface]
        gs = [r for r in valid if r.surface == info.surface and
              scorer._track_condition_similarity(info.track_condition, r.track_condition) is not None]
        rows.append([int(not ds), int(not cs), int(not gs), len(valid), int(not valid), len(ds), len(cs), len(gs)])
    return np.asarray(rows, dtype=float)


def feature_record(race, prediction, name):
    spec = reranker.MODEL_SPECS[scope_of(race.race_info.model_dump())]
    scores = prediction.horse_scores
    matrix, numbers = reranker.build_scope_reranker_feature_matrix(race, scores, spec['feature_set'])
    extra = evidence_matrix(race, scores) if name == 'relevant_evidence' else unknown_matrix(race) if name == 'unknown_neutral' else None
    if extra is not None:
        matrix = np.column_stack((matrix, extra))
    winner = reranker.build_scope_reranker_feature_matrix(race, scores, spec['baseline_feature_set'])[0] if spec.get('baseline_filename') else None
    return {'matrix':matrix, 'winner_matrix':winner, 'raw':[s.horse_no for s in scores],
            'raw_scores':[float(s.total_score) for s in scores], 'numbers':numbers}


def prepare(snapshot: Path, out: Path):
    print('selecting effective predictions', flush=True)
    predictions = select_effective_prediction_paths(snapshot/'agent_predictions')
    print(f'selecting native audit versions; predictions={len(predictions)}', flush=True)
    native = select_effective_backtest_audit_paths(snapshot/'agent_backtest_audit')
    print(f'selecting derived audits; native={len(native)}', flush=True)
    derived = select_effective_backtest_audit_paths(snapshot/'agent_backtest_audit_rebuilt')
    audits = {**derived, **native}
    known_dates = {}; central = {}
    for rid, path in predictions.items():
        value = read_json(path); info = value.get('race_info') or {}
        known_dates[rid] = info.get('race_date') or ''
        if scope_of(info) in SCOPES:
            central[rid] = (path, info)
    counts = defaultdict(Counter); rows = []; excluded = []; sources = []
    print(f'preparing central inputs; candidates={len(central)}', flush=True)
    for i, (rid, (path, info)) in enumerate(sorted(central.items())):
        scope = scope_of(info); counts[scope]['predictions'] += 1
        result_path = snapshot/'agent_results'/f'{rid}.json'
        result = read_json(result_path) if result_path.exists() else {}
        top3 = podium(result)
        if len(top3) != 3:
            counts[scope]['unsettled_or_invalid_result'] += 1
            continue
        audit = read_json(audits[rid]) if rid in audits else {}
        if audit.get('audit_kind') != 'full' or not audit.get('inputs', {}).get('race_data'):
            counts[scope]['settled_prediction_only_or_missing'] += 1
            continue
        counts[scope]['settled_full_available'] += 1
        pred_payload = read_json(path)
        saved_payload = audit.get('outputs', {}).get('prediction') or {}
        reasons = []
        if prediction_order(saved_payload) != prediction_order(pred_payload):
            reasons.append('audit_prediction_ranking_mismatch')
        race = RaceData.model_validate(audit['inputs']['race_data'])
        saved = Prediction.model_validate(saved_payload)
        target = race.race_info.race_date
        try:
            date.fromisoformat(target or '')
        except ValueError:
            counts[scope]['excluded_invalid_target_date'] += 1
            excluded.append({'race_id':rid, 'scope':scope, 'reasons':['invalid_target_date']})
            continue
        numbers = [h.horse_no for h in race.horses]
        if len(numbers) != len(set(numbers)) or not set(top3) <= set(numbers):
            reasons.append('invalid_runner_roster')
        received = pred_payload.get('prediction_received_at')
        if not received:
            counts[scope]['full_missing_received_at'] += 1
        elif datetime.fromisoformat(received.replace('Z','+00:00')).astimezone(
                timezone(timedelta(hours=9))).date().isoformat() > target:
            reasons.append('received_after_race_day')
        for lesson in saved.used_lessons:
            ids = (lesson.source_race_ids or []) + [lesson.source_race_id]
            if any(known_dates.get(r) and known_dates[r] >= target for r in ids):
                reasons.append('same_or_future_lesson_source')
            if lesson.created_at and lesson.created_at[:10] > target:
                reasons.append('lesson_created_after_race_day')
        if reasons:
            reasons = sorted(set(reasons))
            for reason in reasons:
                counts[scope]['excluded_'+reason] += 1
            excluded.append({'race_id':rid, 'scope':scope, 'reasons':reasons})
            continue
        counts[scope]['eligible_full'] += 1
        counts[scope]['eligible_native_full' if rid in native else 'eligible_derived_full'] += 1
        counts[scope]['eligible_empty_history_races'] += not any(r.finish is not None for h in race.horses for r in h.recent_runs)
        for horse in race.horses:
            clean = []
            for run in horse.recent_runs:
                if not run.date or run.date >= target or run.race_id == rid:
                    counts[scope]['removed_nonhistorical_runs'] += 1
                    continue
                date.fromisoformat(run.date)
                run.odds = None; run.popularity = None
                clean.append(run)
            horse.recent_runs = clean
            horse.odds = None; horse.popularity = None
        variants = {}
        current = refreshed_prediction(race, saved, 'current')
        variants['current'] = feature_record(race, current, 'current')
        for name in CANDIDATES:
            candidate = current if name == 'relevant_evidence' else refreshed_prediction(race, saved, name)
            variants[name] = feature_record(race, candidate, name)
        assert all(v['numbers'] == numbers for v in variants.values())
        assert scorer.score_recent_form is ORIGINAL_FORM and scorer.score_course_fit is ORIGINAL_COURSE
        result_popularity = {r['horse_no']:r.get('popularity') for r in result.get('finish_order', [])}
        rows.append({'race_id':rid, 'date':target, 'scope':scope, 'numbers':numbers, 'top3':top3,
            'winner':top3[0], 'course':race.race_info.course, 'field_size':len(numbers),
            'empty_history':not any(r.finish is not None for h in race.horses for r in h.recent_runs),
            'history_counts':{h.horse_no:sum(r.finish is not None for r in h.recent_runs) for h in race.horses},
            'horse_names':{h.horse_no:h.horse_name for h in race.horses},
            'diagnostic_result_popularity':result_popularity, 'variants':variants})
        sources.append({'race_id':rid, 'audit':str(audits[rid]), 'prediction':str(path), 'result':str(result_path),
                        'audit_sha256':digest(audits[rid]), 'prediction_sha256':digest(path), 'result_sha256':digest(result_path)})
        if len(rows)%100 == 0:
            print(f'prepared={len(rows)} last={rid}', flush=True)
    rows.sort(key=lambda r:(r['scope'], r['date'], r['race_id']))
    manifest = {'snapshot':str(snapshot), 'counts':dict(counts), 'excluded':excluded, 'source_files':sources,
        'eligible_dates':{s:sorted({r['date'] for r in rows if r['scope']==s}) for s in SCOPES},
        'policy':{s:dict(reranker.MODEL_SPECS[s]) for s in SCOPES},
        'production_hashes':production_hashes(), 'experiment_source_sha256':digest(Path(__file__)),
        'scikit_learn_version':sklearn.__version__,
        'training_weight_policy':{
            'central_turf':'positive weight=(runners-positive_count)/positive_count; negatives=1',
            'central_dirt':'equal runner weights with reference loss scale: sum((runners-3)+max(runners-3,1))/total_runners'},
        'market_features_used':False, 'frozen_inputs':['captured pedigree_analyses','as-of compatible used_lessons'],
        'recomputed':['deep_analyses','race_level_analyses','pace_analyses','base scoring','all adjustment terms','central top-choice refinement'],
        'scope':'central_only; local neither trained nor evaluated',
        'time_limit':'Missing received_at retained and flagged; exact pre-off availability cannot be proven without scheduled start and captured_at.',
        'extra_features':{'relevant_evidence':list(EVIDENCE_NAMES), 'unknown_neutral':list(UNKNOWN_NAMES)},
        'validation':'Grouped-date repeated OOF for all eligible races; expanding chronological blocks as a separate guard. No fitted rating graph or outcome-informed feature selection.'}
    joblib.dump({'rows':rows, 'manifest':manifest}, out/'prepared.joblib', compress=3)
    save_json(out/'manifest.json', manifest)
    print(json.dumps({'prepared':len(rows), 'counts':dict(counts)}, indent=2), flush=True)
    return rows, manifest


def templates(scope):
    spec = reranker.MODEL_SPECS[scope]; directory = reranker._model_directory()
    place = joblib.load(directory/spec['filename'])
    winner = joblib.load(directory/spec['baseline_filename']) if spec.get('baseline_filename') else None
    return place, winner


def fit_model(training, variant, scope, template, *, winner=False):
    key = 'winner_matrix' if winner else 'matrix'
    x = np.concatenate([r['variants'][variant][key] for r in training])
    labels = []; weights = []
    for row in training:
        positive = {row['winner']} if winner else set(row['top3'])
        count = len(positive); balanced = max(len(row['numbers'])-count, 1)/count
        for number in row['numbers']:
            y = int(number in positive)
            labels.append(y)
            weights.append(balanced if y and scope == 'central_turf' else 1.0)
    model = clone(template)
    if scope == 'central_dirt':
        # Production preserves the reference loss scale so L2 regularization remains comparable.
        total = sum((len(r['numbers'])-3)+max(len(r['numbers'])-3,1) for r in training)
        weights = np.full(len(labels), total/len(labels))
    model.fit(x, np.asarray(labels), sample_weight=np.asarray(weights))
    return model


def runtime_ranking(row, variant, place_signal, winner_signal=None):
    spec = reranker.MODEL_SPECS[row['scope']]
    values = row['variants'][variant]; ranking = list(values['raw'])
    numbers = row['numbers']
    if len(numbers) < int(spec.get('min_field_size', 0)):
        return ranking
    if winner_signal is not None:
        ranking, _ = reranker._blend_model_ranking(ranking, dict(zip(numbers, winner_signal, strict=True)),
            blend_weight=float(spec['baseline_blend_weight']))
    ranking, _ = reranker._blend_model_ranking(ranking, dict(zip(numbers, place_signal, strict=True)),
        blend_weight=float(spec['blend_weight']), preserve_top_n=int(spec['preserve_top_n']))
    return ranking


def metric_vector(row, ranking):
    selected = set(ranking[:5]); actual = set(row['top3']); count = len(selected & actual)
    winner_rank = ranking.index(row['winner'])+1
    return np.asarray([count, count==3, count>=2, winner_rank==1,
                       ranking[0] in actual, winner_rank<=3, winner_rank<=5, winner_rank], dtype=float)


def summary(vectors):
    values = np.asarray(vectors)
    if values.ndim == 3:
        values = values.mean(axis=0)
    n = len(values); totals = values.sum(axis=0)
    names = ('slots','all3','atleast2','top1_win','top1_place','winner_top3','winner_top5','winner_rank_sum')
    result = dict(zip(names, map(float, totals), strict=True))
    result.update(races=n, recall_pct=float(100*totals[0]/(3*n)), mean_winner_rank=float(totals[7]/n))
    return result


def difference(a, b):
    return {k:b[k]-a[k] for k in a if k != 'races'}


def date_splits(rows, seed, folds=5):
    dates = sorted({r['date'] for r in rows})
    np.random.default_rng(seed).shuffle(dates)
    for part in np.array_split(dates, folds):
        chosen = set(part)
        test = np.asarray([i for i, r in enumerate(rows) if r['date'] in chosen])
        train = np.asarray([i for i, r in enumerate(rows) if r['date'] not in chosen])
        if len(test) and len(train):
            yield train, test


def bootstrap(rows, delta, samples=4000):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['date']].append(i)
    clusters = np.asarray([delta[indices].sum(axis=0) for indices in groups.values()])
    sizes = np.asarray([len(indices) for indices in groups.values()])
    rng = np.random.default_rng(20261005)
    draw = rng.integers(len(clusters), size=(samples, len(clusters)))
    resampled = clusters[draw].sum(axis=1) / sizes[draw].sum(axis=1)[:, None]
    ci = np.quantile(resampled, [.025,.975], axis=0)
    return {'recall_pct':[float(x*100/3) for x in ci[:, 0]],
            'all3_rate_pct':[float(x*100) for x in ci[:, 1]],
            'atleast2_rate_pct':[float(x*100) for x in ci[:, 2]]}


def comparison(rows, baseline, candidate):
    b = np.asarray(baseline); c = np.asarray(candidate)
    avg_delta = (c-b).mean(axis=0)
    bv = summary(b); cv = summary(c)
    slices = {}
    for name, selector in (
        ('field_upto8',lambda r:r['field_size']<=8),
        ('field_9to12',lambda r:9<=r['field_size']<=12),
        ('field_13plus',lambda r:r['field_size']>=13),
        ('empty_history',lambda r:r['empty_history']),
        ('nonempty_history',lambda r:not r['empty_history'])):
        indices = [i for i, r in enumerate(rows) if selector(r)]
        if indices:
            bs = summary(b[:,indices,:]); cs = summary(c[:,indices,:])
            slices[name] = {'baseline':bs, 'candidate':cs, 'delta':difference(bs,cs)}
    return {'baseline':bv, 'candidate':cv, 'delta':difference(bv,cv),
        'repeats':[{'baseline':summary(x), 'candidate':summary(y), 'delta':difference(summary(x),summary(y))}
                   for x, y in zip(b, c, strict=True)],
        'positive_repeats':int(np.sum(c[:,:,0].sum(axis=1) > b[:,:,0].sum(axis=1))),
        'date_clustered_95ci':bootstrap(rows, avg_delta), 'slices':slices}


def evaluate_fold(training, testing, scope, cached_templates):
    place_template, winner_template = cached_templates
    vectors = {}; records = []
    winners = {}; predictions = {}
    for variant in VARIANTS:
        place = fit_model(training, variant, scope, place_template)
        reuse_winner = winner_template is not None and variant == 'relevant_evidence'
        winner = fit_model(training, variant, scope, winner_template, winner=True) if winner_template is not None and not reuse_winner else None
        x = np.concatenate([r['variants'][variant]['matrix'] for r in testing])
        predictions[variant] = place.predict_proba(x)[:,1]
        if winner is not None:
            xw = np.concatenate([r['variants'][variant]['winner_matrix'] for r in testing])
            winners[variant] = winner.predict_proba(xw)[:,1]
        elif reuse_winner:
            winners[variant] = winners['current']
        vectors[variant] = []
    offset = 0
    for row in testing:
        n = len(row['numbers']); part = slice(offset,offset+n); offset += n
        rankings = {}
        for variant in VARIANTS:
            ranking = runtime_ranking(row, variant, predictions[variant][part],
                                      winners[variant][part] if variant in winners else None)
            vectors[variant].append(metric_vector(row, ranking))
            rankings[variant] = ranking
        records.append({'race_id':row['race_id'], 'date':row['date'], 'scope':scope,
                        'top3':row['top3'], 'rankings':rankings})
    return {name:np.asarray(v) for name, v in vectors.items()}, records


def signflip(rows, delta):
    groups = defaultdict(float)
    for row, value in zip(rows, delta, strict=True):
        groups[row['date']] += float(value)
    clusters = np.asarray(list(groups.values()))
    signs = np.random.default_rng(42).choice([-1.0,1.0], size=(20000,len(clusters)))
    return float((1+np.sum(signs@clusters >= clusters.sum()-1e-10))/20001)


def evaluate(rows, manifest, out, seeds):
    all_results = {}; tests = []; all_records = []
    for scope in SCOPES:
        part = [r for r in rows if r['scope']==scope]
        cache = templates(scope)
        repeat = {name:[] for name in VARIANTS}
        for seed in seeds:
            values = {name:np.full((len(part),8), np.nan) for name in VARIANTS}
            seen = np.zeros(len(part), dtype=int)
            for fold, (train, test) in enumerate(date_splits(part, seed)):
                fitted, records = evaluate_fold([part[i] for i in train], [part[i] for i in test], scope, cache)
                for name in VARIANTS:
                    values[name][test] = fitted[name]
                seen[test] += 1
                all_records.extend({'validation':'repeated_oof','seed':seed,'fold':fold, **r} for r in records)
                print(f'{scope} seed={seed} fold={fold} train={len(train)} test={len(test)}', flush=True)
                save_json(out/'progress.json', {'scope':scope,'validation':'repeated_oof','seed':seed,'fold':fold})
            assert np.all(seen==1) and all(np.isfinite(v).all() for v in values.values())
            for name in VARIANTS:
                repeat[name].append(values[name])
        arrays = {name:np.asarray(values) for name, values in repeat.items()}
        results = {name:comparison(part, arrays['current'], arrays[name]) for name in CANDIDATES}
        for name in CANDIDATES:
            delta = (arrays[name][:,:,0]-arrays['current'][:,:,0]).mean(axis=0)
            tests.append({'scope':scope,'candidate':name,'one_sided_date_signflip_p':signflip(part,delta)})
        np.savez_compressed(out/f'{scope}_oof_metrics.npz', **arrays)
        blocks = np.array_split(sorted({r['date'] for r in part}), 5)
        time_rows = []; time_values = {name:[] for name in VARIANTS}; time_blocks = []
        for block in blocks[1:]:
            training = [r for r in part if r['date'] < block[0]]
            testing = [r for r in part if r['date'] in set(block)]
            if not testing or not training:
                continue
            fitted, records = evaluate_fold(training, testing, scope, cache)
            time_rows.extend(testing)
            for name in VARIANTS:
                time_values[name].extend(fitted[name])
            all_records.extend({'validation':'chronological','test_from':str(block[0]), **r} for r in records)
            time_blocks.append({'from':str(block[0]),'through':str(block[-1]),'training_races':len(training),
                'training_through':max(r['date'] for r in training),
                'testing_races':len(testing),'delta':{name:difference(summary(fitted['current']),summary(fitted[name])) for name in CANDIDATES}})
            print(f'{scope} chronological from={block[0]} train={len(training)} test={len(testing)}',flush=True)
        for name in CANDIDATES:
            results[name]['chronological'] = comparison(time_rows, np.asarray([time_values['current']]), np.asarray([time_values[name]]))
            results[name]['chronological']['blocks'] = time_blocks
        all_results[scope] = {'races':len(part),'dates':len({r['date'] for r in part}),
            'current_oof':summary(arrays['current']),'candidates':results,
            'recomputed_logic':True,'production_models_replaced':False}
        save_json(out/f'{scope}_results.json', all_results[scope])
    ordered = sorted(tests, key=lambda t:t['one_sided_date_signflip_p'])
    previous = 0.0
    for rank, entry in enumerate(ordered):
        adjusted = min(1.0,max(previous,entry['one_sided_date_signflip_p']*(len(ordered)-rank)))
        entry['holm_adjusted_p_8_comparisons'] = adjusted
        previous = adjusted
    assert production_hashes() == manifest['production_hashes'], 'Production files changed during experiment'
    save_json(out/'multiple_comparisons.json', {'tests':ordered,
        'caveat':'Approximate date-cluster symmetry; exploratory eight comparisons, not untouched forward evidence.'})
    save_json(out/'oof_race_rankings.json', all_records)
    save_json(out/'results.json', all_results)
    save_json(out/'progress.json', {'status':'complete','production_hashes_unchanged':True})
    for scope, result in all_results.items():
        for name, value in result['candidates'].items():
            print(json.dumps({'scope':scope,'candidate':name,'delta':value['delta'],
                'positive_repeats':value['positive_repeats'], 'ci':value['date_clustered_95ci'],
                'chronological_delta':value['chronological']['delta']},ensure_ascii=True),flush=True)
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Central-only independent evidence/scoring experiments')
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare','evaluate','all'), default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'analysis_outputs').resolve()):
        raise SystemExit('Output must be an independent directory under analysis_outputs')
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in ('prepare','all'):
        if (output/'prepared.joblib').exists():
            raise SystemExit('Refusing to overwrite existing preparation; choose a fresh directory')
        rows, manifest = prepare(args.snapshot.resolve(), output)
    else:
        cached = joblib.load(output/'prepared.joblib')
        rows, manifest = cached['rows'], cached['manifest']
        if digest(Path(__file__)) != manifest['experiment_source_sha256']:
            raise SystemExit('Experiment code changed since preparation')
    if args.stage in ('evaluate','all'):
        if production_hashes() != manifest['production_hashes']:
            raise SystemExit('Production baseline changed since preparation')
        with threadpool_limits(limits=2):
            evaluate(rows, manifest, output, tuple(args.seeds))


if __name__ == '__main__':
    main()
