"""Offline central architecture comparison with fixed, market-free candidates."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import joblib
import numpy as np
from lightgbm import LGBMRanker
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import experiment_central_evidence as e
from keiba_llm_agent.analysis.race_class import CENTRAL_COURSES, infer_race_class_level
from keiba_llm_agent.schemas.race_data import RaceData

VARIANTS = ('current','policy_direct','equal_place','rich_hybrid','rich_direct',
            'raw_direct','rich_ranker','rich_linear')
COURSES = sorted(CENTRAL_COURSES)
WINDOWS = ('last3','last8','same_surface','same_course_distance','recent365')
STATS = ('count','finish_mean','finish_best','finish_worst','finish_std','place_rate',
         'win_rate','margin_mean','margin_best','early_mean','early_std','gain_mean',
         'final3f_mean','class_mean','class_max')
CONTEXT = ('field_size','distance','going','class_explicit','class_missing','history_count',
           'rest_days','weight','frame_fraction','last_distance_delta','last_surface_match',
           'same_jockey','last_finish','last_margin','last_early','last_gain','last_final3f')
RAW_NAMES = tuple(CONTEXT)+tuple('course_'+c for c in COURSES)+tuple(
    w+'_'+s for w in WINDOWS for s in STATS)
CUTOFF = '2026-10-22'


def finite_stat(values, operation):
    good = np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    if not len(good):
        return float('nan')
    return float({'mean':np.mean,'max':np.max,'min':np.min,'std':np.std}[operation](good))


def finish_value(run):
    if run.finish is None or not run.field_size or run.field_size<=1:
        return float('nan')
    return float(np.clip((run.field_size-run.finish)/(run.field_size-1),0,1))


def early_value(run):
    if not run.corner_positions or not run.field_size or run.field_size<=1:
        return float('nan')
    return float(np.clip((run.field_size-run.corner_positions[0])/(run.field_size-1),0,1))


def gain_value(run):
    if not run.corner_positions or not run.field_size or run.field_size<=1 or run.finish is None:
        return float('nan')
    return (run.corner_positions[-1]-run.finish)/(run.field_size-1)


def valid_history(horse, info):
    unique = {}
    for run in horse.recent_runs:
        if run.finish is None or not run.date or run.date>=info.race_date or run.race_id==info.race_id:
            continue
        try:
            date.fromisoformat(run.date)
        except ValueError:
            continue
        unique.setdefault(run.race_id or (run.date,run.course,run.distance),run)
    return sorted(unique.values(),key=lambda r:r.date,reverse=True)


def summarize_history(runs):
    finishes = [finish_value(r) for r in runs]
    margins = [e.comparable_margin(r) for r in runs]
    early = [early_value(r) for r in runs]
    classes = [infer_race_class_level(r.race_name) for r in runs]
    return [len(runs), finite_stat(finishes,'mean'),finite_stat(finishes,'max'),
            finite_stat(finishes,'min'),finite_stat(finishes,'std'),
            np.mean([r.finish<=3 for r in runs]) if runs else np.nan,
            np.mean([r.finish==1 for r in runs]) if runs else np.nan,
            finite_stat(margins,'mean'),finite_stat(margins,'min'),
            finite_stat(early,'mean'),finite_stat(early,'std'),
            finite_stat([gain_value(r) for r in runs],'mean'),
            finite_stat([r.final_3f for r in runs],'mean'),
            finite_stat(classes,'mean'),finite_stat(classes,'max')]


def raw_features(race):
    info = race.race_info
    target = date.fromisoformat(info.race_date)
    explicit = infer_race_class_level(info.race_name)
    rows = []
    for horse in race.horses:
        runs = valid_history(horse,info)
        last = runs[0] if runs else None
        values = [len(race.horses),info.distance,e.scorer._track_condition_level(info.track_condition),
            explicit,int(explicit is None),len(runs),
            (target-date.fromisoformat(last.date)).days if last else None,
            horse.carried_weight,horse.frame_no/8 if horse.frame_no else None,
            last.distance-info.distance if last and last.distance and info.distance else None,
            int(last.surface==info.surface) if last and last.surface and info.surface else None,
            int(last.jockey==horse.jockey) if last and last.jockey and horse.jockey else None,
            finish_value(last) if last else None,e.comparable_margin(last) if last else None,
            early_value(last) if last else None,gain_value(last) if last else None,
            last.final_3f if last else None]
        values.extend(int(info.course==course) for course in COURSES)
        groups = [runs[:3],runs[:8],
                  [r for r in runs if r.surface==info.surface and info.surface][:8],
                  [r for r in runs if r.course==info.course and r.surface==info.surface and
                   r.distance and info.distance and abs(r.distance-info.distance)<=200][:8],
                  [r for r in runs if (target-date.fromisoformat(r.date)).days<=365]]
        for group in groups:
            values.extend(summarize_history(group))
        rows.append([float(v) if v is not None else np.nan for v in values])
    matrix = np.asarray(rows,dtype=float)
    assert matrix.shape==(len(race.horses),len(RAW_NAMES))
    # Keep absolute information and add race-relative comparisons; use no cross-race fitting here.
    relative = np.full_like(matrix,np.nan)
    for j in range(matrix.shape[1]):
        valid = np.isfinite(matrix[:,j])
        if valid.any():
            v = matrix[valid,j]
            relative[valid,j] = (v-v.mean())/max(float(v.std()),1e-8)
    return np.column_stack((matrix,relative))


def prepare(snapshot,out):
    rows, manifest = e.prepare(snapshot,out)
    sources = {r['race_id']:r for r in manifest['source_files']}
    for index,row in enumerate(rows):
        source = Path(sources[row['race_id']]['audit'])
        race = RaceData.model_validate(e.read_json(source)['inputs']['race_data'])
        assert [h.horse_no for h in race.horses]==row['numbers']
        row['raw_features'] = raw_features(race)
        current = row['variants']['current']
        row['rich_features'] = np.column_stack((current['matrix'],row['raw_features']))
        row['variants']['rich'] = {**current,'matrix':row['rich_features']}
        if (index+1)%200==0:
            print(f'continuous features={index+1}',flush=True)
    payload = {'rows':rows,'manifest':manifest,'source_sha256':e.digest(Path(__file__)),
               'raw_feature_names':list(RAW_NAMES)+['relative_'+n for n in RAW_NAMES],
               'variants':list(VARIANTS),'holdout_cutoff':CUTOFF,
               'raw_feature_contract':'No odds/popularity/horse identity/result; strictly prior dated histories, explicit class only.'}
    joblib.dump(payload,out/'architecture.joblib',compress=3)
    e.save_json(out/'architecture_manifest.json',{k:v for k,v in payload.items() if k not in ('rows','manifest')})
    return payload


def labels_weights(rows):
    y = np.asarray([int(n in r['top3']) for r in rows for n in r['numbers']])
    scale = sum((len(r['numbers'])-3)+max(len(r['numbers'])-3,1) for r in rows)/len(y)
    return y,np.full(len(y),scale)


def matrix(rows,kind):
    return np.concatenate([r['variants']['current']['matrix'] if kind=='native'
                           else r['rich_features'] if kind=='rich' else r['raw_features'] for r in rows])


def direct_ranking(numbers,signal):
    assert len(numbers)==len(signal) and np.isfinite(signal).all()
    return sorted(numbers,key=lambda n:(-float(signal[numbers.index(n)]),n))


def ranker():
    return LGBMRanker(objective='lambdarank',n_estimators=180,learning_rate=.06,num_leaves=7,
        min_child_samples=40,reg_lambda=2.0,random_state=42,n_jobs=2,verbosity=-1,
        deterministic=True,force_col_wise=True,label_gain=[0,1],lambdarank_truncation_level=8)


def evaluate_fold(training,testing,scope):
    place_template,winner_template = e.templates(scope)
    baseline = e.fit_model(training,'current',scope,place_template)
    hybrid = e.fit_model(training,'rich',scope,place_template)
    xtrain = {k:matrix(training,k) for k in ('native','rich','raw')}
    xtest = {k:matrix(testing,k) for k in ('native','rich','raw')}
    y,weights = labels_weights(training)
    current_p = baseline.predict_proba(xtest['native'])[:,1]
    signals = {'current':current_p,'policy_direct':current_p,
               'rich_hybrid':hybrid.predict_proba(xtest['rich'])[:,1]}
    for name,kind in (('equal_place','native'),('rich_direct','rich'),('raw_direct','raw')):
        if name=='equal_place' and scope=='central_dirt':
            signals[name] = current_p
            continue
        model = clone(place_template)
        model.fit(xtrain[kind],y,sample_weight=weights)
        signals[name] = model.predict_proba(xtest[kind])[:,1]
    grouped = ranker()
    grouped.fit(xtrain['rich'],y,group=[len(r['numbers']) for r in training])
    signals['rich_ranker'] = grouped.predict(xtest['rich'])
    linear = make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                           StandardScaler(),LogisticRegression(C=.1,max_iter=1500))
    linear.fit(xtrain['rich'],y,logisticregression__sample_weight=weights)
    signals['rich_linear'] = linear.predict_proba(xtest['rich'])[:,1]
    winner_p = None
    if winner_template is not None:
        winner = e.fit_model(training,'current',scope,winner_template,winner=True)
        winner_p = winner.predict_proba(np.concatenate([r['variants']['current']['winner_matrix'] for r in testing]))[:,1]
    vectors = {v:[] for v in VARIANTS}
    records = []
    offset = 0
    calibration = {}
    test_y = labels_weights(testing)[0]
    for v,p in signals.items():
        if v!='rich_ranker':
            calibration[v] = {'n':len(test_y),'squared_error':float(np.sum((p-test_y)**2)),
                              'prediction_sum':float(p.sum()),'actual_sum':int(test_y.sum())}
    for row in testing:
        part = slice(offset,offset+len(row['numbers']))
        offset += len(row['numbers'])
        rankings = {}
        for variant in VARIANTS:
            if variant in ('current','rich_hybrid'):
                ranking = e.runtime_ranking(row,'current',signals[variant][part],
                                             winner_p[part] if winner_p is not None else None)
            else:
                ranking = direct_ranking(row['numbers'],signals[variant][part])
            rankings[variant] = ranking
            vectors[variant].append(e.metric_vector(row,ranking))
        records.append({'race_id':row['race_id'],'date':row['date'],'scope':scope,
                        'top3':row['top3'],'rankings':rankings})
    return {k:np.asarray(v) for k,v in vectors.items()},records,calibration


def evaluate(payload,out):
    rows = payload['rows']; results = {}; all_records = []; hypothesis_tests = []
    for scope in e.SCOPES:
        part = [r for r in rows if r['scope']==scope]
        arrays = {v:[] for v in VARIANTS}; cal = {}
        for seed in e.DEFAULT_SEEDS:
            values = {v:np.full((len(part),8),np.nan) for v in VARIANTS}
            seen = np.zeros(len(part),dtype=int)
            for fold,(train,test) in enumerate(e.date_splits(part,seed)):
                fitted,records,c = evaluate_fold([part[i] for i in train],[part[i] for i in test],scope)
                for v in VARIANTS:
                    values[v][test]=fitted[v]
                seen[test]+=1
                all_records.extend({'validation':'repeated_oof','seed':seed,'fold':fold,**r} for r in records)
                for v,stats in c.items():
                    acc = cal.setdefault(v,{k:0 for k in stats})
                    for k,n in stats.items():
                        acc[k]+=n
                print(f'{scope} architecture seed={seed} fold={fold} test={len(test)}',flush=True)
            assert (seen==1).all() and all(np.isfinite(v).all() for v in values.values())
            for v in VARIANTS:
                arrays[v].append(values[v])
        arrays = {v:np.asarray(a) for v,a in arrays.items()}
        candidates = [v for v in VARIANTS if v!='current' and not (scope=='central_dirt' and v=='equal_place')]
        scores = {v:e.comparison(part,arrays['current'],arrays[v]) for v in candidates}
        np.savez_compressed(out/f'{scope}_architecture_oof.npz',**arrays)
        for v in candidates:
            delta = (arrays[v][:,:,0]-arrays['current'][:,:,0]).mean(axis=0)
            hypothesis_tests.append({'scope':scope,'candidate':v,'p':e.signflip(part,delta)})
        time_rows = []; time_vectors = {v:[] for v in VARIANTS}; time_blocks = []
        for block in np.array_split(sorted({r['date'] for r in part}),5)[1:]:
            training = [r for r in part if r['date']<block[0]]
            testing = [r for r in part if r['date'] in set(block)]
            fitted,records,_ = evaluate_fold(training,testing,scope)
            time_rows.extend(testing)
            for v in VARIANTS:
                time_vectors[v].extend(fitted[v])
            time_blocks.append({'training_through':max(r['date'] for r in training),
                'test_from':str(block[0]),'test_through':str(block[-1]),'train':len(training),'test':len(testing),
                'slot_delta':{v:float(fitted[v][:,0].sum()-fitted['current'][:,0].sum()) for v in candidates}})
            all_records.extend({'validation':'chronological',**r} for r in records)
            print(f'{scope} architecture chronological from={block[0]} test={len(testing)}',flush=True)
        for v in candidates:
            scores[v]['chronological'] = e.comparison(time_rows,np.asarray([time_vectors['current']]),np.asarray([time_vectors[v]]))
        old = [r for r in part if r['date']<=CUTOFF]
        new = [r for r in part if r['date']>CUTOFF]
        fixed = None
        if old and new:
            fitted,records,_ = evaluate_fold(old,new,scope)
            fixed = {'cutoff':CUTOFF,'train':len(old),'test':len(new),
                'dates':sorted({r['date'] for r in new}),
                'candidates':{v:e.comparison(new,np.asarray([fitted['current']]),np.asarray([fitted[v]])) for v in candidates}}
            all_records.extend({'validation':'fixed_oct22_cutoff',**r} for r in records)
        results[scope] = {'races':len(part),'dates':len({r['date'] for r in part}),
            'baseline':e.summary(arrays['current']),'candidates':scores,'time_blocks':time_blocks,
            'fixed_cutoff':fixed,'classifier_signal_calibration':{
                v:{'brier':s['squared_error']/s['n'],'mean_probability':s['prediction_sum']/s['n'],
                   'event_rate':s['actual_sum']/s['n']} for v,s in cal.items()},
            'calibration_caveat':'Current/hybrid are place-component signals, not final mixed ranking probabilities.',
            'duplicate_skipped':'equal_place equals policy_direct for central_dirt' if scope=='central_dirt' else None}
        e.save_json(out/f'{scope}_architecture.json',results[scope])
    maximum = 0
    for i,test in enumerate(sorted(hypothesis_tests,key=lambda r:r['p'])):
        maximum = min(1,max(maximum,(len(hypothesis_tests)-i)*test['p']))
        test['holm_p'] = maximum
    assert e.production_hashes()==payload['manifest']['production_hashes']
    assert e.digest(Path(__file__))==payload['source_sha256']
    e.save_json(out/'architecture_results.json',results)
    e.save_json(out/'architecture_rankings.json',all_records)
    e.save_json(out/'architecture_multiple_comparisons.json',hypothesis_tests)
    print('Architecture comparisons complete; production hashes unchanged.',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('prepare','evaluate','all'),default='all')
    args = parser.parse_args(); out = args.output.resolve()
    if not out.is_relative_to((ROOT/'analysis_outputs').resolve()):
        raise SystemExit('Output must be under analysis_outputs')
    out.mkdir(parents=True,exist_ok=True)
    if args.stage in ('prepare','all'):
        if (out/'prepared.joblib').exists():
            raise SystemExit('Choose a new directory; existing preparation is preserved')
        payload = prepare(args.snapshot.resolve(),out)
    else:
        payload = joblib.load(out/'architecture.joblib')
    assert e.digest(Path(__file__))==payload['source_sha256']
    assert e.production_hashes()==payload['manifest']['production_hashes']
    if args.stage in ('evaluate','all'):
        with threadpool_limits(limits=2):
            evaluate(payload,out)


if __name__ == '__main__':
    main()
