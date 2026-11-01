"""Open NEW dates only after the optimizer freezes its model; use exact JS."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

from optimize_simulator_v3 import read,write,aggregate
from prepare_simulator_calibration import candidates,read_json,seconds,valid_date
from prepare_simulator_v3 import FIELDS,clocks

ROOT=Path(__file__).resolve().parents[1]
TEST_SEEDS=[81173+i*104729 for i in range(128)]


def metrics(cases,predictions):
    result=[]
    for i,(c,p) in enumerate(zip(cases,predictions)):
        assert c['race']['race_info']['race_id']==p['race_id']
        labels={h['horse_no']:h for h in c['labels']};truth={h['horse_no'] for h in c['labels'] if h['finish']<=3};winner=next(h['horse_no'] for h in c['labels'] if h['finish']==1)
        order=sorted(p['predictions'],key=lambda h:(-h['top3_count'],h['time'],h['horse_no']));covered=len({h['horse_no'] for h in order[:5]}&truth)
        clock=[];closing=[];single_clock=[];single_closing=[];single_seats=[];crps=[]
        for h in order:
            lab=labels.get(h['horse_no'],{});times=[next(v['time'] for v in run if v['horse_no']==h['horse_no']) for run in p['runs']]
            if lab.get('time') is not None:
                clock.append(abs(h['time']-lab['time']));single_clock.append(sum(abs(t-lab['time']) for t in times)/len(times));ordered=sorted(times);n=len(times)
                crps.append(sum(abs(t-lab['time']) for t in times)/n-sum((2*j-n-1)*t for j,t in enumerate(ordered,1))/n**2)
            if lab.get('final_3f') is not None:closing.append(abs(h['final_3f']-lab['final_3f']))
        for run in p['runs']:single_seats.append(len({h['horse_no'] for h in run[:5]}&truth))
        result.append({'index':i,'race_id':p['race_id'],'scope':c['scope'],'covered':covered,'single_covered':sum(single_seats)/len(single_seats),'winner_rank':next(j+1 for j,h in enumerate(order) if h['horse_no']==winner),'brier':sum((h['top3_count']/p['seeds']-(h['horse_no'] in truth))**2 for h in order)/len(order),'time_error_sum':sum(clock),'time_n':len(clock),'closing_error_sum':sum(closing),'closing_n':len(closing),'single_time_sum':sum(single_clock),'crps_sum':sum(crps)})
    return result


def main(directory,snapshot):
    frozen=directory/'frozen_model.json';receipt=read(directory/'freeze_receipt.json')
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==receipt['model_sha256']
    assert receipt['new_labels_read'] is False
    integrity=read(snapshot/'integrity_verification.json');assert integrity['complete']
    training=read(directory/'training_cases.json');used={c['race']['race_info']['race_id'] for c in training};index=clocks(training)
    cases=[];pending=[];quality={'full_new_date':0,'without_result':0,'unsupported':0,'incomplete_top3':0,'historical_clocks_joined':0}
    for rid,path,audit,sha in candidates(snapshot/'agent_backtest_audit'):
        raw=audit.get('inputs',{}).get('race_data') or {};info=raw.get('race_info') or {}
        if info.get('race_date','')<'2026-11-01':continue
        if audit.get('audit_kind')!='full':continue
        quality['full_new_date']+=1
        if info.get('surface') not in ('芝','ダート') or not 800<=info.get('distance',0)<=3600 or len(raw.get('horses',[]))<5:
            quality['unsupported']+=1;continue
        assert rid not in used
        horses=[]
        for h in raw['horses']:
            identity=str(h.get('horse_id') or h.get('horse_name') or '');runs=[]
            for run in h.get('recent_runs',[]):
                if not valid_date(run.get('date')) or run['date']>=info['race_date'] or str(run.get('race_id'))==rid:continue
                clean={k:run.get(k) for k in FIELDS};historical=index.get((str(run.get('race_id')),identity))
                if historical and historical[:3]==(run['date'],run.get('distance'),run.get('finish')):
                    clean['time_seconds']=historical[3];quality['historical_clocks_joined']+=1
                runs.append(clean)
            horses.append({k:h.get(k) for k in ('horse_no','frame_no','horse_name','jockey','carried_weight')}|{'horse_id':identity,'recent_runs':runs})
        clean_info={k:info.get(k) for k in ('race_id','race_date','race_name','course','surface','distance','track_condition','weather','course_layout','layout','source','scope_key')};clean_info['race_id']=rid
        c={'race':{'race_info':clean_info,'horses':horses},'split':'new_holdout','scope':'local' if info.get('source')=='local' or str(info.get('scope_key','')).startswith('local') else 'central','audit_sha256':sha,'audit_path':str(path)}
        result_file=snapshot/'agent_results'/f'{rid}.json';result=read(result_file) if result_file.exists() else audit.get('outcomes',{}).get('result') or {}
        labels=[];nos={h['horse_no'] for h in horses}
        for row in result.get('finish_order',[]):
            if row.get('horse_no') not in nos or not isinstance(row.get('finish'),int) or row['finish']<1:continue
            t=seconds(row.get('time'));t=t if t and 7<info['distance']/t<22 else None
            labels.append({'horse_no':row['horse_no'],'finish':row['finish'],'time':t,'final_3f':None})
        if not labels:quality['without_result']+=1;pending.append(c);continue
        if len({h['horse_no'] for h in labels if h['finish']<=3})!=3:quality['incomplete_top3']+=1;continue
        c['labels']=labels;cases.append(c)
    cases.sort(key=lambda c:c['race']['race_info']['race_id']);write(directory/'new_cases.json',cases);write(directory/'pending_cases.json',pending)
    protocol={'model_sha256':receipt['model_sha256'],'model_frozen_at':receipt['frozen_at'],'new_labels_opened_at':datetime.now(timezone.utc).isoformat(),'source_captured_at':integrity['captured_at'],'new_date':'2026-11-01','training_overlap':len(used&{c['race']['race_info']['race_id'] for c in cases}),'new_races':len(cases),'pending':len(pending),'scope_counts':{s:sum(c['scope']==s for c in cases) for s in ('central','local')},'quality':quality,'seeds':TEST_SEEDS,'last600_new_labels':0,'caveat':'One partial new race day; retrospective unseen-date holdout, not a model frozen before all races began. No final600 labels available yet.'}
    write(directory/'new_validation_protocol.json',protocol)
    if not cases:print('没有可验证的新赛果；参数已冻结，待新赛果。',flush=True);return
    write(directory/'new_seeds.json',TEST_SEEDS);bridge=ROOT/'scripts/simulator_v3_bridge.mjs';rows={};summary={}
    variants=[('baseline',None),('previous',ROOT/'analysis_outputs/simulator_calibration_20261101/fitted_candidate.json'),('optimized',frozen)]
    for name,model in variants:
        args=['node',str(bridge),'evaluate',str(directory/'new_cases.json'),str(directory/f'new_{name}.json')]
        if model is None:
            write(directory/'baseline_model.json',{});model=directory/'baseline_model.json'
        subprocess.run(args+[str(model),str(directory/'new_seeds.json')],cwd=ROOT,check=True)
        rows[name]=metrics(cases,read(directory/f'new_{name}.json'));summary[name]=aggregate(rows[name]);summary[name]['by_scope']={s:aggregate([r for r in rows[name] if r['scope']==s]) if any(r['scope']==s for r in rows[name]) else {'races':0} for s in ('central','local')}
    write(directory/'new_rows.json',rows);write(directory/'new_summary.json',summary)
    write(directory/'application_decision.json',{'default_enabled':False,'experimental_model_available':True,'reason':'New-day sample is small; preserve independent results and offer explicit experimental comparison. No coefficients changed after opening new labels.'})
    print(json.dumps({'protocol':protocol,'summary':summary},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--snapshot',type=Path,required=True);args=parser.parse_args();main(args.directory,args.snapshot)
