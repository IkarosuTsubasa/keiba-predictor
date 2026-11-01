"""Rebuild richer PRIOR features; historical clocks join only by verified identity."""
import argparse
import json
from pathlib import Path

from prepare_simulator_calibration import read_json

FIELDS=('race_id','date','course','surface','distance','track_condition','finish','field_size','corner_positions','final_3f','time_seconds')


def clocks(cases):
    index={}
    for c in cases:
        info=c['race']['race_info'];horses={h['horse_no']:h for h in c['race']['horses']}
        for lab in c['labels']:
            if lab.get('time') and lab['horse_no'] in horses:
                h=horses[lab['horse_no']];index[(info['race_id'],h['horse_id'])]=(info['race_date'],info['distance'],lab['finish'],lab['time'])
    return index


def enrich(cases, protocol, clock_cases):
    index=clocks(clock_cases);manifest={r['race_id']:r for r in protocol['manifest']};base=Path(protocol['snapshot']);count=0
    for c in cases:
        info=c['race']['race_info'];path=Path(manifest[info['race_id']]['audit'])
        if not path.is_absolute():path=base/path
        audit,_=read_json(path);raw={str(h.get('horse_id') or h.get('horse_name') or ''):h for h in audit['inputs']['race_data']['horses']}
        for h in c['race']['horses']:
            runs=[]
            for run in raw[h['horse_id']].get('recent_runs',[]):
                if not isinstance(run.get('date'),str) or not run['date']<info['race_date'] or str(run.get('race_id'))==info['race_id']:continue
                clean={k:run.get(k) for k in FIELDS};historical=index.get((str(run.get('race_id')),h['horse_id']))
                if historical and historical[:3]==(run['date'],run.get('distance'),run.get('finish')):
                    clean['time_seconds']=historical[3];count+=1
                runs.append(clean)
            h['recent_runs']=runs
        c['split']='development';c['fold']=None
    return count


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--previous',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new experiment directory')
    cases=json.loads((args.previous/'cases.json').read_text(encoding='utf-8'));protocol=json.loads((args.previous/'protocol.json').read_text(encoding='utf-8'))
    assert all(c['race']['race_info']['race_date']<'2026-11-01' for c in cases)
    count=enrich(cases,protocol,cases);days=sorted({c['race']['race_info']['race_date'] for c in cases})
    for c in cases:c['fold']=days.index(c['race']['race_info']['race_date'])%5
    args.output.mkdir(parents=True);(args.output/'training_cases.json').write_text(json.dumps(cases,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    metadata={'train_races':len(cases),'training_first_date':days[0],'training_last_date':days[-1],'holdout_first_date':'2026-11-01','historical_clocks_joined':count,'historical_time_join':'Only earlier races; exact race ID, horse identity, date, distance and finish match','source_protocol':str((args.previous/'protocol.json').resolve()),'previous_holdout_reused_as_development':True}
    (args.output/'data_protocol.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print(json.dumps(metadata,indent=2))
