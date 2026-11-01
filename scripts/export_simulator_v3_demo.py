"""Export a frozen model and input-only new-date examples to the prototype."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);a=p.parse_args()
    fixture=ROOT/'frontend/src/fixtures/raceSimulatorDemo.json';demo=json.loads(fixture.read_text(encoding='utf-8'));cases=json.loads((a.experiment/'new_cases.json').read_text(encoding='utf-8'))
    selected=[]
    for scope in ('central','local'):
        for c in cases:
            if c['scope']==scope and (scope=='local' or not selected or c['race']['race_info']['surface']!=selected[0]['race']['race_info']['surface']):
                selected.append(c)
                if scope=='local' or len(selected)>=2:break
    ids={r['race_info']['race_id'] for r in demo['races']}
    for c in selected:
        race=c['race']
        if race['race_info']['race_id'] in ids:continue
        race['provenance']={'source':'saved_agent_inputs','uses_results':False,'uses_market':False,'validation_role':'unseen_date_input_only'};demo['races'].append(race)
    fixture.write_text(json.dumps(demo,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    model=json.loads((a.experiment/'frozen_model.json').read_text(encoding='utf-8'));(fixture.parent/'raceSimulatorCalibrated.json').write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Exported {len(demo["races"])} input-only examples and frozen model.')
