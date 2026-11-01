"""Fit historical features and residual noise using development races only.

This command never opens the new snapshot or new-race outcome files. A frozen
model is written before the separate new-date validation command may run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from calibrate_simulator import np, dynamics, least_squares, set_num_threads, NAMES as OLD_NAMES, LOWER, UPPER, SCALE
from scipy.optimize import minimize
from simulator_v3_kernel import stochastic, noises

ROOT=Path(__file__).resolve().parents[1]
NAMES=OLD_NAMES+['historicalSpeedWeight','staminaGainWeight','staminaDecayWeight']
CENTRAL=['札幌','函館','福島','新潟','東京','中山','中京','京都','阪神','小倉']
SEEDS=[20261101+i*7919 for i in range(32)]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':')),encoding='utf-8')


def ref_features(info,courses):
    local=info.get('source')=='local' or str(info.get('scope_key','')).startswith('local') or info.get('course') not in CENTRAL
    wet={'良':0,'稍':1,'稍重':1,'重':2,'不良':3}.get(info.get('track_condition'),0)
    return [1,int(info.get('surface')=='芝'),int(local),math.log(info['distance']/1600),wet]+[int(info.get('course')==c) for c in courses]


def references(cases,indices):
    courses=sorted({cases[i]['race']['race_info']['course'] for i in indices});speed_x=[];speed_y=[];close_x=[];close_y=[]
    for i in indices:
        c=cases[i];info=c['race']['race_info'];x=ref_features(info,courses)
        top=[info['distance']/r['time'] for r in c['labels'] if r.get('time') and r['finish']<=3]
        if top:speed_x.append(x);speed_y.append(float(np.mean(top)))
        for row in c['labels']:
            if row.get('final_3f'):close_x.append(x);close_y.append(row['final_3f'])
    out={}
    for key,xx,yy in [('speed',speed_x,speed_y),('closing',close_x,close_y)]:
        x=np.array(xx);y=np.array(yy);penalty=np.array([.001]+[.5]*4+[10.]*len(courses))
        beta=np.linalg.solve(x.T@x+np.diag(penalty),x.T@y)
        for _ in range(3):
            residual=y-x@beta;scale=max(.1,float(np.median(np.abs(residual)))*1.4826);weight=np.minimum(1,1.5*scale/np.maximum(abs(residual),1e-6));beta=np.linalg.solve(x.T@(x*weight[:,None])+np.diag(penalty),x.T@(y*weight))
        out[key]={'courses':courses,'coefficients':beta.tolist()}
    return out


def profile_array(rows,width):
    data=np.zeros((len(rows),width,11))
    keys=['ability','early','closing','stamina','style','positionGain','enduranceSpeed','historicalSpeed','performanceVariance','estimationVariance','uncertainty']
    for r,hs in enumerate(rows):
        for h,p in enumerate(hs):
            for j,key in enumerate(keys):data[r,h,j]=['逃げ','先行','差し','追込','不明'].index(p[key]) if key=='style' else p.get(key,0)
    return data


def effective(parameters,raw,new=True):
    p=parameters[:12].copy();profiles=raw[:,:,:5].copy()
    if new:
        profiles[:,:,0]=.5+parameters[4]*(raw[:,:,0]-.5)+parameters[12]*raw[:,:,7]
        profiles[:,:,3]=np.clip(.5+parameters[13]*raw[:,:,5]+parameters[14]*raw[:,:,6],0,1);p[4]=1.
    return p,profiles


def simulate(parameters,raw,indices,arrays):
    ctx,counts,tids,terrain=arrays;p,profiles=effective(parameters,raw[indices])
    return dynamics(p,ctx[indices],profiles,counts[indices],tids[indices],terrain)


def solve(parameters,raw,indices,arrays,target,label):
    y=target[indices];clock_mask=np.isfinite(y[:,:,0]);close_mask=np.isfinite(y[:,:,1])
    cw=1/np.sqrt(np.maximum(1,clock_mask.sum(axis=1)))[:,None];fw=1/np.sqrt(np.maximum(1,close_mask.sum(axis=1)))[:,None]
    rr,aa,bb=[],[],[]
    for r in range(len(indices)):
        for a in range(arrays[1][indices[r]]):
            if y[r,a,2]<=3:
                for b in range(arrays[1][indices[r]]):
                    if y[r,b,2]>3:rr.append(r);aa.append(a);bb.append(b)
    rr=np.array(rr,dtype=int);aa=np.array(aa,dtype=int);bb=np.array(bb,dtype=int);pw=1/np.sqrt(np.maximum(1,np.bincount(rr,minlength=len(indices)))[rr])
    scale=np.r_[SCALE,[.8,.5,.5]];evaluations=0;start=time.monotonic()
    def residual(x):
        nonlocal evaluations
        pred=simulate(x,raw,indices,arrays);evaluations+=1
        if evaluations%40==0:print(f'{label}: {evaluations}次，{time.monotonic()-start:.1f}秒',flush=True)
        return np.r_[((pred[:,:,0]-y[:,:,0])*cw/3)[clock_mask],((pred[:,:,1]-y[:,:,1])*fw/2)[close_mask],np.logaddexp(0,(pred[rr,aa,0]-pred[rr,bb,0])/1.5)*pw,(x-parameters)/scale*math.sqrt(len(indices)*.01)]
    result=least_squares(residual,parameters,bounds=(np.r_[LOWER,[0,0,0]],np.r_[UPPER,[2.5,1.5,1.5]]),loss='soft_l1',diff_step=.02,x_scale=scale,max_nfev=18,ftol=.005,xtol=.005,gtol=.005)
    print(label,dict(zip(NAMES,result.x)),flush=True);return result.x


def fit_noise(parameters,raw,indices,arrays,target):
    ctx,counts,tids,terrain=arrays;p,profiles=effective(parameters,raw[indices]);n=len(indices);width=raw.shape[1]
    zero=np.zeros((n,width,5));zero[:,:,1]=.15
    base=stochastic(p,ctx[indices],profiles,counts[indices],tids[indices],terrain,zero)
    step=zero.copy();step[:,:,0]=.05;form_sensitivity=(stochastic(p,ctx[indices],profiles,counts[indices],tids[indices],terrain,step)[:,:,0]-base[:,:,0])/.05
    step=zero.copy();step[:,:,4]=.05;pace_sensitivity=(stochastic(p,ctx[indices],profiles,counts[indices],tids[indices],terrain,step)[:,:,0]-base[:,:,0])/.05
    errors=target[indices,:,0]-base[:,:,0];valid=np.isfinite(errors)
    race_error=np.nansum(errors,axis=1)/np.maximum(1,valid.sum(axis=1))
    latent=(errors-race_error[:,None])/np.where(abs(form_sensitivity)>1,form_sensitivity,-1)
    mask=valid&(abs(form_sensitivity)>1);latent=np.clip(latent[mask],-1,1)
    features=np.stack([np.ones_like(raw[indices,:,8]),raw[indices,:,8],raw[indices,:,9]],axis=2)[mask]
    def objective(v):
        variance=np.maximum(.0001,features@v)
        return float(np.mean(.5*(np.log(variance)+latent**2/variance))+.002*np.sum(v[1:]**2))
    fitted=minimize(objective,[.01,.05,.01],bounds=[(.0001,.4),(0,2),(0,.8)],method='L-BFGS-B')
    pace_derivative=np.sum(pace_sensitivity*valid,axis=1)/np.maximum(1,valid.sum(axis=1))
    pace_latent=race_error[np.abs(pace_derivative)>1]/pace_derivative[np.abs(pace_derivative)>1]
    pace_sigma=float(np.clip(1.4826*np.median(abs(pace_latent-np.median(pace_latent))),.02,.8))
    spec={'dayVarianceFloor':float(fitted.x[0]),'performanceVarianceWeight':float(fitted.x[1]),'estimateVarianceWeight':float(fitted.x[2]),'paceSigma':pace_sigma,'startSigma':.12,'rhythmSigma':.04,'formScale':1.}
    return spec


def spec_array(spec):return np.array([spec[k] for k in ['dayVarianceFloor','performanceVarianceWeight','estimateVarianceWeight','paceSigma','startSigma','rhythmSigma','formScale']])


def monte_carlo(parameters,raw,indices,arrays,target,race_ids,spec,seeds=SEEDS,new=True):
    ctx,counts,tids,terrain=arrays;p,profiles=effective(parameters,raw[indices],new);ids=race_ids[indices];nn=counts[indices];stats=raw[indices][:,:,[8,9,10]]
    times=[];closes=[];top=np.zeros(profiles.shape[:2]);single_seats=[]
    truth=target[indices,:,2]<=3;valid=np.arange(profiles.shape[1])[None,:]<nn[:,None]
    for seed in seeds:
        noise=noises(seed,ids,nn,stats,spec_array(spec),not new)
        pred=stochastic(p,ctx[indices],profiles,nn,tids[indices],terrain,noise)
        order=np.argsort(np.where(valid,pred[:,:,0],np.inf),axis=1)
        top[np.arange(len(indices))[:,None],order[:,:3]]+=1
        single_seats.append(np.take_along_axis(truth,order[:,:5],axis=1).sum(axis=1))
        times.append(pred[:,:,0]);closes.append(pred[:,:,1])
    draws=np.array(times);closing_draws=np.array(closes);mean=draws.mean(axis=0);prob=top/len(seeds)
    order=np.lexsort((np.where(valid,mean,np.inf),-np.where(valid,prob,-1)),axis=1)
    seats=np.take_along_axis(truth,order[:,:5],axis=1).sum(axis=1)
    winner=target[indices,:,2]==1;winner_rank=(np.take_along_axis(winner,order,axis=1)*np.arange(1,order.shape[1]+1)[None,:]).sum(axis=1)
    clock_mask=np.isfinite(target[indices,:,0]);close_mask=np.isfinite(target[indices,:,1])
    sorted_draws=np.sort(draws,axis=0);weights=(2*np.arange(1,len(seeds)+1)-len(seeds)-1)[:,None,None]
    crps=np.mean(abs(draws-target[indices,:,0]),axis=0)-np.sum(sorted_draws*weights,axis=0)/len(seeds)**2
    row_metrics=[];single_mean=np.mean(single_seats,axis=0);closing_mean=closing_draws.mean(axis=0);single_clock=np.mean(abs(draws-target[indices,:,0]),axis=0)
    for r,i in enumerate(indices):
        row_metrics.append({'index':int(i),'covered':int(seats[r]),'single_covered':float(single_mean[r]),'winner_rank':int(winner_rank[r]),'brier':float(np.sum((prob[r]-truth[r])**2*valid[r])/nn[r]),'time_error_sum':float(np.sum(abs(mean[r]-target[i,:,0])[clock_mask[r]])),'time_n':int(clock_mask[r].sum()),'closing_error_sum':float(np.sum(abs(closing_mean[r]-target[i,:,1])[close_mask[r]])),'closing_n':int(close_mask[r].sum()),'single_time_sum':float(np.sum(single_clock[r][clock_mask[r]])),'crps_sum':float(crps[r][clock_mask[r]].sum())})
    return row_metrics


def aggregate(rows):
    n=len(rows);tn=sum(r['time_n'] for r in rows);cn=sum(r['closing_n'] for r in rows)
    return {'races':n,'covered_seats':sum(r['covered'] for r in rows),'top3_recall':sum(r['covered'] for r in rows)/(3*n),'all3_races':sum(r['covered']==3 for r in rows),'at_least2_races':sum(r['covered']>=2 for r in rows),'single_run_recall':sum(r['single_covered'] for r in rows)/(3*n),'winner_top3':sum(r['winner_rank']<=3 for r in rows)/n,'winner_top5':sum(r['winner_rank']<=5 for r in rows)/n,'winner_mean_rank':sum(r['winner_rank'] for r in rows)/n,'top3_brier':sum(r['brier'] for r in rows)/n,'mean_time_mae':sum(r['time_error_sum'] for r in rows)/tn,'single_time_mae':sum(r['single_time_sum'] for r in rows)/tn,'closing_mae':sum(r['closing_error_sum'] for r in rows)/cn if cn else None,'time_n':tn,'closing_n':cn,'time_crps':sum(r['crps_sum'] for r in rows)/tn}


def main(directory):
    if (directory/'frozen_model.json').exists():raise ValueError('Already frozen; use a new experiment')
    cases=read(directory/'training_cases.json');assert all(c['race']['race_info']['race_date']<'2026-11-01' for c in cases)
    config={'training_races':len(cases),'new_holdout_first_date':'2026-11-01','new_holdout_read_before_freeze':False,'folds':'5 grouped date folds over all historical full cases; final validation is a later date never used in previous calibration','seeds':SEEDS,'form_scales':[.65,1.,1.35],'selection':'MC Top5/actual Top3 seats, then all3, at_least2; Brier and winner Top5 guardrail; single-run coverage reported separately','unfitted_assumptions':['start delay distribution','rhythm autocorrelation','terrain response','fatigue threshold','routes'],'feature_priors':{'recencyDays':120,'distanceScale':.35,'priorCount':3},'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['scripts/optimize_simulator_v3.py','scripts/simulator_v3_kernel.py','frontend/src/lib/historicalRaceProfiles.js','frontend/src/lib/raceSimulator.js']}}
    write(directory/'fit_protocol.json',config)
    bridge=ROOT/'scripts/simulator_v3_bridge.mjs'
    subprocess.run(['node',str(bridge),'terrain',str(directory/'training_cases.json'),str(directory/'terrain.json')],cwd=ROOT,check=True)
    ter=read(directory/'terrain.json');keys=list(ter['terrain']);maxd=max(len(c) for c in ter['terrain'].values());terrain=np.zeros((len(keys),maxd,2))
    for k,key in enumerate(keys):
        curve=np.array(ter['terrain'][key]);terrain[k,:len(curve)]=curve;terrain[k,len(curve):]=curve[-1]
    n=len(cases);width=max(len(c['race']['horses']) for c in cases);ctx=np.zeros((n,3));counts=np.array([len(c['race']['horses']) for c in cases]);tids=np.array([keys.index(k) for k in ter['keys']]);target=np.full((n,width,3),np.nan);race_ids=np.array([int(c['race']['race_info']['race_id']) for c in cases],dtype=np.int64)
    for r,c in enumerate(cases):
        info=c['race']['race_info'];ctx[r]=[info['distance'],info['surface']=='芝',c['scope']=='local'];labels={x['horse_no']:x for x in c['labels']}
        for h,horse in enumerate(c['race']['horses']):
            lab=labels.get(horse['horse_no'],{});target[r,h]=[lab.get('time') or np.nan,lab.get('final_3f') or np.nan,lab.get('finish') or np.nan]
    arrays=(ctx,counts,tids,terrain);indices=np.arange(n)
    subprocess.run(['node',str(bridge),'profiles',str(directory/'training_cases.json'),str(directory/'baseline_profiles.json')],cwd=ROOT,check=True)
    baseline_raw=profile_array(read(directory/'baseline_profiles.json'),width)
    old=read(ROOT/'analysis_outputs/simulator_calibration_20261101/frozen_model.json')['fitted_parameters'];initial=np.array([old[k] for k in OLD_NAMES]+[.7,.4,.4])
    hand=np.array([16.8,15.8,0,0,1.2,1.5,.55,1.9,.45,.67,.3,5.2]);dummy={'dayVarianceFloor':.01,'performanceVarianceWeight':0,'estimateVarianceWeight':0,'paceSigma':0,'startSigma':.12,'rhythmSigma':.04,'formScale':1}
    rows={'baseline':monte_carlo(hand,baseline_raw,indices,arrays,target,race_ids,dummy,new=False),'previous':monte_carlo(initial[:12],baseline_raw,indices,arrays,target,race_ids,dummy,new=False)}
    for scale in config['form_scales']:rows[str(scale)]=[]
    fold_models=[]
    for fold in range(5):
        train=np.array([i for i,c in enumerate(cases) if c['fold']!=fold]);test=np.array([i for i,c in enumerate(cases) if c['fold']==fold])
        model={'version':'historical-residual-v3','references':references(cases,train),'features':config['feature_priors'],'parameters':dict(zip(NAMES,initial.tolist())),'randomness':dummy}
        write(directory/f'fold_{fold}_reference.json',model)
        subprocess.run(['node',str(bridge),'profiles',str(directory/'training_cases.json'),str(directory/f'fold_{fold}_profiles.json'),str(directory/f'fold_{fold}_reference.json')],cwd=ROOT,check=True)
        raw=profile_array(read(directory/f'fold_{fold}_profiles.json'),width);x=solve(initial,raw,train,arrays,target,f'第{fold+1}折');noise=fit_noise(x,raw,train,arrays,target)
        model['parameters']=dict(zip(NAMES,x.tolist()));model['randomness']=noise;fold_models.append(model);write(directory/f'fold_{fold}_model.json',model)
        for scale in config['form_scales']:
            spec={**noise,'formScale':scale};rows[str(scale)]+=monte_carlo(x,raw,test,arrays,target,race_ids,spec)
        print('开发集当前结果',fold,{k:aggregate(v) for k,v in rows.items() if len(v)==n or k not in ['baseline','previous']},flush=True)
    scores={k:aggregate(v) for k,v in rows.items()};base=scores['baseline']
    eligible=[k for k in rows if k not in ('baseline','previous') and scores[k]['top3_brier']<=base['top3_brier'] and scores[k]['winner_top5']>=base['winner_top5']-.01]
    chosen=max(eligible or [str(s) for s in config['form_scales']],key=lambda k:(scores[k]['covered_seats'],scores[k]['all3_races'],scores[k]['at_least2_races'],-scores[k]['top3_brier']))
    model={'version':'historical-residual-v3','references':references(cases,indices),'features':config['feature_priors'],'parameters':dict(zip(NAMES,initial.tolist())),'randomness':dummy}
    write(directory/'full_reference.json',model)
    subprocess.run(['node',str(bridge),'profiles',str(directory/'training_cases.json'),str(directory/'full_profiles.json'),str(directory/'full_reference.json')],cwd=ROOT,check=True)
    raw=profile_array(read(directory/'full_profiles.json'),width);x=solve(initial,raw,indices,arrays,target,'全量历史');noise=fit_noise(x,raw,indices,arrays,target);noise['formScale']=float(chosen)
    model.update(parameters=dict(zip(NAMES,x.tolist())),randomness=noise,training_last_date='2026-10-31',development_choice=chosen,development_eligible=chosen in eligible)
    # Exact JS agreement is checked on development races, before opening new labels.
    sample_ids=np.arange(0,n,max(1,n//30));sample=[cases[i] for i in sample_ids];write(directory/'parity_cases.json',sample);write(directory/'candidate_model.json',model);write(directory/'parity_seeds.json',[SEEDS[0],SEEDS[1]])
    subprocess.run(['node',str(bridge),'evaluate',str(directory/'parity_cases.json'),str(directory/'parity_exact.json'),str(directory/'candidate_model.json'),str(directory/'parity_seeds.json')],cwd=ROOT,check=True)
    exact=read(directory/'parity_exact.json');p,profiles=effective(x,raw[sample_ids]);differences=[]
    for s,seed in enumerate(SEEDS[:2]):
        noise_array=noises(seed,race_ids[sample_ids],counts[sample_ids],raw[sample_ids][:,:,[8,9,10]],spec_array(noise),False);value=stochastic(p,ctx[sample_ids],profiles,counts[sample_ids],tids[sample_ids],terrain,noise_array)
        for rr,i in enumerate(sample_ids):
            by_no={h['horse_no']:h for h in exact[rr]['runs'][s]}
            for h,horse in enumerate(cases[i]['race']['horses']):differences.append(abs(value[rr,h,0]-by_no[horse['horse_no']]['time']))
    parity={'mean_seconds':float(np.mean(differences)),'max_seconds':float(max(differences)),'horse_runs':len(differences)}
    if parity['max_seconds']>.5:raise ValueError(f'Compiled/JS mismatch: {parity}')
    write(directory/'development_rows.json',rows);write(directory/'development_summary.json',scores);write(directory/'parity.json',parity)
    write(directory/'frozen_model.json',model)
    write(directory/'freeze_receipt.json',{'model_sha256':hashlib.sha256((directory/'frozen_model.json').read_bytes()).hexdigest(),'frozen_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'new_labels_read':False,'new_snapshot_not_opened_by_optimizer':True})
    print('已冻结',chosen,scores,'原版一致性',parity,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args();set_num_threads(8);main(args.directory)
