"""Compiled stochastic evaluator; final new-date evaluation uses JS exactly."""
import math
import numpy as np
from numba import njit, prange
from calibrate_simulator import clip


@njit(cache=True)
def uniform(state):
    mask=4294967295
    state=(state+0x6d2b79f5)&mask
    t=((state^(state>>15))*(1|state))&mask
    t=(t^((t+(((t^(t>>7))*(61|t))&mask))&mask))&mask
    return state,((t^(t>>14))&mask)/4294967296.


@njit(cache=True,parallel=True)
def noises(seed,race_ids,counts,stats,spec,legacy=False):
    # stats: variance, inverse effective sample, legacy uncertainty.
    # spec: variance floor/variance weight/estimation weight/pace SD/start/rhythm/form scale
    result=np.zeros((len(counts),stats.shape[1],5))
    for r in prange(len(counts)):
        state=(seed^race_ids[r])&4294967295
        pace=0.
        if not legacy:
            state,a=uniform(state);state,b=uniform(state);pace=(a+b-1)*math.sqrt(6)*spec[3]
        for h in range(counts[r]):
            state,a=uniform(state);state,b=uniform(state)
            if legacy:form=(a+b-1)*(.25+stats[r,h,2]*1.5)
            else:form=(a+b-1)*math.sqrt(6)*math.sqrt(max(.0001,spec[0]+spec[1]*stats[r,h,0]+spec[2]*stats[r,h,1]))*spec[6]
            state,a=uniform(state)
            delay=.15+a*((.3+stats[r,h,2]*1.1) if legacy else spec[4])
            state,a=uniform(state)
            result[r,h]=[form,delay,a*math.pi*2,.14 if legacy else spec[5],pace]
    return result


@njit(cache=True,parallel=True)
def stochastic(parameters,context,profiles,counts,terrain_ids,terrain,noise):
    out=np.zeros((len(context),profiles.shape[1],2))
    for r in prange(len(context)):
        length,turf,local=context[r];n=counts[r]
        base=(parameters[0] if turf else parameters[1])+local*parameters[2]+parameters[3]*math.log(length/1600.)
        nominal=length/base;front_count=0
        for h in range(n):
            if profiles[r,h,4]<2:front_count+=1
        pressure=clip((front_count-2)/6.,0.,.8)
        d=np.zeros(n);v=np.zeros(n);energy=np.ones(n);finished=np.zeros(n,dtype=np.bool_);closing_start=np.full(n,-1.)
        lanes=np.array([.7+h*6.5/max(1,n-1) for h in range(n)])
        for tick in range(1,6000):
            t=tick*.2;leader=max(d)
            for h in range(n):
                if finished[h] or t<noise[r,h,1]:continue
                ability,early,closing,stamina,style=profiles[r,h]
                progress=d[h]/length;late=clip((progress-.66)/.25);early_phase=1.-clip(progress/.42)
                front_style=1.15 if style==0 else .7 if style==1 else -.6 if style==3 else -.15
                early_drive=(front_style+(early-.5)*parameters[5])*early_phase
                flow=(noise[r,h,4]+pressure*.55)*(1.-late*.75)
                back=style==2 or style==3;kick=late*(parameters[6]+closing*parameters[7]+(parameters[8] if back else 0.))
                fatigue=max(0.,.36-energy[h])*(parameters[11]-stamina*2.)
                target_lane=lanes[h] if progress<.12 else (.85 if style==0 else 1.3+(h%4)*.65)+late*(1.2 if back else .15)
                lane=lanes[h]+clip(target_lane-lanes[h],-.07,.07)
                ix=int(d[h]);frac=d[h]-ix;grade=terrain[terrain_ids[r],ix,0]*(1.-frac)+terrain[terrain_ids[r],ix+1,0]*frac
                outer=lane*.055 if terrain[terrain_ids[r],ix,1]>.5 else 0.
                rhythm=math.sin(t*.32+noise[r,h,2])*noise[r,h,3]
                target=base+(ability-.5)*parameters[4]+noise[r,h,0]+early_drive+flow+kick-fatigue-outer-grade*18.+rhythm
                acceleration=2.8+early if t<6. else .85;speed=clip(v[h]+clip(target-v[h],-.3,acceleration*.2),0.,23.)
                effort=max(.6,speed/base);lead=pressure*.15 if d[h]>=leader-4 else 0.
                terrain_energy=clip(1.+max(0.,grade)*16.+min(0.,grade)*4.,.85,1.8)
                energy[h]=clip(energy[h]-parameters[9]/nominal*effort**3.2*(1.14-stamina*parameters[10]+lead)*terrain_energy*.2)
                new_d=d[h]+speed*.2
                if closing_start[h]<0 and new_d>=length-600:closing_start[h]=t-.2+(length-600-d[h])/max(speed,.01)
                if new_d>=length:
                    out[r,h,0]=t-.2+(length-d[h])/max(speed,.01);out[r,h,1]=out[r,h,0]-closing_start[h];finished[h]=True
                d[h]=min(length,new_d);v[h]=speed;lanes[h]=lane
            if np.all(finished):break
    return out
