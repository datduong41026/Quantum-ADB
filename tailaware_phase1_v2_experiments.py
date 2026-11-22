"""ADB Phase-1 Tail-aware v2: fixed 5-scenario benchmarks, rare-tail exact variance,
200-replicate paired bootstrap, four ADB baselines, and tensor-guided path IS.
Uses no outside catastrophe data. Never uses reference exact targets to choose proposals.
"""
from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
from scipy.stats import qmc
from adb_benchmark_experiment import (REFERENCE,TARGETS,STATES,N,IDX,BASE_IDX,CAPPED_SUM,probability_annual,total_distribution,metrics_from_distribution,metrics_from_paths,payoff,mc_sample,is_sample)
from adb_tailaware_new_pilot import q_tail, sample_ta
OUT=Path('/mnt/data')


def tensor_annual(p,rank):
    m=p.reshape(5,5,2,2).transpose(0,2,1,3).reshape(10,10)
    u,s,vt=np.linalg.svd(m,full_matrices=False)
    mm=(u[:,:rank]*s[:rank])@vt[:rank]
    mm=np.maximum(mm,0); mm=mm/mm.sum()
    return mm.reshape(5,2,5,2).transpose(0,2,1,3).reshape(100)


def tensor_guided_proposal(p,sc,rank=4,mix=0.9,floor=0.02):
    """Lookahead derived ONLY from a rank-r tensor approx of the annual law.
    q_t(x|s)=(1-mix)*p(x)+mix*p(x)*(H_remaining(s+x)+floor*E_p H)/( (1+floor)*E_p H).
    h recursion is on 100 benchmark states: practical only for small state spaces.
    """
    phat=tensor_annual(p,rank)
    g=payoff(STATES,sc)[:,-1]
    h=[g]
    for r in range(1,3):
        h.append(phat@h[r-1][CAPPED_SUM].T)  # (100,) (each current state)
    qs=[]
    for step in range(3):
        look=h[2-step][CAPPED_SUM] # 100x100 state s and annual outcome x
        norm=look@p
        tilted=p[None,:]*(look+floor*norm[:,None])/(norm[:,None]*(1+floor))
        ok=norm>0
        q=(1-mix)*p[None,:]+mix*tilted
        q[~ok,:]=p
        q=q/np.sum(q,axis=1)[:,None]
        qs.append(q)
    qtab=np.array(qs)
    return qtab, phat


def tgis_sample(p,sc,budget,rng,qtab,rare_only=False):
    state=np.full(budget,BASE_IDX,dtype=np.int16)
    weights=np.ones(budget)
    for step in range(3):
        cdf=np.cumsum(qtab[step],axis=1)
        cdf[:,-1]=1.0
        offsets=2*np.arange(N)
        fullcdf=(cdf+offsets[:,None]).ravel()
        ix=np.searchsorted(fullcdf,offsets[state]+rng.random(budget),side='right')-N*state
        if np.any((ix<0)|(ix>=N)):
            raise RuntimeError('inversion error')
        weights*=p[ix]/qtab[step,state,ix]
        state=CAPPED_SUM[state,ix]
    if rare_only:
        return np.mean(payoff(STATES[state],sc)[:,-1]*weights)
    return np.mean(payoff(STATES[state],sc)*weights[:,None],axis=0)


def strat_sample(p,sc,budget,rng):
    p0=p[BASE_IDX]
    str_probs=(1-p0)*p0**np.arange(3)
    counts=np.floor(budget*str_probs/np.sum(str_probs)).astype(int)
    counts[np.argmax(str_probs)]+=budget-counts.sum()
    out=np.zeros(5)
    pnon=p.copy();pnon[BASE_IDX]=0;pnon/=pnon.sum()
    cnon=np.cumsum(pnon);c=np.cumsum(p)
    for k,n in enumerate(counts):
        if n==0:continue
        ix=np.full((n,3),BASE_IDX,dtype=np.int16)
        ix[:,k]=np.searchsorted(cnon,rng.random(n),side='right').clip(0,N-1)
        if k<2:
            ix[:,k+1:]=np.searchsorted(c,rng.random((n,2-k)),side='right').clip(0,N-1)
        out+=str_probs[k]*np.mean(metrics_from_paths(ix,sc),axis=0)
    return out


def sobol_sample(p,sc,budget,seed,scrambles=8):
    assert (budget//scrambles).bit_count()==1 and budget%scrambles==0
    m=int(np.log2(budget//scrambles))
    c=np.cumsum(p);result=np.zeros(5)
    for j in range(scrambles):
        u=qmc.Sobol(3,scramble=True,seed=int(seed+j*7919)).random_base2(m)
        ix=np.searchsorted(c,u,side='right').clip(0,N-1)
        result+=metrics_from_paths(ix,sc).mean(axis=0)/scrambles
    return result


def exact_single_path_moments(p,sc,qtab):
    """Exact E[W*g] and E[W^2*g^2], under adaptive IS, via backward DP."""
    g=payoff(STATES,sc)[:,-1]
    h=g*g
    for step in range(2,-1,-1):
        weights=(p[None,:]**2)/qtab[step]
        h=np.sum(weights*h[CAPPED_SUM],axis=1)
    second=h[BASE_IDX]
    truth=metrics_from_distribution(total_distribution(p),sc)[-1]
    return truth, max(0,second-truth**2)


def bootstrap_ci(arr, baseline, B=3000,seed=8031):
    """Paired bootstrap over independent rep IDs; ratio < 1 means model improves."""
    arr=np.asarray(arr);baseline=np.asarray(baseline)
    rng=np.random.default_rng(seed)
    ind=rng.integers(0,len(arr),size=(B,len(arr)))
    a=np.sqrt(np.mean(arr[ind]**2,axis=1))
    b=np.sqrt(np.mean(baseline[ind]**2,axis=1))
    return np.quantile(a/b,[.025,.975]),np.mean(a<b)


def main():
    arg=argparse.ArgumentParser();arg.add_argument('--reps',type=int,default=200);arg.add_argument('--budget',type=int,default=32768);a=arg.parse_args()
    methods=['Monte Carlo','ADB IS','Stratified','Scrambled Sobol','Heuristic Tail-IS','Tensor-Guided IS (r4)']
    rows=[];models=[]
    for si,sc in enumerate(REFERENCE):
        p,_=probability_annual(sc)
        exact=metrics_from_distribution(total_distribution(p),sc)
        qtab, phat=tensor_guided_proposal(p,sc,rank=4)
        truth,var=exact_single_path_moments(p,sc,qtab)
        models.append(dict(scenario=sc,rare_exact=truth,variance_tgis=var,theoretical_rel_rmse_32768=np.sqrt(var/a.budget)/truth,
             rank4_l1=np.sum(abs(p-phat)),rank4_rare_bias=metrics_from_distribution(total_distribution(phat),sc)[-1]-truth))
        print(f'[{sc}] exact rare {truth:.8g}, Tensor rank 4 theoretical rmse % {100*np.sqrt(var/a.budget)/truth:.4f}',flush=True)
        t0=time.monotonic()
        for rep in range(a.reps):
            for mi,m in enumerate(methods):
                rng=np.random.default_rng(202611220+si*13000000+rep*1019+mi*110000)
                tick=time.perf_counter()
                if m=='Monte Carlo':res=mc_sample(p,sc,a.budget,rng)
                elif m=='ADB IS':res=is_sample(p,sc,a.budget,rng)
                elif m=='Stratified':res=strat_sample(p,sc,a.budget,rng)
                elif m=='Scrambled Sobol':res=sobol_sample(p,sc,a.budget,202611220+si*13000000+rep*1019+mi*110000)
                elif m=='Heuristic Tail-IS':res=sample_ta(p,sc,a.budget,rng)
                elif m=='Tensor-Guided IS (r4)':res=tgis_sample(p,sc,a.budget,rng,qtab)
                sec=time.perf_counter()-tick
                rows.append(dict(scenario=sc,method=m,replicate=rep,budget=a.budget,runtime_sec=sec,
                                 **{f'est_{name}':res[j] for j,name in enumerate(TARGETS)}))
            if (rep+1)%50==0:print('  replicates',rep+1,'/',a.reps,'elapsed',round(time.monotonic()-t0,1),'sec',flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'tailaware_v2_runs.csv',index=False)
    pd.DataFrame(models).to_csv(OUT/'tailaware_v2_model_audit.csv',index=False)
    summaries=[];compet=[]
    for si,sc in enumerate(REFERENCE):
        p,_=probability_annual(sc);exact=metrics_from_distribution(total_distribution(p),sc)
        for mi,m in enumerate(methods):
            dm=df[(df.scenario==sc)&(df.method==m)].sort_values('replicate')
            for j,tname in enumerate(TARGETS):
                e=dm['est_'+tname].to_numpy()-exact[j]
                relrmse=100*np.sqrt(np.mean(e*e))/exact[j]
                summaries.append(dict(scenario=sc,method=m,metric=tname,exact=exact[j],relative_rmse_pct=relrmse,
                    rel_bias_pct=100*np.mean(e)/exact[j],abs_error=np.mean(abs(e)),mean_runtime_sec=dm.runtime_sec.mean(),reps=a.reps,budget=a.budget))
                if tname=='P_both_exhausted' and m!='ADB IS':
                    b=df[(df.scenario==sc)&(df.method=='ADB IS')].sort_values('replicate')['est_'+tname].to_numpy()-exact[j]
                    ci,pwin=bootstrap_ci(e,b,seed=si*100+mi+551)
                    compet.append(dict(scenario=sc,method=m,metric=tname,rmse_ratio_vs_adb_is=np.sqrt(np.mean(e**2)/np.mean(b**2)),
                         ci_low=ci[0],ci_high=ci[1],bootstrap_p_improvement=pwin))
    pd.DataFrame(summaries).to_csv(OUT/'tailaware_v2_summary.csv',index=False)
    pd.DataFrame(compet).to_csv(OUT/'tailaware_v2_bootstrap.csv',index=False)
    print('RARE RMSE:',flush=True)
    print(pd.DataFrame(summaries).query('metric=="P_both_exhausted"').pivot(index='scenario',columns='method',values='relative_rmse_pct').round(2).to_string(),flush=True)
    print('RARE CI:',flush=True)
    print(pd.DataFrame(compet).query('method=="Tensor-Guided IS (r4)"').to_string(index=False),flush=True)
    (OUT/'tailaware_v2_manifest.json').write_text(json.dumps({'seed_pattern':'202611220+scenario*13000000+replicate*1019+method*110000','reps':a.reps,'path_budget':a.budget,
         'tensor_rank':4,'tail_IS_alpha':0.9,'h_floor':0.02,'sobol_scrambles':8,'bootstrap_reps':3000,
         'data':'ADB 7-page Challenge Brief (2026), Tables 1-5','note':'all methods receive the same target distribution; target-specific tensor proposal uses ADB analytical annual PMF, not exact 3-year truth or held-out outcomes; precomputation favors small benchmark'},indent=2))
if __name__=='__main__':main()
