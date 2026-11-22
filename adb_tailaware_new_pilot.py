"""Pre-registered simple tail-aware yearly importance proposal (heuristic, not quantum).
No sampling of empirical disasters; retain exact ADB yearly PMF.
Uses five official scenarios with 24 replicates at 50k term paths/method.
Design weights fixed before simulated evaluation; not optimized on reference truth.
"""
from pathlib import Path
import time
import numpy as np, pandas as pd
from adb_benchmark_experiment import (
    REFERENCE,TARGETS,STATES,probability_annual,total_distribution,
    metrics_from_distribution,metrics_from_paths,payoff,mc_sample,is_sample
)
OUT=Path('/mnt/data')
BUDGET=50000; REPS=24

def q_tail(p,sc):
    # A conservative 'tail-aware' proposal, with 40% true-mixture support.
    e,r = (60.,20.) if sc=='adb-rain-cover' else (70.,10.)
    l1=e*STATES[:,0]/4.+r*STATES[:,2]
    l2=e*STATES[:,1]/4.+r*STATES[:,3]
    z1=l1/80.;z2=l2/80.
    score = .05 + .25*(z1+z2) + 4.*z1*z2 + ((z1+z2)/2)**2
    tilted=p*score/(p@score)
    q=.4*p+.6*tilted
    assert np.isclose(q.sum(),1.0)
    return q

def sample_ta(p,sc,N,rng):
    q=q_tail(p,sc)
    ix=np.searchsorted(np.cumsum(q),rng.random((N,3)),side='right').clip(0,99)
    w=(p[ix]/q[ix]).prod(axis=1)
    return np.mean(metrics_from_paths(ix,sc)*w[:,None],axis=0)

def main():
    records=[]
    for si,sc in enumerate(REFERENCE):
        p,_=probability_annual(sc)
        truth=metrics_from_distribution(total_distribution(p),sc)
        for mi,(method,fun) in enumerate([('Monte Carlo',mc_sample),('ADB loss-tilted IS',is_sample),('Heuristic tail-aware IS',sample_ta)]):
            for rep in range(REPS):
                rng=np.random.default_rng(20261122+si*100000+rep*101+mi*10_000)
                t=time.perf_counter()
                result=fun(p,sc,BUDGET,rng)
                elapsed=time.perf_counter()-t
                for j,metric in enumerate(TARGETS):
                    records.append(dict(scenario=sc,method=method,replicate=rep,budget=BUDGET,metric=metric,estimate=result[j],exact=truth[j],absolute_error=abs(result[j]-truth[j]),time_sec=elapsed))
    runs=pd.DataFrame(records)
    runs.to_csv(OUT/'adb_tailaware_pilot_runs.csv',index=False)
    summaries=[]
    for (sc,method,metric),g in runs.groupby(['scenario','method','metric']):
        e=g.estimate.to_numpy()-g.exact.to_numpy(); t=g.exact.iloc[0]
        summaries.append(dict(scenario=sc,method=method,metric=metric,nrep=len(g),budget=BUDGET,exact=t,mean_estimate=g.estimate.mean(),relative_rmse_pct=100*np.sqrt(np.mean(e**2))/t,relative_mae_pct=100*np.mean(abs(e))/t,mean_runtime_sec=g.time_sec.mean()))
    summ=pd.DataFrame(summaries)
    summ.to_csv(OUT/'adb_tailaware_pilot_summary.csv',index=False)
    print(summ[summ.metric=='P_both_exhausted'][['scenario','method','relative_rmse_pct','relative_mae_pct','mean_runtime_sec']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nportfolio exceedance')
    print(summ[summ.metric=='P_L_gt_80'][['scenario','method','relative_rmse_pct']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
if __name__=='__main__':main()
