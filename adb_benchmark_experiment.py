"""Reproducible ADB Joint-Loss Challenge pilot.
Source: ADB Quantum Innovation Challenge Brief, pages 2-5.
Uses only the numerical model disclosed in the brief, no third-party disaster observations.
Outputs: benchmark values, SVD low-rank experiments, seeded Monte Carlo and importance sampling.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import expm

OUT = Path('/mnt/data')
# Exact data as presented in ADB Table 1 (percent to probability).
TAIL = np.array([[.03883,.02331,.01790,.01184],[.03925,.01743,.01332,.01043]],float)
REFERENCE = {
'adb-base': [10.06, .04414, .01205, 282355., .000032],
'adb-independent': [10.06, .02716, .00719, 171266., .000004],
'adb-common-shock': [10.06, .05053, .01607, 385043., .000044],
'adb-rain-stress': [11.17, .07046, .01322, 306286., .000069],
'adb-rain-cover': [9.97, .04414, .00700, 170927., .000032],
}
TARGETS = ['EL_mUSD','P_joint_attach','P_L_gt_80','stop_loss_USD','P_both_exhausted']
STATES = np.array([(a,b,r1,r2) for a in range(5) for b in range(5) for r1 in range(2) for r2 in range(2)],dtype=int)
IDX = {tuple(v):i for i,v in enumerate(STATES)}
N=100
CAPPED_SUM = np.empty((N,N),dtype=np.int16)
for i,s in enumerate(STATES):
    for j,t in enumerate(STATES):
        z=np.minimum(s+t,[4,4,1,1]); CAPPED_SUM[i,j] = IDX[tuple(z)]
BASE_IDX=IDX[(0,0,0,0)]

def rates(tails):
    t1,t2,t3,t4 = tails
    p0=1-t1; p1=t1-t2; p2=t2-t3; p3=t3-t4
    lam0=-np.log(p0)
    lam1=p1/p0
    lam2=p2/p0 - .5*lam1**2
    lam3=p3/p0 - lam1*lam2 - (lam1**3)/6
    lam4=lam0-lam1-lam2-lam3
    return np.array([lam1,lam2,lam3,lam4]),lam0

RATES=[rates(row) for row in TAIL]

def probability_annual(scenario='adb-base'):
    la1,L1=RATES[0]; la2,L2=RATES[1]
    w1=la1/L1; w2=la2/L2
    multiplier = {'adb-independent':0., 'adb-common-shock':2.}.get(scenario,1.)
    c=multiplier*(.075*L1+.074*L2)/2
    G=np.zeros((25,25),float)
    for a in range(5):
      for b in range(5):
        si=5*a+b
        # only Kyrgyz
        for k,w in enumerate(w1,1):
            tj=5*min(a+k,4)+b; rate=(L1-c)*w
            if tj!=si: G[si,tj]+=rate; G[si,si]-=rate
        # only Tajik
        for k,w in enumerate(w2,1):
            tj=5*a+min(b+k,4); rate=(L2-c)*w
            if tj!=si: G[si,tj]+=rate; G[si,si]-=rate
        # shared earthquake
        for k,wk in enumerate(w1,1):
          for j,wj in enumerate(w2,1):
            tj=5*min(a+k,4)+min(b+j,4); rate=c*wk*wj
            if tj!=si: G[si,tj]+=rate; G[si,si]-=rate
    peq=expm(G)[0,:].reshape(5,5)
    assert np.min(peq)>-1e-12
    rate_r=-np.log(.98)*(2 if scenario=='adb-rain-stress' else 1)
    cr=rate_r*(0 if scenario=='adb-independent' else .25)
    P00=np.exp(-(2*rate_r-cr))
    P10=np.exp(-rate_r)-P00
    P01=np.exp(-rate_r)-P00
    P11=1-P00-P10-P01
    pr=np.array([[P00,P01],[P10,P11]])
    p=(peq[:,:,None,None]*pr[None,None,:,:]).flatten()
    assert abs(p.sum()-1)<1e-11 and np.min(p)>-1e-13
    return p, {'eq_shared_rate':c,'rain_shared_rate':cr,'p_annual_zero':p[BASE_IDX]}

def total_distribution(p,years=3):
    cur=np.zeros(N,float); cur[BASE_IDX]=1.
    for year in range(years):
        # distribution of capped total states after this year
        cur=np.bincount(CAPPED_SUM.flatten(), weights=np.multiply.outer(cur,p).flatten(),minlength=N)
    assert abs(cur.sum()-1)<1e-10
    return cur

def payoff(states, scenario='adb-base'):
    e,r=(60.,20.) if scenario=='adb-rain-cover' else (70.,10.)
    l1=e*states[...,0]/4. + r*states[...,2]
    l2=e*states[...,1]/4. + r*states[...,3]
    return np.column_stack([l1+l2, ((l1>0)&(l2>0)).astype(float), ((l1+l2)>80).astype(float), np.maximum(l1+l2-80,0)*1e6, ((np.isclose(l1,80))&(np.isclose(l2,80))).astype(float)])

def metrics_from_distribution(p,scenario):
    x = payoff(STATES,scenario)
    res = p@x
    res[0]=res[0] # million USD
    return res

def metrics_from_paths(idx,scenario):
    a=STATES[idx]
    summed=np.minimum(a.sum(axis=1),[4,4,1,1])
    return payoff(summed,scenario)

def mc_sample(p, scenario, budget, rng):
    u=rng.random((budget,3))
    idx=np.searchsorted(np.cumsum(p),u,side='right').clip(0,N-1)
    return metrics_from_paths(idx,scenario).mean(axis=0)

def is_sample(p,scenario,budget,rng):
    # ADB disclosed mixture proposal, with total annual payout as the tilt.
    loss=payoff(STATES,scenario)[:,0]
    tilted=p*loss / (p@loss)
    q=.5*p+.5*tilted
    idx=np.searchsorted(np.cumsum(q),rng.random((budget,3)),side='right').clip(0,N-1)
    weight=(p[idx]/q[idx]).prod(axis=1)
    return np.mean(metrics_from_paths(idx,scenario)*weight[:,None],axis=0)

def rb_sample(p,scenario,budget,rng):
    # New prototype: Rao-Blackwellize over 3rd annual state.
    # Each sample pays for up to 100 conditional payoff evaluations.
    idx=np.searchsorted(np.cumsum(p),rng.random((budget,2)),side='right').clip(0,N-1)
    capped=np.minimum(STATES[idx[:,0]]+STATES[idx[:,1]],[4,4,1,1])
    # cache all 100 possible 2-year cumulative states -> expectation over last year
    q=np.zeros((N,5),float)
    for k in range(N):
       dest=CAPPED_SUM[k]
       q[k]=p@payoff(STATES[dest],scenario)
    key=np.fromiter((IDX[tuple(s)] for s in capped),dtype=int,count=budget)
    return q[key].mean(axis=0)

def low_rank_svd(p, scenario, rank):
    # Reshape as (Kyrgyz earthquake state, rain flag) x (Tajik earthquake state, rain flag)
    mat=p.reshape(5,5,2,2).transpose(0,2,1,3).reshape(10,10)
    U,s,Vt=np.linalg.svd(mat,full_matrices=False)
    trunc=(U[:,:rank]*s[:rank])@Vt[:rank]
    # rank truncation can introduce tiny negative entries, project to a probability simplex
    trunc=np.maximum(trunc,0.0)
    trunc/=trunc.sum()
    p_approx=trunc.reshape(5,2,5,2).transpose(0,2,1,3).reshape(N)
    return metrics_from_distribution(total_distribution(p_approx),scenario),float(np.linalg.norm(mat-trunc)/np.linalg.norm(mat)),float(np.sum(np.abs(mat-trunc)))/2

def main():
    import argparse
    arg=argparse.ArgumentParser();arg.add_argument('--budget',type=int,default=60000);arg.add_argument('--replicates',type=int,default=8)
    cfg=arg.parse_args()
    data=[];details={}
    for i,scenario in enumerate(REFERENCE):
        p,info=probability_annual(scenario)
        exact=metrics_from_distribution(total_distribution(p),scenario)
        target=np.array(REFERENCE[scenario]);
        row={'scenario':scenario,**info}
        for j,name in enumerate(TARGETS):
            row['model_'+name]=exact[j]
            row['ADB_'+name]=target[j]
            row['abs_diff_'+name]=abs(exact[j]-target[j])
        data.append(row)
        print(scenario,'computed',np.round(exact,10),'ADB published',target)
        details[scenario]={'exact':exact.tolist(),'rates_eq':[v[0].tolist() for v in RATES], 'conditional_rain':info}
    df=pd.DataFrame(data);df.to_csv(OUT/'adb_exact_results.csv',index=False)
    scenarios=['adb-base','adb-common-shock']
    draws=[]
    for si,scenario in enumerate(scenarios):
        p,info=probability_annual(scenario)
        exact=metrics_from_distribution(total_distribution(p),scenario)
        for method in ['Monte Carlo','Importance sampling','Conditional integration']:
          for rep in range(cfg.replicates):
            rng=np.random.default_rng(20261122+si*10000+rep*97+['Monte Carlo','Importance sampling','Conditional integration'].index(method)*1000)
            tick=time.perf_counter()
            estimate={'Monte Carlo':mc_sample,'Importance sampling':is_sample,'Conditional integration':rb_sample}[method](p,scenario,cfg.budget,rng)
            sec=time.perf_counter()-tick
            for j,metric in enumerate(TARGETS):
                draws.append({'scenario':scenario,'method':method,'replicate':rep,'budget':cfg.budget,'metric':metric,'estimate':estimate[j],'exact':exact[j],'absolute_error':abs(estimate[j]-exact[j]),'time_sec':sec})
            print('run',scenario,method,rep,'time',round(sec,3))
    ds=pd.DataFrame(draws);ds.to_csv(OUT/'adb_sampling_runs.csv',index=False)
    summary=(ds.groupby(['scenario','method','metric'],as_index=False)
       .agg(mean_estimate=('estimate','mean'),mean_absolute_error=('absolute_error','mean'),rmse=('estimate',lambda s: np.sqrt(np.mean((s-ds.loc[s.index,'exact'])**2))),mean_runtime_sec=('time_sec','mean'),exact=('exact','mean')))
    summary.to_csv(OUT/'adb_sampling_summary.csv',index=False)
    ranks=[]
    for scenario in REFERENCE:
        p,_=probability_annual(scenario)
        ex=metrics_from_distribution(total_distribution(p),scenario)
        for rank in [1,2,3,4,6,8,10]:
            ap,relativeFro,tv=low_rank_svd(p,scenario,rank)
            d={'scenario':scenario,'rank':rank,'Fro_relative_error':relativeFro,'TV_upper_est':tv}
            for j,name in enumerate(TARGETS):
                d['estimate_'+name]=ap[j];d['abs_error_'+name]=abs(ap[j]-ex[j]);
            ranks.append(d)
    rk=pd.DataFrame(ranks);rk.to_csv(OUT/'adb_lowrank_results.csv',index=False)
    print('\nSUBSET SAMPLING RESULTS:')
    print(summary[summary.metric.isin(['P_L_gt_80','P_both_exhausted'])].to_string(index=False))
    print('\nLOW RANK:')
    print(rk[rk.scenario=='adb-base'][['rank','Fro_relative_error','abs_error_P_joint_attach','abs_error_P_L_gt_80','abs_error_P_both_exhausted']].to_string(index=False))
    with open(OUT/'adb_experiment_metadata.json','w') as f:
        json.dump({'budget_per_method_per_replicate':cfg.budget,'replicates':cfg.replicates,'seed_base':20261122,'implementation':'direct CTMC matrix exponential; exact 3 year convolution; sampled MC/ADB importance proposal; Rao-Blackwell conditional expectation over last year; approximate low rank SVD annual joint distribution','notes':'Reference values ADB brief Table5 rounded; all simulations use ADB parameters, not new disaster observations'},f,indent=2)

if __name__=='__main__':main()
