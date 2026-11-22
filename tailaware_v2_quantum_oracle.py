"""Executable 8-qubit coherent ADB-term state-preparation and threshold oracle.
Numpy pure-state simulation with *explicit* multi-controlled R_y rotations;
not transpiled to a hardware native basis and not a QPU measurement.
Independent ideal-noiseless maximum-likelihood amplitude-estimation sampler
uses shot-noise binomials from independently verified Grover response.
"""
import math,time,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from adb_benchmark_experiment import STATES,REFERENCE,probability_annual,total_distribution,metrics_from_distribution,payoff
OUT=Path('/mnt/data')

class ExplicitControlledRotationOracle:
    def __init__(self,term_p,target_value,scale=1.0):
        assert len(term_p)==100
        self.m=7; self.d=1<<self.m
        self.probs=np.pad(term_p,(0,self.d-100))
        assert np.isclose(self.probs.sum(),1.)
        self.f=np.pad(target_value/scale,(0,self.d-100))
        assert np.min(self.f)>=-1e-15 and np.max(self.f)<1.+1e-12
        self.f=np.clip(self.f,0,1)
        self.theta=2*np.arcsin(np.sqrt(self.f))
        self.prepop=[]
        # Each operation is a multi-controlled Ry on a data bit given all higher bits.
        for level in range(self.m):
            for prefix in range(1<<level):
                width=(1 << (self.m-level-1));start=prefix*(2*width)
                mass0=self.probs[start:start+width].sum()
                mass1=self.probs[start+width:start+2*width].sum()
                if mass0+mass1 <= 0:continue
                angle=2*np.arctan2(np.sqrt(mass1),np.sqrt(mass0))
                if abs(angle)<1e-14:continue
                x0=start+np.arange(width);x1=x0+width
                ix0=np.ravel(np.stack([2*x0,2*x0+1],axis=1))
                ix1=np.ravel(np.stack([2*x1,2*x1+1],axis=1))
                self.prepop.append((ix0,ix1,angle))
        self.nprep=len(self.prepop)
        self.npay=int(np.sum(abs(self.theta)>1e-15))
        self.expected=np.dot(term_p,target_value)/scale

    @staticmethod
    def rot(state,idx0,idx1,angle):
        c=np.cos(angle/2);s=np.sin(angle/2)
        a=state[idx0].copy();b=state[idx1].copy()
        state[idx0]=c*a-s*b;state[idx1]=s*a+c*b

    def prep(self,state,inverse=False):
        ops=reversed(self.prepop) if inverse else self.prepop
        for i0,i1,ang in ops:self.rot(state,i0,i1,-ang if inverse else ang)

    def payoff(self,state,inverse=False):
        for x in range(self.d):
            if abs(self.theta[x])>1e-15:
                self.rot(state,[2*x],[2*x+1],-self.theta[x] if inverse else self.theta[x])

    def A(self,state):
        self.prep(state)
        self.payoff(state)

    def Ainv(self,state):
        self.payoff(state,inverse=True)
        self.prep(state,inverse=True)

    def S_good(self,state):state[1::2]*=-1
    def S_zero(self,state):state*= -1;state[0]*=-1

    def iterate(self,state):
        self.S_good(state)
        self.Ainv(state)
        self.S_zero(state)
        self.A(state)

    def check(self,powers=(0,1,2,4,8,16)):
        psi=np.zeros(2*self.d);psi[0]=1.;self.A(psi)
        measured0=np.dot(psi[1::2],psi[1::2])
        inverse=psi.copy();self.Ainv(inverse)
        err_inverse=np.linalg.norm(inverse-np.eye(1,2*self.d,0).reshape(-1))
        theta=np.arcsin(np.sqrt(self.expected))
        rows=[];i=0
        for k in range(max(powers)+1):
            if k in powers:
                prob_good=np.dot(psi[1::2],psi[1::2]);ideal=np.sin((2*k+1)*theta)**2
                rows.append(dict(k=k,probability=prob_good,theory=ideal,abs_err=abs(prob_good-ideal)))
            if k<max(powers):self.iterate(psi)
        return dict(expected=self.expected,p0=measured0,prep_error=abs(measured0-self.expected),inverse_error=err_inverse,
                    norm_error=abs(np.dot(psi,psi)-1),max_grover_error=max(x['abs_err'] for x in rows),
                    n_data_qubits=self.m,n_objective_qubits=1,n_multicontrol_prep=self.nprep,n_multicontrol_payoff=self.npay),rows


def mle_qae_simulation(p,p_upper,powers,shots,reps=200,seed=2069):
    """MLAE oracle measurement, evaluated from the analytically verified Grover angles.
    Search prior bound is declared, not selected from measurement outcomes.
    Probes are simulated Bernoulli experiments, not noisy device circuits.
    """
    rng=np.random.default_rng(seed)
    k=np.asarray(powers);theta=np.arcsin(np.sqrt(p))
    true_probs=np.sin((2*k+1)*theta)**2
    xgrid=np.linspace(0,np.arcsin(np.sqrt(p_upper)),10001)
    phat=np.sin((2*k[:,None]+1)*xgrid[None,:])**2
    phat=np.clip(phat,1e-12,1-1e-12)
    lg=np.log(phat);lg1=np.log1p(-phat)
    estimates=[]
    for j in range(reps):
        y=rng.binomial(shots,true_probs)
        obj=np.sum(y[:,None]*lg+(shots-y)[:,None]*lg1,axis=0)
        im=int(np.argmax(obj));lo=max(0,im-4);hi=min(len(xgrid)-1,im+4)
        if lo!=hi:
            def nll(x):
                pr=np.clip(np.sin((2*k+1)*x)**2,1e-14,1-1e-14)
                return -np.sum(y*np.log(pr)+(shots-y)*np.log1p(-pr))
            res=minimize_scalar(nll,bounds=(xgrid[lo],xgrid[hi]),method='bounded',options={'xatol':1e-12})
            th=res.x if res.success else xgrid[im]
        else:th=xgrid[im]
        estimates.append(np.sin(th)**2)
    est=np.array(estimates)
    queries=int(np.sum(shots*(2*k+1)))
    return {'true_p':p,'shots_each':shots,'powers':','.join(map(str,powers)),'reps':reps,'oracle_calls':queries,
            'rmse_rel_pct':100*np.sqrt(np.mean((est-p)**2))/p,
            'median_rel_abs_error_pct':100*np.median(np.abs(est-p))/p,
            'bias_rel_pct':100*np.mean(est-p)/p,
            'estimated_rel_95pct_interval':np.quantile((est-p)/p,[.025,.975]).tolist(),
            'qae_p_upper':p_upper},est

def main():
    data=[];reps=[];mles=[]
    targets=['P_joint_attach','P_L_gt_80','P_both_exhausted']
    for sc in ['adb-base','adb-independent']:
        p,_=probability_annual(sc)
        term=total_distribution(p)
        fs=payoff(STATES,sc)
        for target,j in [('P_joint_attach',1),('P_L_gt_80',2),('P_both_exhausted',4)]:
            oracle=ExplicitControlledRotationOracle(term,fs[:,j])
            result,series=oracle.check()
            result.update(scenario=sc,target=target)
            data.append(result)
            for a in series:reps.append(dict(scenario=sc,target=target,**a))
            print(sc,target,'p',result['p0'],'prep err',result['prep_error'],'grover err',result['max_grover_error'],flush=True)
        target_p=metrics_from_distribution(term,sc)[4]
        # maximum search probability 1e-3 is fixed for rare-event demonstration.
        # max power 128; multiple phase points allow alias resolution.
        r,est=mle_qae_simulation(target_p,0.001,[0,1,2,4,8,16,32,64,128],96,200,seed=8400+len(data))
        r.update(scenario=sc,target='P_both_exhausted');mles.append(r)
        print('MLAE',sc,r,flush=True)
        np.save(OUT/('tailaware_v2_qae_estimates_'+sc+'.npy'),est)
    pd.DataFrame(data).to_csv(OUT/'tailaware_v2_oracle_verification.csv',index=False)
    pd.DataFrame(reps).to_csv(OUT/'tailaware_v2_grover_checks.csv',index=False)
    pd.DataFrame(mles).to_csv(OUT/'tailaware_v2_qae_ideal_sim.csv',index=False)
if __name__=='__main__':main()
