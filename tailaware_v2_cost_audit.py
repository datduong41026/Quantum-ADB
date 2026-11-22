import time
import numpy as np,pandas as pd
from adb_benchmark_experiment import REFERENCE,probability_annual,total_distribution,metrics_from_distribution,low_rank_svd
from tailaware_phase1_v2_experiments import tensor_guided_proposal,exact_single_path_moments
rows=[]
for sc in REFERENCE:
 p,_=probability_annual(sc)
 true=metrics_from_distribution(total_distribution(p),sc)[4]
 for rank in [2,3,4,6]:
  ts=[]
  for _ in range(30):
   t=time.perf_counter();q,ph=tensor_guided_proposal(p,sc,rank=rank);ts.append((time.perf_counter()-t)*1e3)
  truth,v=exact_single_path_moments(p,sc,q)
  est,rel,tv=low_rank_svd(p,sc,rank)
  rows.append(dict(scenario=sc,rank=rank,proposal_setup_msec_median=np.median(ts),rank_probability_l1=np.sum(abs(p-ph)),
       compressed_distribution_rare_rel_bias_pct=100*(est[-1]-true)/true,
       unbiased_tgis_theoretical_rare_rRMSE_pct=100*np.sqrt(v/32768)/true))
 print(sc,[round(rows[-i]['unbiased_tgis_theoretical_rare_rRMSE_pct'],2) for i in (4,3,2,1)])
pd.DataFrame(rows).to_csv('/mnt/data/tailaware_v2_cost_rank_audit.csv',index=False)
