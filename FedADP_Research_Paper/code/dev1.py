import sys, numpy as np, data, run, methods as M
M.METHODS['Ours ARRG'] = lambda F,k,s: M.run_ours(F,k,s,arrg_gate=True)
M.METHODS['Ours debiased'] = lambda F,k,s: M.run_ours(F,k,s,cost='debiased')
for name in sys.argv[1:]:
    D=data.load(name)
    rows=run.run_one(D,run.cfg_for(name),0,['Local-TFM','Fed-HybridPFN','Ours ARRG','Ours debiased'])
    for r in rows:
        if r['method']=='Ours ARRG': print('ARRG',name,'S_t',r['gate'],'R',np.round(r['rel'],3),flush=True)
