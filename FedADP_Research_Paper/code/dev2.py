import sys, numpy as np, data, run, methods as M
M.METHODS['ARRG snr+compat'] = lambda F,k,s: M.run_ours(F,k,s,arrg_gate=True)
M.METHODS['oracle gate'] = M.run_oracle_gate
def lamgate(F,k,s):
    F.cfg['rel_mode']='lam'; F.cfg['compat']=0
    try: return M.run_ours(F,k,s,arrg_gate=True)
    finally: F.cfg['rel_mode']='snr'; F.cfg['compat']=1
M.METHODS['lambda gate'] = lamgate
for name in sys.argv[1:]:
    D=data.load(name)
    rows=run.run_one(D,run.cfg_for(name),0,['ARRG snr+compat','lambda gate','oracle gate'])
    for r in rows: print('G',name,r['method'],r['gate'],np.round(r['rel'],3),flush=True)
