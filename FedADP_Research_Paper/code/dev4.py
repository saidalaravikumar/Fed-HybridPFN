import sys, numpy as np, data, run, methods as M
for name in sys.argv[1:]:
    D=data.load(name)
    for kap in [2, 4, 8]:
        M.METHODS['res k=%d'%kap] = M.run_ours
        r=run.run_one(D,run.cfg_for(name,S_rule='resolution',kappa=kap),0,['res k=%d'%kap])
        print('S',name,kap,r[0]['gate'][:8],flush=True)
