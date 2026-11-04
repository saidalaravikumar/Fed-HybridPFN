import sys, numpy as np, data, run, methods as M
M.METHODS['codebook'] = lambda F,k,s: M.run_ours(F,k,s,cmode='codebook')
for name in sys.argv[1:]:
    D=data.load(name)
    for mg in [16, 32]:
        run.run_one(D,run.cfg_for(name,Mglob=mg),0,['codebook'])
