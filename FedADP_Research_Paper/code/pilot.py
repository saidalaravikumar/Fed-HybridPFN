"""Pilot of the small sample refinements on the credit tasks (and a census check)."""
import sys, json, os, run, data
d = sys.argv[1]; seeds = range(int(sys.argv[2]))
V = [dict(small_T=1), dict(mix=1, small_T=1), dict(mix=1)]
f = '../results/pilot.jsonl'
done = set()
if os.path.exists(f):
    for l in open(f):
        r = json.loads(l); c = r['cfg']
        done.add((r['data'], r['seed'], c.get('mix', 0), c.get('small_T', 0), c.get('em', 0)))
D = data.load(d)
for sd in seeds:
    for ov in V:
        if (d, sd, ov.get('mix', 0), ov.get('small_T', 0), 0) in done:
            continue
        run.run_one(D, run.cfg_for(d, **ov), sd, ['Fed-HybridPFN'], log=f)
