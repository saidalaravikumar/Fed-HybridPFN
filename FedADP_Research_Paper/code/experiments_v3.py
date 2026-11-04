"""Version 3 suites. The final method (run.DEFAULT) adds the small class shortcut (small_T) and the
resolution weighted local hedge (mix). Usage: python3 experiments_v3.py SUITE WORKER NWORKERS"""
import sys, json, os
import numpy as np
import run, data
import experiments  # noqa: F401  (registers the ablation variants in methods.METHODS)

INF = float('inf')
S3, S5, S10 = range(3), range(5), range(10)
ALL = ['income', 'employment', 'taiwan', 'gmsc']


def jobs(suite):
    J = []   # (dataset, seed, overrides, methods, outfile)
    if suite == 'main3':
        for sd in S10:
            for d in ALL:
                J.append((d, sd, {}, ['Fed-HybridPFN', 'DP-Proto-TFM', 'DP-Synth-GBDT'], 'main3'))
    elif suite == 'abl3':
        V = ['full context', 'w/o SoftJoin', 'balanced OT', 'E=1', 'no posterior', 'no local labels']
        for sd in S5:
            for d in ['taiwan', 'gmsc', 'income']:
                if d == 'income' and sd >= 3:
                    continue
                J.append((d, sd, {}, V, 'abl3'))
                J.append((d, sd, dict(mix=0), ['Fed-HybridPFN'], 'abl3'))
                J.append((d, sd, dict(small_T=0), ['Fed-HybridPFN'], 'abl3'))
    elif suite == 'fact3':
        for sd in S5:
            for d in ['taiwan', 'gmsc', 'income']:
                if d == 'income' and sd >= 3:
                    continue
                for a in (0, 1):
                    for sh in (0, 1):
                        for sp in (0, 1):
                            if a and sh and sp:
                                continue          # the full method is in main3
                            split = (0.4, 0.15, 0.70, 0.15) if sp else (0.4, 1 / 3, 1 / 3, 1 / 3)
                            J.append((d, sd, dict(adaptM=a, shrink=sh, split=split), ['Fed-HybridPFN'], 'fact3'))
    elif suite == 'tfm3':
        for sd in S5:
            for d in ['taiwan', 'gmsc', 'income']:
                if d == 'income' and sd >= 3:
                    continue
                J.append((d, sd, dict(tfm_ckpt='tabicl-classifier-v1.1-20250506.ckpt'),
                          ['Local-TFM', 'DP-Proto-TFM', 'Fed-HybridPFN'], 'tfm3'))
    elif suite == 'tau3':
        for sd in S5:
            for t in [0.1, 0.5, 2.0, 5.0]:
                J.append(('taiwan', sd, dict(tau=t), ['Fed-HybridPFN'], 'tau3'))
    elif suite == 'donors3':
        for sd in S3:
            for nd in [1, 3, 6]:
                J.append(('income', sd, dict(single_donor=nd), ['Fed-HybridPFN', 'DP-Proto-TFM', 'Central-TFM'], 'donors3'))
            J.append(('income', sd, dict(K=40, n_targets=8), ['Fed-HybridPFN', 'Local-TFM'], 'donors3'))
            J.append(('income', sd, dict(K=10, n_targets=4), ['Fed-HybridPFN', 'Local-TFM'], 'donors3'))
    elif suite == 'epsc3':
        for sd in S3:
            for d in ['taiwan', 'gmsc']:
                for e in [2.0, 4.0, 8.0, 16.0, INF]:
                    J.append((d, sd, dict(eps=e), ['Fed-HybridPFN', 'DP-Proto-TFM'], 'epsc3'))
    elif suite == 'kappa3':
        for sd in S3:
            for d in ['taiwan', 'gmsc']:
                for kp in [1, 2, 8, 16]:
                    J.append((d, sd, dict(kappa=kp), ['Fed-HybridPFN'], 'kappa3'))
    return J


if __name__ == '__main__':
    suite, w, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    J = jobs(suite)
    cache = {}
    for i, (d, sd, ov, meths, out) in enumerate(J):
        if i % nw != w:
            continue
        f = '../results/%s.jsonl' % out
        done = set()
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if r['data'] == d and r['seed'] == sd and all(
                        (tuple(r['cfg'].get(k)) if isinstance(v, tuple) else r['cfg'].get(k)) == v for k, v in ov.items()) \
                        and all(r['cfg'].get(k) == run.DEFAULT.get(k) for k in ('mix', 'small_T') if k not in ov):
                    done.add(r['method'])
        meths = [m for m in meths if m not in done]
        if not meths:
            continue
        if d not in cache:
            cache = {d: data.load(d)}
        cfg = run.cfg_for(d, **ov)
        run.run_one(cache[d], cfg, sd, meths, log=f)
