"""Named experiment suites for version 2. python experiments.py <suite> <worker> <n_workers>
Main table: seeds 0..9. Every other suite: seeds 0..4. Each logged line is one (data, seed,
config, method) run; finished runs are skipped on restart."""
import sys, json, os
import numpy as np
import data, run, methods as M

INF = float('inf')
MAIN = ['Local-GBDT', 'Local-TFM', 'DP-FedAvg', 'DP-FedAvg-FT', 'DP-Synth-GBDT', 'DP-Proto-TFM', 'Fed-HybridPFN', 'Central-TFM']
S3, S5, S10 = range(3), range(5), range(10)

M.METHODS.update({
    'full context': lambda F, k, s: M.run_ours(F, k, s, srule='fixed'),
    'w/o SoftJoin': lambda F, k, s: M.run_ours(F, k, s, softjoin=False),
    'balanced OT': lambda F, k, s: M.run_ours(F, k, s, tau=INF),
    'nearest prototype': lambda F, k, s: M.run_ours(F, k, s, cmode='nearest'),
    'E=1': lambda F, k, s: M.run_ours(F, k, s, ens=1),
    'no posterior': lambda F, k, s: M.run_ours(F, k, s, perturb=False),
    'no local labels': lambda F, k, s: M.run_ours(F, k, s, use_local=False),
    'CV gate': lambda F, k, s: M.run_ours(F, k, s, gate=True),
    'oracle gate': M.run_oracle_gate,
    'Ours E1': lambda F, k, s: M.run_ours(F, k, s, ens=1),
})


def _cfg_gate(F, rel_mode, compat):
    F.cfg['rel_mode'] = rel_mode; F.cfg['compat'] = compat


def lam_gate(F, k, s):
    old = (F.cfg['rel_mode'], F.cfg['compat'])
    _cfg_gate(F, 'lam', 0)
    try:
        return M.run_ours(F, k, s, arrg_gate=True, srule='fixed')
    finally:
        _cfg_gate(F, *old)


def arrg_gate(F, k, s):
    old = (F.cfg['rel_mode'], F.cfg['compat'])
    _cfg_gate(F, 'lam', 1)
    try:
        return M.run_ours(F, k, s, arrg_gate=True, srule='fixed')
    finally:
        _cfg_gate(F, *old)


M.METHODS['lambda gate'] = lam_gate
M.METHODS['ARRG'] = arrg_gate


def jobs(suite):
    J = []
    if suite in ('msweep', 'scale', 'mrule', 'shift'):   # dropped for compute; covered by audits
        return J   # (dataset, seed, overrides, methods, outfile)
    if suite == 'main':
        for sd in S10:
            for d in ['income', 'employment', 'taiwan', 'gmsc']:
                J.append((d, sd, {}, MAIN, 'main'))
    elif suite == 'disjoint':
        for sd in S5:
            for d in ['income', 'employment']:
                J.append((d, sd, dict(disjoint=1), ['Local-TFM', 'DP-FedAvg', 'DP-Proto-TFM', 'Fed-HybridPFN', 'Central-TFM'], 'disjoint'))
    elif suite == 'ablation':
        V = ['full context', 'w/o SoftJoin', 'balanced OT', 'E=1', 'no posterior', 'no local labels']
        for sd in S5:
            for d in ['income', 'taiwan']:
                J.append((d, sd, {}, V, 'ablation'))
                J.append((d, sd, dict(adaptM=0, shrink=0, M=32), ['Fed-HybridPFN'], 'ablation'))
                J.append((d, sd, dict(equal_split=1, adaptM=0, shrink=0, M=32), ['Fed-HybridPFN'], 'ablation'))
    elif suite == 'gates':
        for sd in S5:
            for d in ['taiwan', 'gmsc', 'income']:
                J.append((d, sd, {}, ['CV gate', 'lambda gate', 'ARRG', 'oracle gate'], 'gates'))
    elif suite == 'kappa':
        for sd in S3:
            for d in ['taiwan', 'gmsc', 'income']:
                for kp in [1, 2, 8, 16]:
                    J.append((d, sd, dict(kappa=kp), ['Fed-HybridPFN'], 'kappa'))
    elif suite == 'eps':
        for sd in S3:
            for e in [0.5, 2.0, 4.0, 8.0, INF]:
                J.append(('income', sd, dict(eps=e), ['Fed-HybridPFN', 'DP-Proto-TFM', 'DP-FedAvg'], 'eps'))
    elif suite == 'eps_credit':
        for sd in S3:
            for d in ['taiwan', 'gmsc']:
                for e in [2.0, 4.0, 8.0, 16.0, INF]:
                    J.append((d, sd, dict(eps=e), ['Fed-HybridPFN', 'full context', 'DP-Proto-TFM'], 'eps_credit'))
    elif suite == 'nlab':
        for sd in S3:
            for n in [16, 32, 128, 256]:
                J.append(('income', sd, dict(n_lab=n), ['Fed-HybridPFN', 'DP-Proto-TFM', 'Local-TFM', 'DP-FedAvg-FT'], 'nlab'))
    elif suite == 'ens':
        for sd in S5:
            for d in ['income', 'taiwan']:
                for E in [1, 2, 8]:
                    J.append((d, sd, dict(ens=E), ['Fed-HybridPFN'], 'ens'))
    elif suite == 'shift':
        for sd in S5:
            J.append(('income', sd, dict(sel_strength=2.5), ['Fed-HybridPFN', 'balanced OT', 'nearest prototype', 'w/o SoftJoin'], 'shift'))
    elif suite == 'msweep':
        for sd in S3:
            for d in ['income', 'taiwan']:
                for e in [0.5, 4.0]:
                    for m in [4, 8, 16, 32, 64]:
                        J.append((d, sd, dict(eps=e, M=m, adaptM=0, S_rule='fixed'), ['Ours E1'], 'msweep'))
    elif suite == 'mrule':
        for sd in S5:
            for d in ['income', 'employment', 'taiwan', 'gmsc']:
                J.append((d, sd, dict(mrule='dim'), ['Fed-HybridPFN'], 'mrule'))
    elif suite == 'scale':
        for sd in S3:
            for K in [50, 100]:
                J.append(('income', sd, dict(K=K), ['Fed-HybridPFN', 'DP-Proto-TFM', 'DP-FedAvg'], 'scale'))
    elif suite == 'phase':
        for sd in S3:
            for cap in [2000, 6000, 30000]:
                for e in [0.5, 1.0, 4.0, 16.0]:
                    J.append(('income', sd, dict(cap=cap, eps=e), ['Local-TFM', 'DP-Proto-TFM', 'Ours E1'], 'phase'))
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
                if r['data'] == d and r['seed'] == sd and all(r['cfg'].get(k) == v for k, v in ov.items()) \
                        and all(r['cfg'].get(k) == run.DEFAULT.get(k) for k in ('adaptM', 'equal_split', 'disjoint')
                                if k not in ov):
                    done.add(r['method'])
        meths = [m for m in meths if m not in done]
        if not meths:
            continue
        if d not in cache:
            cache = {d: data.load(d)}
        cfg = run.cfg_for(d, **ov)
        run.run_one(cache[d], cfg, sd, meths, log=f)
