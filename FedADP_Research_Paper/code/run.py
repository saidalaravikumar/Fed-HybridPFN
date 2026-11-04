"""Experiment runner. Usage: python run.py --exp main --data income --seeds 0 1 2 --out ../results/x.jsonl"""
import argparse, json, time, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, log_loss
import data
import methods as Mth

DEFAULT = dict(K=20, n_targets=8, p_feat=0.6, overlap=0.5, n_lab=64, eps=1.0, delta=1e-5,
               M=64, T=3, eta=0.05, tau=1.0, S=1500, n_est=1, sel_strength=1.0, cap=30000, lam=0.6, bw=1.0, two_round=0, beta=0.3, T1=2, adaptM=1, shrink=1, gamma=0.05, Mmax=0, distill=0, n_distill=1500, keep_synth=500, label_mode='sample', align_mode='kernel', pool_mult=4, n_dr=4000, ens=4, perturb=1, align=0, fa_lr=0.02, fa_rounds=60, fa_batch=8192, equal_split=0, gate=0, gate_grid=(0.0, 0.33, 1.0), gate_folds=3, shrink_mode='eb', mrule='linear', cost='plain', arrg=0, disjoint=0, rel_mode='snr', compat=1, codebook=0, Mglob=32, S_rule='resolution', kappa=4, small_T=1, split=(0.4, 0.15, 0.70, 0.15), mix=1, em=0, single_donor=0, tfm_ckpt=None, tscale=0, topk=None)
PER_DATA = {'taiwan': dict(K=10, n_targets=4), 'gmsc': dict(K=20, n_targets=8)}
GROUP_COL = {'income': ('SEX', None), 'employment': ('SEX', None), 'taiwan': ('SEX', None),
             'gmsc': ('AGE', 60.0)}


def group_of(D, Z):
    names = [c['name'] for c in D['schema']]
    col, thr = GROUP_COL[D['name']]
    v = Z[:, names.index(col)]
    return (v >= thr).astype(int) if thr is not None else v.astype(int)


def ece_score(y, p, bins=15):
    e = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(y, P, g):
    from sklearn.metrics import average_precision_score
    p1 = np.clip(P[:, 1], 1e-6, 1 - 1e-6)
    yh = (p1 >= 0.5).astype(int)
    out = dict(auc=roc_auc_score(y, p1) if len(np.unique(y)) > 1 else np.nan,
               auprc=average_precision_score(y, p1),
               f1=f1_score(y, yh, average='macro'),
               nll=log_loss(y, np.column_stack([1 - p1, p1]), labels=[0, 1]),
               brier=float(np.mean((p1 - y) ** 2)), ece=ece_score(y, p1))
    wg, pr, tpr = [], [], []
    for gg in np.unique(g):
        m = g == gg
        if m.sum() > 30 and len(np.unique(y[m])) > 1:
            wg.append(roc_auc_score(y[m], p1[m]))
            pr.append(yh[m].mean()); tpr.append(yh[m][y[m] == 1].mean())
    out['wg_auc'] = min(wg) if wg else np.nan
    out['gap_auc'] = (max(wg) - min(wg)) if len(wg) > 1 else np.nan
    out['dp_gap'] = (max(pr) - min(pr)) if len(pr) > 1 else np.nan      # demographic parity gap
    out['eo_gap'] = (max(tpr) - min(tpr)) if len(tpr) > 1 else np.nan   # equal opportunity gap
    return out


def run_one(D, cfg, seed, method_names, extra=None, log=None):
    silos, enc = data.make_silos(D, K=cfg['K'], n_targets=cfg['n_targets'], p_feat=cfg['p_feat'],
                                 overlap=cfg['overlap'], n_lab=cfg['n_lab'], cap=cfg['cap'],
                                 sel_strength=cfg['sel_strength'], seed=seed, disjoint=bool(cfg['disjoint']))
    Mth.TFM_CKPT = cfg.get('tfm_ckpt')
    F = Mth.Federation(silos, enc, cfg, np.random.default_rng(1000 + seed))
    if cfg.get('single_donor'):
        # restrict the federation to the first `single_donor` label rich agencies
        F.rich = F.rich[:int(cfg['single_donor'])]
    rows = []
    for name in method_names:
        fn = Mth.METHODS[name] if extra is None or name not in extra else extra[name]
        t0 = time.time()
        res = []
        F.gate_log = []; F.rel_log = []; F.var_log = []; F.ctx_log = {}
        for k in F.targets:
            s = silos[k]
            P = fn(F, k, seed)
            res.append(metrics(s['y'][s['test']], P, group_of(D, s['Z'][s['test']])))
        r = {m: float(np.nanmean([x[m] for x in res])) for m in res[0]}
        r.update(method=name, seed=seed, data=D['name'], time=time.time() - t0, gate=list(F.gate_log), rel=list(F.rel_log), pvar=list(F.var_log), ctx=[F.ctx_log.get(k) for k in F.targets],
                 comm=F.comm_bytes, server_t=F.server_t, n_ot=F.n_ot, n_donors=len(F.rich), proto_mse=F.proto_mse, per_silo=[x['auc'] for x in res], per_silo_nll=[x['nll'] for x in res])
        rows.append(r)
        msg = '%s s%d %-14s auc=%.4f f1=%.4f nll=%.4f wg=%.4f t=%.1fs' % (D['name'], seed, name, r['auc'], r['f1'], r['nll'], r['wg_auc'], r['time'])
        print(msg, flush=True)
        if log:
            with open(log, 'a') as f:
                f.write(json.dumps(dict(r, cfg=cfg)) + '\n')
    return rows


def cfg_for(name, **over):
    c = dict(DEFAULT); c.update(PER_DATA.get(name, {})); c.update(over)
    return c


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='income')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--methods', nargs='+', default=list(Mth.METHODS))
    ap.add_argument('--out', default=None)
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split('=')
        over[k] = (v if isinstance(DEFAULT.get(k), str) else type(DEFAULT[k])(float(v))) if k in DEFAULT else float(v)
    D = data.load(a.data)
    cfg = cfg_for(a.data, **over)
    for sd in a.seeds:
        run_one(D, cfg, sd, a.methods, log=a.out)
