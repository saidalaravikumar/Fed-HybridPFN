"""Label free audits that need no foundation model calls.

  shrink   : MSE of released prototype means for five estimators, vs class size (Sec. on release)
  mrule    : release error of fixed, linear, dimension adaptive and oracle prototype counts
  cmi      : plug in estimates of I(U;V|S) and I(U;V|S,Y) for every pair of non bridge columns
  shift    : completion error, transported mass and marginal violation vs selection strength
"""
import sys, json, os
import numpy as np
import data, run, methods as Mth, fedhybrid as fh

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
INF = float('inf')


def _class_records(name, y=1, seed=0):
    D = data.load(name)
    enc = data.Encoder(D['schema'])
    cols = list(range(len(D['schema'])))
    rng = np.random.default_rng(seed)
    idx = np.where(D['y'] == y)[0]
    idx = rng.permutation(idx)[:60000]
    E = enc.encode(D['Z'][idx])
    nm = np.zeros(enc.dim, bool)
    for j, c in enumerate(D['schema']):
        if c['type'] == 'num':
            nm[enc.slices[j]] = True
    return E, nm, len(cols)


def release_error(E, r):
    """Coupling cost of Theorem 2: each record moves to the released mean of its own cell, or
    to the nearest released mean when its cell was filtered out."""
    if len(r['n']) == 0:
        return np.nan
    a, keep = r['assign'], r['keep']
    kept_ids = np.where(keep)[0]
    pos = -np.ones(len(keep), int); pos[kept_ids] = np.arange(len(kept_ids))
    mu = r['mu']
    cost = np.empty(len(E))
    ok = pos[a] >= 0
    cost[ok] = np.linalg.norm(E[ok] - mu[pos[a[ok]]], axis=1)
    if (~ok).any():
        d2 = (E[~ok] ** 2).sum(1)[:, None] - 2 * E[~ok] @ mu.T + (mu ** 2).sum(1)[None]
        cost[~ok] = np.sqrt(np.maximum(d2.min(1), 0))
    w = r['n'] / r['n'].sum(); wt = r['n_true'] / max(r['n_true'].sum(), 1)
    D = np.sqrt(E.shape[1])
    return float(cost.mean() + D * 0.5 * np.abs(w - wt).sum())


def m0_of(C2, rho, gamma=0.05):
    r = 0.98 * rho
    return fh.gauss_sigma(np.sqrt(C2), 0.70 * 0.6 * r) / gamma


def shrink_study(name='income', reps=30):
    E_all, nm, C2 = _class_records(name)
    sizes = [50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000]
    res = {}
    for eps in [1.0]:
        rho = fh.rho_from_eps(eps, 1e-5)
        m0 = m0_of(C2, rho)
        for n in sizes:
            for rep in range(reps):
                rng0 = np.random.default_rng(1000 * rep + n)
                E = E_all[rng0.choice(len(E_all), n, replace=False)]
                M = int(np.clip(round(max(n / m0, 1)), 2, 32))
                M = max(M, 4)                     # at least four cells so shrinkage has a target
                for mode in ['none', 'pooled', 'js', 'eb', 'robust']:
                    r = fh.dp_prototypes(E, nm, M, rho * 0.98, C2=C2, T=3, rng=np.random.default_rng(rep * 7 + n),
                                         shrink=True, shrink_mode=mode, keep_truth=True, min_count=0.0)
                    if len(r['n']) == 0:
                        continue
                    err = ((r['mu'] - r['mu_true']) ** 2).mean(1)
                    mse = float(np.average(err, weights=np.maximum(r['n_true'], 1)))
                    res.setdefault((eps, n, mode), []).append(mse)
    out = [dict(eps=k[0], n=k[1], mode=k[2], mse=float(np.mean(v)), sd=float(np.std(v)), reps=len(v))
           for k, v in sorted(res.items())]
    json.dump(out, open(os.path.join(OUT, 'audit_shrink_%s.json' % name), 'w'), indent=1)
    print('shrink done', name)


def mrule_study(name='income', reps=10):
    E_all, nm, C2 = _class_records(name)
    sizes = [300, 500, 800, 1200, 2000, 3000, 5000, 8000, 12000, 20000]
    grid = [2, 4, 8, 16, 32, 64]
    res = {}
    for eps in [0.5, 1.0, 4.0]:
        rho = fh.rho_from_eps(eps, 1e-5) * 0.98
        m0 = m0_of(C2, rho / 0.98)
        for n in sizes:
            for rep in range(reps):
                rng0 = np.random.default_rng(1000 * rep + n)
                E = E_all[rng0.choice(len(E_all), n, replace=False)]
                ratio = max(n / m0, 1.0)
                rules = {'fixed (M=32)': 32, 'linear (ours)': int(np.clip(round(ratio), 2, 64)),
                         'dimension adaptive': int(np.clip(round(ratio ** (C2 / (C2 + 1.0))), 2, 64))}
                errs = {}
                for Mv in sorted(set(list(rules.values()) + grid)):
                    r = fh.dp_prototypes(E, nm, Mv, rho, C2=C2, T=3, rng=np.random.default_rng(rep * 31 + n + Mv),
                                         keep_truth=True)
                    errs[Mv] = release_error(E, r)
                for k, Mv in rules.items():
                    res.setdefault((eps, n, k), []).append((errs[Mv], Mv))
                best = min(grid, key=lambda g: errs[g] if np.isfinite(errs[g]) else 1e9)
                res.setdefault((eps, n, 'oracle'), []).append((errs[best], best))
    out = [dict(eps=k[0], n=k[1], rule=k[2], err=float(np.nanmean([x[0] for x in v])),
                M=float(np.mean([x[1] for x in v])), reps=len(v)) for k, v in sorted(res.items())]
    json.dump(out, open(os.path.join(OUT, 'audit_mrule_%s.json' % name), 'w'), indent=1)
    print('mrule done', name)


def _mi_cond(u, v, s):
    """Plug in I(U;V|S) with the Miller Madow correction, all arguments integer codes."""
    from collections import Counter
    n = len(u)
    def H(*cols):
        keys = np.stack(cols, 1)
        _, cnt = np.unique(keys, axis=0, return_counts=True)
        p = cnt / n
        return -(p * np.log(p)).sum() + (len(cnt) - 1) / (2 * n)
    return H(u, s) + H(v, s) - H(u, v, s) - H(s)


def cmi_study(name='income'):
    D = data.load(name)
    Z, y = D['Z'], D['y']
    names = [c['name'] for c in D['schema']]
    def code(j):
        c = D['schema'][j]
        if c['type'] == 'cat':
            return Z[:, j].astype(int)
        return np.digitize(Z[:, j], np.quantile(Z[:, j], [0.2, 0.4, 0.6, 0.8]))
    bidx = [names.index(b) for b in D['bridge']]
    S = np.zeros(len(Z), int)
    for j in bidx:
        S = S * 50 + code(j)
    _, S = np.unique(S, return_inverse=True)
    SY = S * 2 + y
    nonb = [j for j in range(len(names)) if j not in bidx]
    out = []
    for a in range(len(nonb)):
        for b in range(a + 1, len(nonb)):
            u, v = code(nonb[a]), code(nonb[b])
            i1 = _mi_cond(u, v, S); i2 = _mi_cond(u, v, SY)
            out.append(dict(u=names[nonb[a]], v=names[nonb[b]], I_S=float(i1), I_SY=float(i2)))
    json.dump(out, open(os.path.join(OUT, 'audit_cmi_%s.json' % name), 'w'), indent=1)
    print('cmi done', name, 'pairs', len(out), 'reduced', sum(o['I_SY'] < o['I_S'] for o in out))


def completion_rmse(silos, enc, protos, comp, all_cols):
    err, wsum = 0.0, 0.0
    for p, q in zip(protos, comp):
        s = silos[p['silo']]
        miss_cols = [c for c in all_cols if c not in p['cols']]
        if not miss_cols:
            continue
        obs, miss = enc.dims(p['cols']), enc.dims(miss_cols)
        Ef = enc.encode(s['Z'][s['y'] == p['y']])
        a = np.argmin(((Ef[:, None, obs] - p['mu'][None, :, obs]) ** 2).sum(-1), 1)
        for m in range(len(p['n'])):
            rows = a == m
            if rows.sum() < 20:
                continue
            err += rows.sum() * np.mean((q['mu'][m, miss] - Ef[rows][:, miss].mean(0)) ** 2); wsum += rows.sum()
    return float(np.sqrt(err / max(wsum, 1)))


def shift_study(name='income', seeds=range(5), strengths=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), eps_list=(1.0, INF)):
    D = data.load(name)
    out = []
    for eps in eps_list:
        for st in strengths:
            for sd in seeds:
                cfg = run.cfg_for(name, sel_strength=st, eps=eps)
                silos, enc = data.make_silos(D, K=cfg['K'], n_targets=cfg['n_targets'], p_feat=cfg['p_feat'],
                                             overlap=cfg['overlap'], n_lab=cfg['n_lab'], cap=cfg['cap'],
                                             sel_strength=st, seed=sd)
                F = Mth.Federation(silos, enc, cfg, np.random.default_rng(1000 + sd))
                protos = [p for p in F.release() if len(p['n'])]
                for lab, kw in [('unbalanced OT', dict(mode='softjoin', tau=1.0)),
                                ('balanced OT', dict(mode='softjoin', tau=INF)),
                                ('unbalanced OT, debiased cost', dict(mode='softjoin', tau=1.0, cost='debiased')),
                                ('nearest prototype', dict(mode='nearest')),
                                ('pooled mean fill', dict(mode='mean'))]:
                    dg = {}
                    comp = fh.complete_prototypes(protos, enc, F.all_cols, eta=cfg['eta'], diag=dg, **kw)
                    out.append(dict(eps=eps, strength=st, seed=sd, rule=lab,
                                    rmse=completion_rmse(silos, enc, protos, comp, F.all_cols),
                                    mass_kept=float(np.mean(dg['mass_kept'])) if dg else np.nan,
                                    cost=float(np.mean(dg['cost'])) if dg else np.nan,
                                    marg_viol=float(np.mean(dg['marg_viol'])) if dg else np.nan))
                print('shift', eps, st, sd, flush=True)
                json.dump(out, open(os.path.join(OUT, 'audit_shift_%s.json' % name), 'w'), indent=1)


def mia_study(name='taiwan', seeds=range(3), eps_list=(1.0, 4.0, INF), n_att=300):
    """Loss based membership inference on the deployed predictor. 5% of every donor is held out
    before any release; members are donor records, non members the held out records."""
    from sklearn.metrics import roc_auc_score
    D = data.load(name); out = []
    for sd in seeds:
        for eps in list(eps_list) + ['raw']:
            cfg = run.cfg_for(name, eps=(INF if eps == 'raw' else eps), ens=1)
            silos, enc = data.make_silos(D, K=cfg['K'], n_targets=cfg['n_targets'], p_feat=cfg['p_feat'],
                                         overlap=cfg['overlap'], n_lab=cfg['n_lab'], cap=cfg['cap'], sel_strength=cfg['sel_strength'], seed=sd)
            rng = np.random.default_rng(sd); hold = []
            for t in silos:
                if t['target']:
                    continue
                perm = rng.permutation(len(t['idx'])); h, kp = perm[:len(perm) // 20], perm[len(perm) // 20:]
                hold.append(t['idx'][h]); t['idx'] = t['idx'][kp]; t['Z'] = t['Z'][kp]; t['y'] = t['y'][kp]
                t['lab'] = np.arange(len(kp))
            hold = np.concatenate(hold)
            F = Mth.Federation(silos, enc, cfg, np.random.default_rng(1000 + sd))
            k = F.targets[0]; cols = silos[k]['cols']
            Xl, yl, _, _ = F.local(k)
            if eps == 'raw':      # non private reference: raw donor rows placed in the context
                pool = np.concatenate([silos[j]['idx'] for j in F.rich])
                ctx = rng.choice(pool, min(cfg['S'], len(pool)), replace=False)
                Xs = enc.to_model(D['Z'][ctx])[:, cols]; ys = D['y'][ctx]
                mem = rng.choice(ctx, n_att, replace=False)
            else:
                Xs, ys = Mth.build_context(F, k, rng=rng)
                mem = rng.choice(np.concatenate([silos[j]['idx'] for j in F.rich]), n_att, replace=False)
            non = rng.choice(hold, n_att, replace=False)
            q = np.r_[mem, non]
            Xq = enc.to_model(D['Z'][q])[:, cols]; yq = D['y'][q]
            P = Mth.tfm_predict(np.vstack([Xl, Xs]), np.r_[yl, ys], Xq, 1, sd)
            a = roc_auc_score(np.r_[np.ones(n_att), np.zeros(n_att)], P[np.arange(len(q)), yq])
            out.append(dict(seed=sd, eps=str(eps), mia_auc=float(a)))
            print('mia', name, sd, eps, round(a, 4), flush=True)
            json.dump(out, open(os.path.join(OUT, 'audit_mia_%s.json' % name), 'w'), indent=1)


def tau_study(name='income', seeds=range(5), strengths=(0.0, 3.0), overlaps=(0.5, 0.1),
              taus=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, INF)):
    """Sensitivity of SoftJoin to the marginal penalty tau: absolute completion RMSE, transported
    mass and marginal violation, at two selection strengths and two overlap fractions (eps = 1)."""
    D = data.load(name); out = []
    for ov in overlaps:
        for st in strengths:
            for sd in seeds:
                cfg = run.cfg_for(name, sel_strength=st, overlap=ov)
                silos, enc = data.make_silos(D, K=cfg['K'], n_targets=cfg['n_targets'], p_feat=cfg['p_feat'],
                                             overlap=ov, n_lab=cfg['n_lab'], cap=cfg['cap'], sel_strength=st, seed=sd)
                F = Mth.Federation(silos, enc, cfg, np.random.default_rng(1000 + sd))
                protos = [p for p in F.release() if len(p['n'])]
                for t in list(taus) + ['mean']:
                    dg = {}
                    kw = dict(mode='mean') if t == 'mean' else dict(mode='softjoin', tau=t)
                    comp = fh.complete_prototypes(protos, enc, F.all_cols, eta=cfg['eta'], diag=dg, **kw)
                    out.append(dict(overlap=ov, strength=st, seed=sd, tau=str(t),
                                    rmse=completion_rmse(silos, enc, protos, comp, F.all_cols),
                                    mass_kept=float(np.mean(dg['mass_kept'])) if dg else np.nan,
                                    marg_viol=float(np.mean(dg['marg_viol'])) if dg else np.nan))
                print('tau', ov, st, sd, flush=True)
                json.dump(out, open(os.path.join(OUT, 'audit_tau_%s.json' % name), 'w'), indent=1)


if __name__ == '__main__':
    what = sys.argv[1]
    names = sys.argv[2:] or ['income']
    for nm_ in names:
        dict(shrink=shrink_study, mrule=mrule_study, cmi=cmi_study, shift=shift_study, mia=mia_study, tau=tau_study)[what](nm_)
