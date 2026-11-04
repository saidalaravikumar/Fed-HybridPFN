"""Audits: (1) completion error of SoftJoin against hidden ground truth;
(2) black box membership inference against the deployed in-context predictor."""
import sys, json
import numpy as np
from sklearn.metrics import roc_auc_score
import data, run, methods as M, fedhybrid as fh

INF = float('inf')


def completion_error(name, seed, eps, out):
    D = data.load(name)
    cfg = run.cfg_for(name, eps=eps)
    silos, enc = data.make_silos(D, K=cfg['K'], n_targets=cfg['n_targets'], p_feat=cfg['p_feat'],
                                 overlap=cfg['overlap'], n_lab=cfg['n_lab'], cap=cfg['cap'], seed=seed)
    F = M.Federation(silos, enc, cfg, np.random.default_rng(1000 + seed))
    protos = F.release()
    res = {}
    for mode, tau in [('SoftJoin (unbalanced)', 1.0), ('SoftJoin (balanced)', INF), ('Pooled mean fill', None)]:
        comp = fh.complete_prototypes(protos, enc, F.all_cols, eta=cfg['eta'], tau=tau if tau else 1.0,
                                      mode='softjoin' if tau else 'mean')
        err, wsum, cost = 0.0, 0.0, []
        for p, q in zip(protos, comp):
            s = silos[p['silo']]
            miss_cols = [c for c in F.all_cols if c not in p['cols']]
            if not miss_cols:
                continue
            obs, miss = enc.dims(p['cols']), enc.dims(miss_cols)
            Efull = enc.encode(s['Z'][s['y'] == p['y']])           # hidden truth (simulation only)
            a = np.argmin(((Efull[:, None, obs] - p['mu'][None, :, obs]) ** 2).sum(-1), 1)
            for m in range(len(p['n'])):
                rows = a == m
                if rows.sum() < 20:
                    continue
                truth = Efull[rows][:, miss].mean(0)
                err += rows.sum() * np.mean((q['mu'][m, miss] - truth) ** 2); wsum += rows.sum()
        res[mode] = float(np.sqrt(err / wsum))
    rec = dict(data=name, seed=seed, eps=eps, rmse=res)
    print(json.dumps(rec), flush=True)
    with open(out, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def mia(name, seed, eps_list, out, n_att=400):
    D = data.load(name)
    for eps in eps_list:
        cfg = run.cfg_for(name, eps=eps, ens=1)
        silos, enc = data.make_silos(D, K=cfg['K'], n_targets=cfg['n_targets'], p_feat=cfg['p_feat'],
                                     overlap=cfg['overlap'], n_lab=cfg['n_lab'], cap=cfg['cap'], seed=seed)
        rng = np.random.default_rng(seed)
        hold = []
        for t in silos:            # 5% of every donor silo is held out before any release
            if t['target']:
                continue
            perm = rng.permutation(len(t['idx'])); h = perm[:len(perm) // 20]; kp = perm[len(perm) // 20:]
            hold.append(t['idx'][h])
            t['idx'] = t['idx'][kp]; t['Z'] = t['Z'][kp]; t['y'] = t['y'][kp]; t['lab'] = np.arange(len(kp))
        hold = np.concatenate(hold)
        F = M.Federation(silos, enc, cfg, np.random.default_rng(1000 + seed))
        k = F.targets[0]; s = silos[k]
        cols = s['cols']
        Xl, yl, _, _ = F.local(k)
        donors = np.concatenate([silos[j]['idx'] for j in F.rich])
        mem = rng.choice(donors, n_att, replace=False)
        non = rng.choice(hold, n_att, replace=False)
        Xq = enc.to_model(D['Z'][np.r_[mem, non]])[:, cols]; yq = D['y'][np.r_[mem, non]]
        lab = np.r_[np.ones(n_att), np.zeros(n_att)]
        # deployed Fed-HybridPFN predictor (context = local labels + DP synthetic rows)
        Xs, ys = M.build_context(F, k, rng=rng)
        P = M.tfm_predict(np.vstack([Xl, Xs]), np.r_[yl, ys], Xq, 1, seed)
        auc_ours = roc_auc_score(lab, P[np.arange(len(yq)), yq])
        rec = dict(data=name, seed=seed, eps=eps, method='Fed-HybridPFN', mia_auc=float(auc_ours))
        print(json.dumps(rec), flush=True)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    # non private pooled context: members are rows placed in the context
    pool = np.concatenate([silos[j]['idx'] for j in F.rich])
    ctx = rng.choice(pool, cfg['S'], replace=False)
    Xc = enc.to_model(D['Z'][ctx]).copy()
    mem = rng.choice(ctx, n_att, replace=False); non = rng.choice(hold, n_att, replace=False)
    Xq = enc.to_model(D['Z'][np.r_[mem, non]])[:, cols]; yq = D['y'][np.r_[mem, non]]
    P = M.tfm_predict(np.vstack([Xl, Xc[:, cols]]), np.r_[yl, D['y'][ctx]], Xq, 1, seed)
    auc_c = roc_auc_score(np.r_[np.ones(n_att), np.zeros(n_att)], P[np.arange(len(yq)), yq])
    rec = dict(data=name, seed=seed, eps=INF, method='Central-TFM', mia_auc=float(auc_c))
    print(json.dumps(rec), flush=True)
    with open(out, 'a') as f:
        f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'completion':
        for name in sys.argv[2:]:
            for sd in range(3):
                for e in [1.0, INF]:
                    completion_error(name, sd, e, '../results/completion.jsonl')
    else:
        for sd in range(3):
            mia(sys.argv[2], sd, [0.5, 1.0, 4.0, INF], '../results/mia.jsonl')
