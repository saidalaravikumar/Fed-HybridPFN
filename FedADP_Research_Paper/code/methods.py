"""Methods compared in the paper. Every method returns, for each target silo, predicted
probabilities on that silo's held out test rows."""
import time
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
import fedhybrid as fh

torch.set_num_threads(1)
_TFM = {}


TFM_CKPT = None     # None: TabICLv2 (default); else a TabICL checkpoint name


def tfm_predict(Xc, yc, Xq, n_est=1, seed=0):
    from tabicl import TabICLClassifier
    if len(np.unique(yc)) < 2:
        p = np.full((len(Xq), 2), 1e-3); p[:, int(yc[0])] = 1 - 1e-3; return p
    kw = {} if TFM_CKPT is None else dict(checkpoint_version=TFM_CKPT)
    m = TabICLClassifier(n_estimators=n_est, device='cpu', random_state=seed, n_jobs=1, **kw)
    m.fit(Xc, yc)
    return m.predict_proba(Xq)


def gbdt_predict(Xc, yc, Xq, cat_mask, seed=0):
    if len(np.unique(yc)) < 2:
        p = np.full((len(Xq), 2), 1e-3); p[:, int(yc[0])] = 1 - 1e-3; return p
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed,
                                       categorical_features=cat_mask if cat_mask.any() else None,
                                       early_stopping=False, min_samples_leaf=max(2, min(20, len(yc) // 10)))
    m.fit(Xc, yc)
    return m.predict_proba(Xq)


# ---------------------------------------------------------------------------
class Federation:
    """Holds the silos and the (cached) DP releases of the label rich silos."""

    def __init__(self, silos, enc, cfg, rng):
        self.S, self.enc, self.cfg, self.rng = silos, enc, cfg, rng
        self.all_cols = list(range(len(enc.schema)))
        self.num_dims = np.zeros(enc.dim, bool)
        for j, c in enumerate(enc.schema):
            if c['type'] == 'num':
                self.num_dims[enc.slices[j]] = True
        self.rich = [k for k, s in enumerate(silos) if not s['target']]
        self.targets = [k for k, s in enumerate(silos) if s['target']]
        self._protos = {}
        self.comm_bytes = 0
        self.gate_log = []
        self.rel_log = []
        self.var_log = []
        self.ctx_log = {}
        self.server_t = 0.0
        self.proto_mse = float('nan')
        self.n_ot = 0

    def release(self, eps=None, M=None, T=None, two_round=None):
        """DP class conditional prototypes of every label rich silo (cached per setting).

        two_round=True (default): round 1 spends a fraction beta of each silo's budget on a
        short DP Lloyd run; the server pools the round 1 prototypes into M/2 global centres
        per class (post processing); round 2 assigns records to those centres and releases
        counts, sums and sums of squares once with the remaining (1-beta) budget. Total
        privacy per silo is rho-zCDP by sequential composition of the two rounds."""
        eps = self.cfg['eps'] if eps is None else eps
        M = self.cfg['M'] if M is None else M
        T = self.cfg['T'] if T is None else T
        two_round = self.cfg['two_round'] if two_round is None else two_round
        key = (eps, M, T, two_round, self.cfg['adaptM'], self.cfg['shrink'], self.cfg['gamma'], self.cfg['equal_split'],
               self.cfg['shrink_mode'], self.cfg['mrule'], self.cfg['small_T'], self.cfg['split'])
        if key in self._protos:
            return self._protos[key]
        rho = fh.rho_from_eps(eps, self.cfg['delta'])
        beta = self.cfg['beta']
        self._mse = []
        My = max(2, M // 2)
        loc = {}
        for k in self.rich:
            s = self.S[k]
            dims = self.enc.dims(s['cols'])
            E = self.enc.encode(s['Z'], s['cols'])[:, dims]
            loc[k] = (dims, [E[s['y'] == y] for y in (0, 1)])

        adapt = self.cfg['adaptM']

        def one(k, y, rho_k, T_k, init=None):
            dims, Es = loc[k]
            if np.isfinite(eps):
                Mk = My
                if adapt and init is None:
                    # noise aware resolution: 2% of the budget buys a noisy class count; aim for
                    # m0 records per prototype so that the mean noise stays near gamma
                    r0 = 0.02 * rho_k; rho_k = rho_k - r0
                    n_t = len(Es[y]) + self.rng.normal(0, fh.gauss_sigma(1.0, r0))
                    s_sum = fh.gauss_sigma(np.sqrt(len(self.S[k]['cols'])), 0.70 * 0.6 * rho_k)
                    m0 = s_sum / self.cfg['gamma']
                    ratio = max(n_t / m0, 1.0)
                    if self.cfg['mrule'] == 'dim':      # Corollary 1 scaling with d' = |A_k|
                        dprime = len(self.S[k]['cols'])
                        ratio = ratio ** (dprime / (dprime + 1.0))
                    Mk = int(np.clip(np.round(ratio), 2, My))
                    if self.cfg['small_T'] and Mk <= 2:
                        T_k = 0     # two cells: Lloyd steps cannot move much, so the whole budget funds the final release
                r = fh.dp_prototypes(Es[y], self.num_dims[dims], Mk, rho_k, C2=len(self.S[k]['cols']),
                                     T=T_k, rng=self.rng, init=init, shrink=self.cfg['shrink'], shrink_mode=self.cfg['shrink_mode'],
                                     split=(0.75, 1/3, 1/3, 1/3) if self.cfg['equal_split'] else tuple(self.cfg['split']),
                                     keep_truth=True)
                if len(r['n']):   # simulation audit only: squared error of released means against true cell means
                    self._mse.append((float(np.sum(((r['mu'] - r['mu_true']) ** 2).sum(1) * r['n_true'])), float(r['n_true'].sum())))
                return r
            return fh.exact_prototypes(Es[y], self.num_dims[dims], My, rng=self.rng)

        protos, nbytes = [], 0
        if two_round and np.isfinite(eps):
            from sklearn.cluster import KMeans
            glob = {}
            for y in (0, 1):
                mus, ns = [], []
                for k in self.rich:
                    r = one(k, y, beta * rho, self.cfg['T1'])
                    mu = np.full((len(r['n']), self.enc.dim), np.nan); mu[:, loc[k][0]] = r['mu']
                    mus.append(mu); ns.append(r['n'])
                    nbytes += 4 * len(r['n']) * (1 + len(loc[k][0]))
                mus, ns = np.vstack(mus), np.concatenate(ns)
                fill = np.nansum(mus * ns[:, None], 0) / np.maximum((~np.isnan(mus) * ns[:, None]).sum(0), 1e-9)
                mus = np.where(np.isnan(mus), fill[None], mus)
                km = KMeans(min(My, len(ns)), n_init=4, random_state=int(self.rng.integers(1 << 30)))
                km.fit(mus, sample_weight=np.maximum(ns, 1e-6))
                glob[y] = km.cluster_centers_
            for k in self.rich:
                dims = loc[k][0]
                for y in (0, 1):
                    r = one(k, y, (1 - beta) * rho, 0, init=glob[y][:, dims])
                    mu = np.full((len(r['n']), self.enc.dim), np.nan); var = mu.copy(); pv = mu.copy()
                    mu[:, dims] = r['mu']; var[:, dims] = r['var']; pv[:, dims] = r['pv']
                    if len(r['n']):   # every cell may be filtered out as too noisy
                        protos.append(dict(silo=k, y=y, cols=self.S[k]['cols'], n=r['n'], mu=mu, var=var, pv=pv,
                                           lam=r['lam'], nv=r['nv'],
                                           snr=(r['var'].sum(1) / (r['var'].sum(1) + len(dims) * r['nv']))))
                    nbytes += 4 * len(r['n']) * (1 + 2 * len(dims))
        else:
            for k in self.rich:
                dims = loc[k][0]
                for y in (0, 1):
                    r = one(k, y, rho, T)
                    mu = np.full((len(r['n']), self.enc.dim), np.nan); var = mu.copy(); pv = mu.copy()
                    mu[:, dims] = r['mu']; var[:, dims] = r['var']; pv[:, dims] = r['pv']
                    if len(r['n']):   # every cell may be filtered out as too noisy
                        protos.append(dict(silo=k, y=y, cols=self.S[k]['cols'], n=r['n'], mu=mu, var=var, pv=pv,
                                           lam=r['lam'], nv=r['nv'],
                                           snr=(r['var'].sum(1) / (r['var'].sum(1) + len(dims) * r['nv']))))
                    nbytes += 4 * len(r['n']) * (1 + 2 * len(dims))
        self.comm_bytes = nbytes
        self.proto_mse = (sum(a for a, _ in self._mse) / max(sum(b for _, b in self._mse), 1.0)) if self._mse else float('nan')
        self._protos[key] = protos
        return protos

    def release_codebook(self, eps=None, M=None):
        """Codebook pooled release. Round 1 (fraction beta of rho): every label rich silo runs
        a short DP Lloyd per class. The server clusters the pooled round 1 prototypes into a
        shared codebook (post processing). Round 2 (1 - beta): every silo assigns its records to
        the codebook on its own columns and releases noisy counts, sums and numeric square
        sums per cell, without filtering or shrinkage. The server adds the releases of all silos
        cell by cell and dimension by dimension (post processing), so signal grows with the
        number of silos while independent noise grows with its square root."""
        from sklearn.cluster import KMeans
        eps = self.cfg['eps'] if eps is None else eps
        key = ('codebook', eps, self.cfg['Mglob'], self.cfg['beta'])
        if key in self._protos:
            return self._protos[key]
        if not np.isfinite(eps):
            raise ValueError('codebook mode needs a finite budget')
        rho = fh.rho_from_eps(eps, self.cfg['delta']); beta = self.cfg['beta']
        Mg = self.cfg['Mglob']
        d = self.enc.dim
        out, nbytes = [], 0
        for y in (0, 1):
            mus, ns = [], []
            loc = {}
            for k in self.rich:
                s = self.S[k]; dims = self.enc.dims(s['cols'])
                E = self.enc.encode(s['Z'][s['y'] == y], s['cols'])[:, dims]
                loc[k] = (dims, E)
                r = fh.dp_prototypes(E, self.num_dims[dims], max(2, Mg // 2), beta * rho, C2=len(s['cols']),
                                     T=self.cfg['T1'], rng=self.rng, shrink=False)
                if len(r['n']):
                    mu = np.full((len(r['n']), d), np.nan); mu[:, dims] = r['mu']
                    mus.append(mu); ns.append(r['n'])
                    nbytes += 4 * len(r['n']) * (1 + len(dims))
            mus, ns = np.vstack(mus), np.concatenate(ns)
            fill = np.nansum(mus * ns[:, None], 0) / np.maximum((~np.isnan(mus) * ns[:, None]).sum(0), 1e-9)
            mus = np.where(np.isnan(mus), fill[None], mus)
            km = KMeans(min(Mg, len(ns)), n_init=4, random_state=int(self.rng.integers(1 << 30)))
            cen = km.fit(mus, sample_weight=np.maximum(ns, 1e-6)).cluster_centers_
            G = len(cen)
            CNT = np.zeros(G); S1 = np.zeros((G, d)); S2 = np.zeros((G, d)); W = np.zeros((G, d))
            NV = np.zeros((G, d))
            r2 = (1 - beta) * rho
            for k in self.rich:
                dims, E = loc[k]
                C2 = len(self.S[k]['cols']); nn = int(self.num_dims[dims].sum())
                s_c = fh.gauss_sigma(1.0, 0.15 * r2); s_s = fh.gauss_sigma(np.sqrt(C2), 0.70 * r2)
                s_q = fh.gauss_sigma(np.sqrt(max(nn, 1)), 0.15 * r2)
                c = cen[:, dims]
                a = np.argmin((E ** 2).sum(1)[:, None] - 2 * E @ c.T + (c ** 2).sum(1)[None], 1) if len(E) else np.zeros(0, int)
                cnt = np.bincount(a, minlength=G).astype(float)
                sm = np.zeros((G, len(dims))); np.add.at(sm, a, E)
                sq = np.zeros((G, len(dims))); np.add.at(sq, a, E ** 2)
                cnt_n = cnt + self.rng.normal(0, s_c, G)
                sm_n = sm + self.rng.normal(0, s_s, sm.shape)
                sq_n = sq + self.rng.normal(0, s_q, sq.shape)
                CNT += cnt_n
                S1[:, dims] += sm_n; S2[:, dims] += sq_n
                W[:, dims] += cnt_n[:, None]; NV[:, dims] += s_s ** 2
                nbytes += 4 * G * (1 + 2 * len(dims))
            ok = (W > 3 * np.sqrt(NV / np.maximum(1, 1)) * 0 + 5.0)
            mu = np.where(ok, S1 / np.maximum(W, 1.0), np.nan)
            var = np.where(ok, S2 / np.maximum(W, 1.0) - mu ** 2, np.nan)
            nvm = NV / np.maximum(W, 1.0) ** 2
            keep = CNT > 3 * fh.gauss_sigma(1.0, 0.15 * r2) * np.sqrt(len(self.rich))
            mu, var, nvm, CNT = mu[keep], var[keep], nvm[keep], CNT[keep]
            # dims observed by nobody in a cell: class pooled mean of the observed cells
            colmean = np.nanmean(mu, 0)
            mu = np.where(np.isnan(mu), colmean[None], mu)
            # empirical Bayes shrinkage per dimension across cells
            pool = (CNT[:, None] * mu).sum(0) / CNT.sum()
            tau2 = np.maximum(np.average((mu - pool) ** 2, axis=0, weights=CNT) - np.average(np.nan_to_num(nvm), axis=0, weights=CNT), 1e-4)
            lam = tau2[None] / (tau2[None] + np.nan_to_num(nvm))
            mu = np.clip(pool + lam * (mu - pool), 0, 1)
            var = np.where(self.num_dims[None], np.clip(np.nan_to_num(var, nan=0.01), 1e-3, 0.25),
                           np.clip(mu * (1 - mu), 1e-4, 0.25))
            out.append(dict(silo=-1, y=y, cols=self.all_cols, n=CNT, mu=mu, var=var, pv=lam * np.nan_to_num(nvm),
                            lam=lam.mean(1), nv=np.nan_to_num(nvm).mean(1),
                            snr=var.sum(1) / (var.sum(1) + np.nan_to_num(nvm).sum(1))))
        self.comm_bytes = nbytes
        self._protos[key] = out
        return out

    def local(self, k):
        s = self.S[k]
        Xm = self.enc.to_model(s['Z'])[:, s['cols']]
        return Xm[s['lab']], s['y'][s['lab']], Xm[s['test']], s['y'][s['test']]

    def cat_mask(self, cols):
        return np.array([self.enc.schema[j]['type'] == 'cat' for j in cols])


# ---------------------------------------------------------------------------
def run_local_tfm(F, k, seed):
    Xl, yl, Xq, _ = F.local(k)
    return tfm_predict(Xl, yl, Xq, F.cfg['n_est'], seed)


def run_local_gbdt(F, k, seed):
    Xl, yl, Xq, _ = F.local(k)
    return gbdt_predict(Xl, yl, Xq, F.cat_mask(F.S[k]['cols']), seed)


def build_context(F, k, softjoin=True, align=True, tau=None, eps=None, M=None, S=None, eta=None, rng=None, lam=None, bw=None,
                  two_round=None, perturb=False, cost=None, cmode=None, gate_arrg=False, srule=None):
    cfg = F.cfg
    cost = cfg['cost'] if cost is None else cost
    cmode = ('softjoin' if softjoin else 'mean') if cmode is None else cmode
    if cmode == 'codebook' or (cmode == 'softjoin' and cfg['codebook']):
        cmode = 'codebook'
        protos = F.release_codebook(eps=eps)
    else:
        protos = F.release(eps=eps, M=M, two_round=two_round)
    key = ('comp', eps, M, cmode, tau, eta, two_round, F.cfg['adaptM'], F.cfg['shrink'], F.cfg['gamma'], F.cfg['equal_split'],
           cost, F.cfg['shrink_mode'], F.cfg['mrule'], F.cfg['small_T'], F.cfg['split'], F.cfg.get('topk'))
    if key not in F._protos:
        t0 = time.time(); n0 = fh.N_SINKHORN
        F._protos[key] = protos if cmode == 'codebook' else fh.complete_prototypes(
            protos, F.enc, F.all_cols, eta=cfg['eta'] if eta is None else eta,
            tau=cfg['tau'] if tau is None else tau, mode=cmode, cost=cost, topk=cfg.get('topk'))
        F.server_t = time.time() - t0; F.n_ot = fh.N_SINKHORN - n0
    comp = F._protos[key]
    s = F.S[k]
    if gate_arrg:
        gk = ('arrg', k, F.cfg['rel_mode'], F.cfg['compat']) + key
        if gk not in F._protos:
            F._protos[gk] = arrg(F, k, comp, cfg['S'] if S is None else S)
            F.gate_log.append(F._protos[gk][0])
        S_t, comp = F._protos[gk]
        S = S_t
        if S == 0:
            return np.zeros((0, len(s['cols']))), np.zeros(0, int)
    if perturb:   # one draw of the prototype means from their posterior given the DP release
        comp = [dict(p, mu=np.clip(p['mu'] + np.sqrt(np.nan_to_num(p['pv'])) * rng.normal(size=p['mu'].shape), 0, 1))
                for p in comp]
    S = cfg['S'] if S is None else S
    if (cfg['S_rule'] if srule is None else srule) == 'resolution':
        # resolution matched context: the release carries information about at most as many
        # distinct cells as it has prototypes, so draw at most kappa rows per released prototype
        n_proto = sum(len(p['n']) for p in comp)
        S = int(min(S, cfg['kappa'] * n_proto))
        F.ctx_log.setdefault(k, S)
        if S == 0:
            return np.zeros((0, len(s['cols']))), np.zeros(0, int)
    amode = cfg['align_mode']
    if align and amode == 'kernel':
        U = F.enc.encode(s['Z'][s['unl']], s['cols'])
        ctx = fh.context_weights(comp, F.enc, s['cols'], U=U, align=True,
                                lam=cfg['lam'] if lam is None else lam, bw=cfg['bw'] if bw is None else bw)
        return fh.synthesize(ctx, F.enc, s['cols'], S, rng)
    ctx = fh.context_weights(comp, F.enc, s['cols'], U=None, align=False)
    if align:   # density ratio resampling from a larger synthetic pool
        Xp, yp = fh.synthesize(ctx, F.enc, s['cols'], cfg['pool_mult'] * S, rng)
        pick = rng.choice(len(s['unl']), size=min(len(s['unl']), cfg['n_dr']), replace=False)
        Xu = F.enc.to_model(s['Z'][s['unl'][pick]])[:, s['cols']]
        Xs, ys, _ = fh.density_ratio_resample(Xp, yp, Xu, S, rng, learner=amode)
        return Xs, ys
    return fh.synthesize(ctx, F.enc, s['cols'], S, rng)


def arrg(F, k, comp, S_max):
    """Analytical release reliability gate (no labels used). For prototype m with released
    mass p_m and mean shrinkage factor lam_m, the target silo computes the share c_m of the
    prototype's mass that its own unlabelled population supports, c_m = min(1, q_m / p_m), where
    q_m is the target's soft assignment mass on its own columns. The reliability score is
    R_t = sum_m p_m lam_m c_m (reliability weighted population overlap); the context keeps
    S_t = floor(S_max R_t) synthetic rows drawn with weights proportional to p_m lam_m c_m."""
    s = F.S[k]
    cols = s['cols']; dims = F.enc.dims(cols)
    mu = np.concatenate([p['mu'][:, dims] for p in comp])
    n = np.concatenate([p['n'] for p in comp]); pm = n / n.sum()
    key_rel = 'snr' if F.cfg['rel_mode'] == 'snr' else 'lam'
    lam = np.concatenate([p.get(key_rel, np.ones(len(p['n']))) for p in comp])
    rng = np.random.default_rng(17 + k)
    U = F.enc.encode(s['Z'][s['unl'][rng.choice(len(s['unl']), min(3000, len(s['unl'])), replace=False)]], cols)[:, dims]
    d2 = np.maximum((U ** 2).sum(1)[:, None] - 2 * U @ mu.T + (mu ** 2).sum(1)[None], 0)
    h2 = max(np.median(d2.min(1)), 1e-3)
    ll = np.log(pm + 1e-300)[None] - d2 / (2 * h2)
    from scipy.special import logsumexp as lse
    q = np.exp(ll - lse(ll, 1, keepdims=True)).mean(0)
    c = np.minimum(1.0, q / np.maximum(pm, 1e-12)) if F.cfg['compat'] else np.ones_like(pm)
    a = lam * c
    R = float((pm * a).sum())
    F.rel_log.append(R)
    w = pm * a
    off = 0; out = []
    for p in comp:
        L = len(p['n'])
        out.append(dict(p, n=np.maximum(w[off:off + L], 1e-12) * n.sum()))
        off += L
    return int(np.floor(S_max * R)), out


def temperature_scale(F, k, seed, P, Xl, yl, kw, softjoin, align, folds=4, grid=np.linspace(0.6, 2.5, 39)):
    """Calibrate the hedged predictor with one temperature fitted on the target's own labels by
    cross fitting: each fold of labelled rows is predicted from a context without it (synthetic rows
    plus the other folds, hedged like the deployed predictor). A temperature leaves AUROC unchanged;
    it uses only target data and DP outputs, so the privacy guarantee is unaffected."""
    rng = np.random.default_rng(10007 * seed + k)          # separate stream: the prediction P is untouched
    n = len(yl); idx = rng.permutation(n); oof = np.zeros(n)
    for f in range(folds):
        va = idx[f::folds]; tr = np.setdiff1d(idx, va)
        if len(np.unique(yl[tr])) < 2:
            return P
        Xs, ys = build_context(F, k, softjoin=softjoin, align=align, rng=rng, **kw)
        pf = tfm_predict(np.vstack([Xl[tr], Xs]), np.r_[yl[tr], ys], Xl[va], F.cfg['n_est'], seed)[:, 1]
        if F.cfg['mix']:
            w = len(tr) / (len(tr) + len(Xs))
            pf = (1 - w) * pf + w * tfm_predict(Xl[tr], yl[tr], Xl[va], F.cfg['n_est'], seed)[:, 1]
        oof[va] = pf
    z = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / np.clip(1 - oof, 1e-6, 1))
    nll = [np.mean(np.logaddexp(0, -(2 * yl - 1) * z / t)) for t in grid]
    T = float(grid[int(np.argmin(nll))])
    F.gate_log.append(T)
    zq = np.log(np.clip(P[:, 1], 1e-6, 1 - 1e-6) / np.clip(P[:, 0], 1e-6, 1))
    p1 = 1 / (1 + np.exp(-zq / T))
    return np.c_[1 - p1, p1]


def run_ours(F, k, seed, softjoin=True, align=None, use_local=True, learner='tfm', em=False,
             distill=None, n_distill=None, keep_synth=None, label_mode=None, ens=None, perturb=None, gate=None, arrg_gate=None, **kw):
    rng = np.random.default_rng(seed * 7919 + k)
    distill = F.cfg['distill'] if distill is None else distill
    align = bool(F.cfg['align']) if align is None else align
    Xl, yl, Xq, _ = F.local(k)
    E = F.cfg['ens'] if ens is None else ens
    gate = F.cfg['gate'] if gate is None else gate
    arrg_gate = bool(F.cfg['arrg']) if arrg_gate is None else arrg_gate
    if arrg_gate:
        kw = dict(kw, gate_arrg=True)
    s_keep = None
    if gate and learner == 'tfm' and use_local:
        # local validation gate: the target silo picks how many synthetic rows to admit by
        # stratified K fold log loss on its own labelled rows (no data leaves the silo)
        Xg, yg = build_context(F, k, softjoin=softjoin, align=align, rng=rng, **dict(kw, srule='fixed'))
        s_keep, _ = choose_context_size(Xl, yl, Xg, yg, F.cfg, seed, rng)
        kw = dict(kw, srule='fixed')
        F.gate_log.append(s_keep)
        if s_keep == 0:
            return tfm_predict(Xl, yl, Xq, F.cfg['n_est'], seed)
    if E > 1 and learner == 'tfm' and not distill:
        Ps = []
        for e in range(E):
            Xs, ys = build_context(F, k, softjoin=softjoin, align=align, rng=rng,
                                   perturb=bool(F.cfg['perturb']) if perturb is None else perturb, **kw)
            if s_keep is not None:
                Xs, ys = Xs[:s_keep], ys[:s_keep]
            if len(Xs) == 0 and use_local:
                P = tfm_predict(Xl, yl, Xq, F.cfg['n_est'], seed); F.var_log.append(0.0)
                return P
            Xc, yc = (np.vstack([Xl, Xs]), np.concatenate([yl, ys])) if use_local else (Xs, ys)
            Ps.append(tfm_predict(Xc, yc, Xq, F.cfg['n_est'], seed + 101 * e))
        F.var_log.append(float(np.var(np.array(Ps)[:, :, 1], 0).mean()))
        P = np.mean(Ps, 0)
        if F.cfg['mix']:
            # resolution weighted hedge: the local predictor gets the share of the context that
            # the target's own labels represent, w = n_lab / (n_lab + S_t); no labels are used
            w = len(yl) / (len(yl) + len(Xs))
            P = (1 - w) * P + w * tfm_predict(Xl, yl, Xq, F.cfg['n_est'], seed)
        if F.cfg.get('tscale'):
            P = temperature_scale(F, k, seed, P, Xl, yl, kw, softjoin, align)
        if F.cfg['em']:
            # label shift correction (Saerens EM) on the unlabelled query inputs
            prior = np.bincount(np.concatenate([yl, ys]), minlength=2) / (len(yl) + len(ys))
            P = fh.em_label_shift(P, prior)
        return P
    Xs, ys = build_context(F, k, softjoin=softjoin, align=align, rng=rng, **kw)
    if len(Xs) == 0 and use_local and learner == 'tfm':
        return tfm_predict(Xl, yl, Xq, F.cfg['n_est'], seed)
    if use_local:
        Xc, yc = np.vstack([Xl, Xs]), np.concatenate([yl, ys])
    else:
        Xc, yc = Xs, ys
    if learner == 'tfm' and distill:
        # prior aligned context distillation: the silo labels a sample of its own unlabelled
        # rows with the federated posterior and uses them (real local covariates) as context
        s = F.S[k]
        nd = F.cfg['n_distill'] if n_distill is None else n_distill
        ks = F.cfg['keep_synth'] if keep_synth is None else keep_synth
        lm = F.cfg['label_mode'] if label_mode is None else label_mode
        pick = rng.choice(len(s['unl']), size=min(nd, len(s['unl'])), replace=False)
        Xu = F.enc.to_model(s['Z'][s['unl'][pick]])[:, s['cols']]
        Pu = tfm_predict(Xc, yc, np.vstack([Xu, Xq]), F.cfg['n_est'], seed)
        Pu, P0 = Pu[:len(Xu)], Pu[len(Xu):]
        if lm == 'sample':
            yu = (rng.random(len(Xu)) < Pu[:, 1]).astype(int)
        else:
            yu = Pu.argmax(1)
        sel = rng.choice(len(Xs), size=min(ks, len(Xs)), replace=False)
        Xc = np.vstack([Xl, Xu, Xs[sel]]); yc = np.concatenate([yl, yu, ys[sel]])
    if learner == 'tfm':
        P = tfm_predict(Xc, yc, Xq, F.cfg['n_est'], seed)
    else:
        P = gbdt_predict(Xc, yc, Xq, F.cat_mask(F.S[k]['cols']), seed)
    if em:
        prior = np.bincount(yc, minlength=2) / len(yc)
        P = fh.em_label_shift(P, prior)
    return P


def choose_context_size(Xl, yl, Xs, ys, cfg, seed, rng):
    """Local validation gate run inside the target silo: pick the number of synthetic rows
    (fractions cfg['gate_grid'] of S) that minimises stratified K fold log loss of the in
    context predictor on the silo's own labelled rows; ties go to the larger context."""
    from sklearn.model_selection import StratifiedKFold
    grid = sorted(set(int(round(f * len(Xs))) for f in cfg['gate_grid']))
    kf = StratifiedKFold(n_splits=cfg['gate_folds'], shuffle=True, random_state=seed)
    loss = {}
    for s_n in grid:
        tot = 0.0
        for tr, va in kf.split(Xl, yl):
            Xc = np.vstack([Xl[tr], Xs[:s_n]]); yc = np.concatenate([yl[tr], ys[:s_n]])
            P = tfm_predict(Xc, yc, Xl[va], 1, seed)
            tot += -np.log(np.clip(P[np.arange(len(va)), yl[va]], 1e-4, 1)).sum()
        loss[s_n] = tot / len(yl)
    best = min(grid, key=lambda g: (loss[g], -g))
    return best, loss


def run_oracle_gate(F, k, seed):
    """Upper bound reference only: picks the synthetic context size with the TEST labels."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed * 7919 + k)
    Xl, yl, Xq, yq = F.local(k)
    Xs, ys = build_context(F, k, rng=rng, srule='fixed')
    best, bestP = -1, None
    for s_n in [0, 500, 1000, 1500]:
        P = tfm_predict(np.vstack([Xl, Xs[:s_n]]), np.r_[yl, ys[:s_n]], Xq, 1, seed)
        a = roc_auc_score(yq, P[:, 1])
        if a > best:
            best, bestP = a, P
    return bestP


def run_dp_proto(F, k, seed):
    """Pooled DP summary: same release, missing columns filled by pooled DP means, a single
    context draw (no SoftJoin, no release uncertainty marginalisation)."""
    return run_ours(F, k, seed, softjoin=False, align=False, ens=1, gate=False, srule='fixed')


def run_central(F, k, seed):
    """Non private reference: raw labelled rows of all label rich silos pooled, NaN for
    columns a silo does not hold, uniformly subsampled to the same context budget."""
    rng = np.random.default_rng(seed * 31 + k)
    s = F.S[k]
    Xl, yl, Xq, _ = F.local(k)
    rows, ys = [], []
    for j in F.rich:
        t = F.S[j]
        Xm = F.enc.to_model(t['Z']).copy()
        miss = [c for c in F.all_cols if c not in t['cols']]
        Xm[:, miss] = np.nan
        rows.append(Xm[:, s['cols']]); ys.append(t['y'])
    Xp, yp = np.vstack(rows), np.concatenate(ys)
    sel = rng.choice(len(Xp), size=min(F.cfg['S'], len(Xp)), replace=False)
    Xc, yc = np.vstack([Xl, Xp[sel]]), np.concatenate([yl, yp[sel]])
    return tfm_predict(Xc, yc, Xq, F.cfg['n_est'], seed)


# ---------------------------------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, h),
                                       torch.nn.ReLU(), torch.nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _mlp_input(F, s, rows):
    E = F.enc.encode(s['Z'][rows], s['cols'])
    mask = np.zeros(len(F.all_cols)); mask[s['cols']] = 1
    return np.hstack([E, np.repeat(mask[None], len(rows), 0)])


def clipped_grad_sum(model, X, y, clip):
    """Sum over examples of per example gradients clipped to L2 norm `clip`, computed
    exactly with the ghost norm identity for linear layers: for W in a Linear layer the
    per example gradient is delta_i a_i^T, so ||g_i||^2 = sum_l (||a_li||^2 + 1)||delta_li||^2."""
    acts, outs = [], []

    def hook(m, i, o):
        acts.append(i[0]); o.retain_grad(); outs.append(o)
    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Linear)]
    model.zero_grad()
    out = model(X)
    torch.nn.functional.binary_cross_entropy_with_logits(out, y, reduction='sum').backward()
    for h in hooks:
        h.remove()
    with torch.no_grad():
        n2 = torch.zeros(len(X))
        for a, o in zip(acts, outs):
            d = o.grad.reshape(len(X), -1)
            n2 += ((a.reshape(len(X), -1) ** 2).sum(1) + 1.0) * (d ** 2).sum(1)
        c = torch.clamp(clip / (torch.sqrt(n2) + 1e-12), max=1.0)
    model.zero_grad()
    out = model(X)
    (c * torch.nn.functional.binary_cross_entropy_with_logits(out, y, reduction='none')).sum().backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def train_dp_fedavg(F, seed, rounds=None, clip=1.0, lr=None, batch=None):
    """DP-FedSGD over all silos with labels (union schema, zero fill + availability mask).
    Each silo clips per example gradients to `clip` and adds N(0, sigma^2 I) to the summed
    gradient of a minibatch; full participation, no amplification by sampling is claimed,
    so R rounds are R * clip^2 / (2 sigma^2)-zCDP per silo (sigma set to meet the budget)."""
    rounds = F.cfg.get('fa_rounds', 100) if rounds is None else rounds
    lr = F.cfg.get('fa_lr', 1e-2) if lr is None else lr
    batch = F.cfg.get('fa_batch', 4096) if batch is None else batch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = []
    for k, s in enumerate(F.S):
        rows = s['lab']
        data.append((torch.tensor(_mlp_input(F, s, rows), dtype=torch.float32),
                     torch.tensor(s['y'][rows], dtype=torch.float32)))
    model = MLP(data[0][0].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rho = fh.rho_from_eps(F.cfg['eps'], F.cfg['delta'])
    sigma = fh.gauss_sigma(clip, rho / rounds)
    ntot = sum(len(d[1]) for d in data)
    for r in range(rounds):
        agg = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        for X, y in data:
            idx = torch.as_tensor(rng.choice(len(y), size=min(batch, len(y)), replace=False))
            g = clipped_grad_sum(model, X[idx], y[idx], clip)
            for n in agg:
                gs = g[n] + (sigma * torch.randn_like(g[n]) if sigma > 0 else 0)
                agg[n] += gs / len(idx) * (len(y) / ntot)
        opt.zero_grad()
        for n, p in model.named_parameters():
            p.grad = agg[n]
        opt.step()
    return model


def run_fedavg(F, k, seed, finetune=False):
    key = ('fedavg', F.cfg['eps'], seed)
    if key not in F._protos:
        F._protos[key] = train_dp_fedavg(F, seed)
    model = F._protos[key]
    s = F.S[k]
    if finetune:
        import copy
        model = copy.deepcopy(model)
        X = torch.tensor(_mlp_input(F, s, s['lab']), dtype=torch.float32)
        y = torch.tensor(s['y'][s['lab']], dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(30):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(model(X), y).backward()
            opt.step()
    with torch.no_grad():
        p1 = torch.sigmoid(model(torch.tensor(_mlp_input(F, s, s['test']), dtype=torch.float32))).numpy()
    return np.column_stack([1 - p1, p1])


METHODS = {
    'Local-GBDT': run_local_gbdt,
    'Local-TFM': run_local_tfm,
    'DP-FedAvg': lambda F, k, s: run_fedavg(F, k, s, False),
    'DP-FedAvg-FT': lambda F, k, s: run_fedavg(F, k, s, True),
    'DP-Proto-TFM': run_dp_proto,
    'DP-Synth-GBDT': lambda F, k, s: run_ours(F, k, s, learner='gbdt'),
    'Central-TFM': run_central,
    'Fed-HybridPFN': run_ours,
}
