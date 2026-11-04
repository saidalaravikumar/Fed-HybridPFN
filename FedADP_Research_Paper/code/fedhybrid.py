"""Fed-HybridPFN core: zCDP prototype release, unbalanced Sinkhorn SoftJoin completion,
prior-aligned context resampling (PACR), and context synthesis for a frozen tabular
foundation model."""
import numpy as np
from scipy.special import logsumexp


# ----------------------------------------------------------------- accounting
def rho_from_eps(eps, delta):
    """Largest zCDP rho whose Bun-Steinke conversion eps = rho + 2 sqrt(rho ln(1/delta)) meets eps."""
    if not np.isfinite(eps):
        return np.inf
    L = np.log(1.0 / delta)
    return (np.sqrt(L + eps) - np.sqrt(L)) ** 2


def eps_from_rho(rho, delta):
    return rho + 2.0 * np.sqrt(rho * np.log(1.0 / delta))


def gauss_sigma(sens, rho):
    """Gaussian mechanism with L2 sensitivity `sens` is rho-zCDP for sigma = sens/sqrt(2 rho)."""
    return 0.0 if not np.isfinite(rho) else sens / np.sqrt(2.0 * rho)


# ----------------------------------------------------------------- DP prototypes
def dp_prototypes(E, num_mask, M, rho, C2, T=3, rng=None, min_count=None, init=None,
                  split=(0.4, 0.15, 0.70, 0.15), shrink=True, shrink_mode='eb', keep_truth=False):
    """Noise aware DP Lloyd clustering of encoded records E (n x d), ||e||^2 <= C2 = |A|.

    Budget: a fraction split[0] of rho is spread over T Lloyd iterations (each releases
    noisy counts and sums, 1:3 budget ratio); the rest goes to one final release of
    counts, sums and numeric sums of squares in proportions split[1:]. By zCDP sequential
    composition the routine is rho-zCDP under add/remove of one record. Post processing:
    empirical Bayes shrinkage of every noisy mean towards the count weighted pooled mean,
    with the between prototype variance estimated from the released means."""
    rng = np.random.default_rng() if rng is None else rng
    n, d = E.shape
    assert n == 0 or np.max(np.sum(E ** 2, 1)) <= C2 + 1e-9
    nn = int(num_mask.sum())
    r_it = rho * split[0] / max(T, 1) if T > 0 else 0.0
    r_f = rho * (1 - split[0]) if T > 0 else rho
    w = np.array(split[1:]); w = w / w.sum()
    if nn == 0:
        w = np.array([w[0], w[1] + w[2], 0.0])
    s_cnt = gauss_sigma(1.0, 0.25 * r_it) if T > 0 else 0
    s_sum = gauss_sigma(np.sqrt(C2), 0.75 * r_it) if T > 0 else 0
    s_cnt_f = gauss_sigma(1.0, w[0] * r_f)
    s_sum_f = gauss_sigma(np.sqrt(C2), w[1] * r_f)
    s_sq_f = gauss_sigma(np.sqrt(max(nn, 1)), w[2] * r_f) if nn else 0.0
    if init is None:
        cen = rng.random((M, d)) * 0.5 + 0.25
        cen[:, ~num_mask] = rng.random((M, int((~num_mask).sum())))
    else:                       # centres that are themselves DP outputs (post processing)
        cen = np.array(init, dtype=float); M = len(cen)
    if min_count is None:       # discard prototypes whose count is within 3 noise std of 0
        min_count = max(5.0, 3.0 * s_cnt_f)
    for t in range(T + 1):
        dist = (E ** 2).sum(1)[:, None] - 2 * E @ cen.T + (cen ** 2).sum(1)[None]
        a = np.argmin(dist, 1)
        cnt = np.bincount(a, minlength=M).astype(float)
        sm = np.zeros((M, d)); np.add.at(sm, a, E)
        if t < T:
            cnt_n = cnt + rng.normal(0, s_cnt, M)
            sm_n = sm + rng.normal(0, s_sum, (M, d))
            ok = cnt_n > max(5.0, 3.0 * s_cnt)
            cen[ok] = np.clip(sm_n[ok] / cnt_n[ok, None], 0, 1)
            cen[~ok] = rng.random((int((~ok).sum()), d))   # re-seed from the public domain
        else:
            sq = np.zeros((M, d)); np.add.at(sq, a, E ** 2)
            sq[:, ~num_mask] = 0.0      # only numeric square sums are released (sensitivity sqrt(nn))
            cnt_n = cnt + rng.normal(0, s_cnt_f, M)
            sm_n = sm + rng.normal(0, s_sum_f, (M, d))
            sq_n = sq + (rng.normal(0, s_sq_f, (M, d)) if nn else 0.0)
    keep = cnt_n > min_count
    cnt_n = np.maximum(cnt_n[keep], 1.0); sm_n = sm_n[keep]; sq_n = sq_n[keep] if nn else None
    mu = sm_n / cnt_n[:, None]
    nv = (s_sum_f / cnt_n) ** 2                                     # noise variance per coord
    lamv = np.ones_like(mu)
    if shrink and len(mu) > 1 and shrink_mode != 'none':
        pool = (cnt_n[:, None] * mu).sum(0) / cnt_n.sum()
        dev = mu - pool
        if shrink_mode == 'pooled':                                 # full pooling
            lamv = np.zeros_like(mu)
        elif shrink_mode == 'js':                                   # positive part James-Stein
            K = len(mu)
            ss = (dev ** 2).sum(0)
            c = np.clip(1 - max(K - 3, 0) * nv.mean() / np.maximum(ss, 1e-12), 0, 1) if K > 3 else np.ones(d)
            lamv = np.repeat(c[None], K, 0)
        else:                                                       # empirical Bayes
            if shrink_mode == 'robust':
                mad = np.median(np.abs(dev - np.median(dev, 0)), 0) * 1.4826
                tau2 = np.maximum(mad ** 2 - np.median(nv), 1e-4)
            else:
                tau2 = np.maximum(np.average(dev ** 2, axis=0, weights=cnt_n)
                                  - np.average(nv, weights=cnt_n), 1e-4)
            lamv = tau2[None] / (tau2[None] + nv[:, None])
        mu = pool + lamv * dev
    pv = lamv * nv[:, None]                     # Gaussian posterior variance of each mean
    mu = np.clip(mu, 0, 1)
    if nn:
        var = sq_n / cnt_n[:, None] - mu ** 2
    else:
        var = np.zeros_like(mu)
    var = np.where(num_mask[None], np.clip(var, 1e-3, 0.25), np.clip(mu * (1 - mu), 1e-4, 0.25))
    out = dict(n=cnt_n, mu=mu, var=var, pv=pv, lam=lamv.mean(1), nv=nv,
                sigma=dict(cnt=s_cnt_f, sum=s_sum_f, sq=s_sq_f, it_cnt=s_cnt, it_sum=s_sum))
    if keep_truth:          # simulation only: true cell means and counts for error audits
        out['mu_true'] = sm[keep] / np.maximum(cnt[keep], 1)[:, None]
        out['n_true'] = cnt[keep]
        out['assign'] = a; out['keep'] = keep
    return out


def exact_prototypes(E, num_mask, M, rng=None, T=10):
    """Non-private Lloyd (epsilon = infinity reference)."""
    from sklearn.cluster import KMeans
    km = KMeans(M, n_init=1, max_iter=T * 5, random_state=int(rng.integers(1 << 30))).fit(E)
    a = km.labels_
    cnt = np.bincount(a, minlength=M).astype(float)
    mu = np.zeros((M, E.shape[1])); sq = np.zeros_like(mu)
    np.add.at(mu, a, E); np.add.at(sq, a, E ** 2)
    mu /= np.maximum(cnt, 1)[:, None]; var = sq / np.maximum(cnt, 1)[:, None] - mu ** 2
    var = np.where(num_mask[None], np.clip(var, 1e-4, 0.25), np.clip(mu * (1 - mu), 1e-4, 0.25))
    keep = cnt > 0
    return dict(n=cnt[keep], mu=mu[keep], var=var[keep], pv=np.zeros_like(mu[keep]),
                lam=np.ones(int(keep.sum())), nv=np.zeros(int(keep.sum())), sigma=None)


# ----------------------------------------------------------------- Sinkhorn
N_SINKHORN = 0


def sinkhorn_unbalanced(a, b, Cm, eta=0.05, tau=1.0, iters=500, tol=1e-9):
    """Entropic unbalanced OT (KL marginal penalties of weight tau), log-domain scaling
    (Chizat et al.). tau = inf gives balanced Sinkhorn. Returns the coupling."""
    global N_SINKHORN
    N_SINKHORN += 1
    la, lb = np.log(a + 1e-300), np.log(b + 1e-300)
    f = np.zeros(len(a)); g = np.zeros(len(b))
    lam = 1.0 if not np.isfinite(tau) else tau / (tau + eta)
    K = -Cm / eta
    for _ in range(iters):
        f_old = f
        f = lam * eta * (la - logsumexp(K + g[None] / eta, 1))
        g = lam * eta * (lb - logsumexp(K + f[:, None] / eta, 0))
        if np.max(np.abs(f - f_old)) < tol:
            break
    return np.exp(K + f[:, None] / eta + g[None] / eta)


# ----------------------------------------------------------------- SoftJoin completion
def complete_prototypes(protos, enc, all_cols, eta=0.05, tau=1.0, mode='softjoin', cost='plain', diag=None, topk=None):
    """Fill the columns each silo does not observe.

    protos: list of dicts with keys silo, cols, y, n, mu, var (mu/var on full encoded dim,
    NaN on unobserved dims). mode='softjoin' uses class matched unbalanced Sinkhorn on shared
    columns; mode='mean' uses the pooled DP marginal mean (no linkage)."""
    d = enc.dim
    num_dims = np.zeros(d, bool)
    for j, c in enumerate(enc.schema):
        if c['type'] == 'num':
            num_dims[enc.slices[j]] = True
    # pooled marginal per class (post processing of DP releases)
    protos = [p for p in protos if len(p['n'])]
    pooled = {}
    for y in (0, 1):
        P = [p for p in protos if p['y'] == y]
        mu_sum = np.zeros(d); w_sum = np.zeros(d)
        for p in P:
            obs = ~np.isnan(p['mu'][0])
            mu_sum[obs] += (p['n'][:, None] * p['mu'][:, obs]).sum(0)
            w_sum[obs] += p['n'].sum()
        pooled[y] = mu_sum / np.maximum(w_sum, 1e-9)
    out = []
    for p in protos:
        mu, var = p['mu'].copy(), p['var'].copy()
        miss_cols = [c for c in all_cols if c not in p['cols']]
        if not miss_cols:
            out.append(dict(p, mu=mu, var=var, mass=np.ones(len(mu)))); continue
        miss = enc.dims(miss_cols)
        acc_mu = np.zeros((len(mu), len(miss))); acc_m2 = np.zeros_like(acc_mu)
        acc_w = np.zeros((len(mu), len(miss)))
        if mode in ('softjoin', 'nearest'):
            cand = []
            for q in protos:
                if q is p or q['y'] != p['y'] or q['silo'] == p['silo']:
                    continue
                shared = [c for c in p['cols'] if c in q['cols']]
                give = [c for c in miss_cols if c in q['cols']]
                if shared and give:
                    cand.append((len(shared), q['silo'], q, shared, give))
            if topk:
                # pruned matching: every missing column is filled from at most topk donors, taken in
                # order of the number of shared columns (the donors the weight below favours anyway)
                cand.sort(key=lambda t: (-t[0], t[1]))
                used = {c: 0 for c in miss_cols}; keep = []
                for t in cand:
                    if any(used[c] < topk for c in t[4]):
                        keep.append(t)
                        for c in t[4]:
                            used[c] += 1
                cand = keep
            for _, _, q, shared, give in cand:
                sd = enc.dims(shared)
                Cm = ((mu[:, None, sd] - q['mu'][None, :, sd]) ** 2).sum(-1) / len(shared)
                if cost == 'debiased':
                    # E||mu_hat_m - mu_hat_q||^2 = ||mu_m - mu_q||^2 + tr(V_m) + tr(V_q): remove the
                    # noise part so that noisy prototypes are not pushed away from every partner
                    bias = (np.nan_to_num(p['pv'][:, sd]).sum(1)[:, None] + np.nan_to_num(q['pv'][:, sd]).sum(1)[None]) / len(shared)
                    Cm = np.maximum(Cm - bias, 0.0)
                a = p['n'] / p['n'].sum(); b = q['n'] / q['n'].sum()
                if mode == 'nearest':
                    pi = np.zeros_like(Cm); pi[np.arange(len(a)), Cm.argmin(1)] = a
                else:
                    pi = sinkhorn_unbalanced(a, b, Cm, eta=eta, tau=tau)
                if diag is not None:
                    diag.setdefault('mass_kept', []).append(float(pi.sum()))
                    diag.setdefault('cost', []).append(float((pi * Cm).sum() / max(pi.sum(), 1e-12)))
                    diag.setdefault('marg_viol', []).append(float(np.abs(pi.sum(1) - a).sum() + np.abs(pi.sum(0) - b).sum()))
                row = pi.sum(1) + 1e-300
                gd = enc.dims(give)
                pos = np.searchsorted(miss, gd)
                m_q = q['mu'][:, gd]; v_q = q['var'][:, gd]
                cond_mu = pi @ m_q / row[:, None]
                cond_m2 = pi @ (v_q + m_q ** 2) / row[:, None]
                # donor weight: transported mass fraction x number of shared columns
                w = (row / a)[:, None] * len(shared)
                acc_mu[:, pos] += w * cond_mu; acc_m2[:, pos] += w * cond_m2; acc_w[:, pos] += w
        fill = pooled[p['y']][miss][None].repeat(len(mu), 0)
        ok = acc_w > 0
        cm = np.where(ok, acc_mu / np.maximum(acc_w, 1e-300), fill)
        c2 = np.where(ok, acc_m2 / np.maximum(acc_w, 1e-300), fill * (1 - fill) + fill ** 2)
        mu[:, miss] = np.clip(cm, 0, 1)
        vv = c2 - cm ** 2
        var[:, miss] = np.where(num_dims[miss][None], np.clip(vv, 1e-4, 0.25),
                                np.clip(mu[:, miss] * (1 - mu[:, miss]), 1e-4, 0.25))
        out.append(dict(p, mu=mu, var=var, mass=ok.mean(1)))
    return out


# ----------------------------------------------------------------- PACR + synthesis
def mixture_resp(U, mu, var, logw):
    """Posterior responsibilities of a diagonal Gaussian mixture for rows U."""
    iv = 1.0 / var
    quad = (U ** 2) @ iv.T - 2.0 * U @ (mu * iv).T + (mu ** 2 * iv).sum(1)[None]
    ll = -0.5 * quad - 0.5 * np.log(var).sum(-1)[None]
    ll = ll + logw[None]
    return np.exp(ll - logsumexp(ll, 1, keepdims=True))


def context_weights(protos, enc, cols, U=None, align=True, lam=0.5, bw=1.0, batch=4000):
    """Weight of every completed prototype in the context of a target silo with columns
    `cols`. With align=False the weights are the released (global) counts. With
    align=True they are shrunk towards the local responsibility mass of the silo's own
    unlabelled rows U under an isotropic kernel of bandwidth h^2 = bw * median nearest
    prototype distance (prior aligned context resampling, PACR):
        w_p = (1 - lam) * n_p / sum n + lam * mean_u r_p(u),
        r_p(u) proportional to n_p exp(-||u - mu_p||^2 / (2 h^2)).
    U never leaves the silo, so the weights cost no privacy budget."""
    dims = enc.dims(cols)
    mu = np.concatenate([p['mu'][:, dims] for p in protos])
    var = np.concatenate([p['var'][:, dims] for p in protos])
    n = np.concatenate([p['n'] for p in protos])
    y = np.concatenate([np.full(len(p['n']), p['y']) for p in protos])
    w0 = n / n.sum()
    w = w0
    if align and U is not None and len(U):
        Ud = U[:, dims]
        m = np.zeros(len(n)); h2 = None
        for s in range(0, len(Ud), batch):
            B = Ud[s:s + batch]
            d2 = (B ** 2).sum(1)[:, None] - 2 * B @ mu.T + (mu ** 2).sum(1)[None]
            d2 = np.maximum(d2, 0)
            if h2 is None:
                h2 = bw * max(np.median(d2.min(1)), 1e-3)
            ll = np.log(w0 + 1e-300)[None] - d2 / (2 * h2)
            m += np.exp(ll - logsumexp(ll, 1, keepdims=True)).sum(0)
        w = (1 - lam) * w0 + lam * m / m.sum()
    return dict(mu=mu, var=var, y=y, w=w, dims=dims)


def synthesize(ctx, enc, cols, S, rng):
    """Draw S synthetic context rows (model space) from the weighted prototype mixture."""
    k = rng.choice(len(ctx['w']), size=S, p=ctx['w'])
    X = np.zeros((S, len(cols)))
    off = 0
    for t, j in enumerate(cols):
        c = enc.schema[j]
        w = 1 if c['type'] == 'num' else c['levels']
        m = ctx['mu'][k, off:off + w]; v = ctx['var'][k, off:off + w]
        if c['type'] == 'num':
            X[:, t] = np.clip(m[:, 0] + np.sqrt(v[:, 0]) * rng.normal(size=S), 0, 1)
        else:
            pr = np.clip(m, 1e-6, None); pr /= pr.sum(1, keepdims=True)
            u = rng.random(S)[:, None]
            X[:, t] = (pr.cumsum(1) < u).sum(1).clip(0, w - 1)
        off += w
    return X, ctx['y'][k]


def density_ratio_resample(Xs, ys, Xu, S, rng, clip=(0.05, 20.0), learner='lr'):
    """Prior aligned context resampling by discriminative density ratio estimation.
    A probabilistic classifier separates synthetic rows Xs (label 0) from the silo's own
    unlabelled rows Xu (label 1); w(x) = c * p(1|x)/p(0|x) estimates p_t(x)/p_s(x).
    Synthetic rows (with their labels) are resampled proportionally to clipped w, which
    targets P_t(x) P(y|x) under the covariate shift assumption. Runs inside the silo."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    Z = np.vstack([Xs, Xu]); t = np.r_[np.zeros(len(Xs)), np.ones(len(Xu))]
    sw = np.r_[np.full(len(Xs), 0.5 / len(Xs)), np.full(len(Xu), 0.5 / len(Xu))] * len(Z)
    if learner == 'lr':
        clf = LogisticRegression(C=1.0, max_iter=500).fit(Z, t, sample_weight=sw)
    else:
        clf = HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.1).fit(Z, t, sample_weight=sw)
    p = np.clip(clf.predict_proba(Xs)[:, 1], 1e-4, 1 - 1e-4)
    w = np.clip(p / (1 - p), *clip); w /= w.sum()
    k = rng.choice(len(Xs), size=S, p=w)
    ess = 1.0 / np.sum(w ** 2)
    return Xs[k], ys[k], ess


def em_label_shift(P, prior_src, iters=50):
    """Saerens et al. EM correction of predicted probabilities to an unknown target prior."""
    pt = prior_src.copy()
    for _ in range(iters):
        Q = P * (pt / prior_src)[None]; Q /= Q.sum(1, keepdims=True)
        pt_new = Q.mean(0)
        if np.abs(pt_new - pt).max() < 1e-6:
            break
        pt = pt_new
    Q = P * (pt / prior_src)[None]
    return Q / Q.sum(1, keepdims=True)
