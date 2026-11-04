"""Audit tests for the privacy accounting, Sinkhorn solver and SoftJoin completion."""
import numpy as np
import fedhybrid as fh
import data


def test_accounting():
    for eps in [0.5, 1, 2, 4, 8]:
        for delta in [1e-5, 1e-6]:
            rho = fh.rho_from_eps(eps, delta)
            assert abs(fh.eps_from_rho(rho, delta) - eps) < 1e-9
    # zCDP -> (eps,delta) must dominate the exact RDP conversion min_a rho*a + log(1/d)/(a-1)
    rho, d = 0.02, 1e-5
    a = np.linspace(1.001, 500, 200000)
    eps_rdp = np.min(rho * a + np.log(1 / d) / (a - 1))
    assert eps_rdp <= fh.eps_from_rho(rho, d) + 1e-9
    # Gaussian sigma: sens^2/(2 sigma^2) == rho
    s = fh.gauss_sigma(3.0, 0.01); assert abs(9 / (2 * s * s) - 0.01) < 1e-12
    print('accounting ok; eps=1,delta=1e-5 -> rho=%.5f' % fh.rho_from_eps(1, 1e-5))


def test_dp_noise_scale():
    """Empirical check: released counts have the advertised noise std."""
    rng = np.random.default_rng(0)
    E = np.zeros((5000, 2)); E[:, 0] = rng.random(5000); E[:, 1] = 1.0
    nm = np.array([True, False])
    errs = []
    for t in range(300):
        r = fh.dp_prototypes(E, nm, 1, rho=0.05, C2=2, T=0, rng=rng, min_count=-1e9, shrink=False)
        errs.append(r['n'][0] - 5000)
    sd = np.std(errs)
    expect = r['sigma']['cnt']
    # the three final components must exhaust rho exactly: sum sens^2/(2 sigma^2) = rho
    sg = r['sigma']
    used = 1 / (2 * sg['cnt'] ** 2) + 2 / (2 * sg['sum'] ** 2) + 1 / (2 * sg['sq'] ** 2)
    assert abs(used - 0.05) < 1e-9, used
    assert abs(sd / expect - 1) < 0.15, (sd, expect)
    print('noise scale ok: empirical %.2f vs advertised %.2f' % (sd, expect))


def test_sinkhorn():
    rng = np.random.default_rng(1)
    a = rng.random(7); a /= a.sum(); b = rng.random(9); b /= b.sum()
    C = rng.random((7, 9))
    P = fh.sinkhorn_unbalanced(a, b, C, eta=0.05, tau=np.inf, iters=5000)
    assert np.allclose(P.sum(1), a, atol=1e-6) and np.allclose(P.sum(0), b, atol=1e-6)
    import ot
    P2 = ot.sinkhorn(a, b, C, 0.05, numItermax=20000, stopThr=1e-12)
    assert np.allclose(P, P2, atol=1e-5)
    Pu = fh.sinkhorn_unbalanced(a, b, C, eta=0.05, tau=1.0, iters=5000)
    P3 = ot.unbalanced.sinkhorn_unbalanced(a, b, C, 0.05, 1.0, numItermax=20000, stopThr=1e-12, reg_type='entropy')
    assert np.allclose(Pu, P3, atol=1e-5), np.abs(Pu - P3).max()
    print('sinkhorn ok (balanced and unbalanced match POT)')


def test_softjoin_recovers_conditional():
    """Two silos with columns {b,u} and {b,v}; v depends on b. SoftJoin should fill v
    for silo 1 close to E[v|b], while the mean fill ignores b."""
    rng = np.random.default_rng(2)
    schema = [data.num('b', 0, 1), data.num('u', 0, 1), data.num('v', 0, 1)]
    enc = data.Encoder(schema)
    n = 20000
    b = rng.random(n); u = np.clip(b + 0.1 * rng.normal(size=n), 0, 1); v = np.clip(1 - b + 0.05 * rng.normal(size=n), 0, 1)
    Z = np.column_stack([b, u, v])
    protos = []
    for k, cols in enumerate([[0, 1], [0, 2]]):
        idx = np.arange(k, n, 2)
        E = enc.encode(Z[idx], cols)[:, enc.dims(cols)]
        r = fh.exact_prototypes(E, np.ones(2, bool), 12, rng=rng)
        mu = np.full((len(r['n']), 3), np.nan); var = mu.copy()
        mu[:, enc.dims(cols)] = r['mu']; var[:, enc.dims(cols)] = r['var']
        protos.append(dict(silo=k, y=0, cols=cols, n=r['n'], mu=mu, var=var))
    out = fh.complete_prototypes(protos, enc, [0, 1, 2], eta=0.01, tau=np.inf)
    p = out[0]
    err_sj = np.mean(np.abs(p['mu'][:, 2] - (1 - p['mu'][:, 0])))
    out_m = fh.complete_prototypes(protos, enc, [0, 1, 2], mode='mean')
    err_mean = np.mean(np.abs(out_m[0]['mu'][:, 2] - (1 - out_m[0]['mu'][:, 0])))
    assert err_sj < 0.5 * err_mean, (err_sj, err_mean)
    print('softjoin ok: |E[v|b] error| softjoin %.3f vs mean fill %.3f' % (err_sj, err_mean))


def test_ghost_clipping():
    import torch
    from torch.func import functional_call, vmap, grad
    import methods as M
    torch.manual_seed(0)
    model = M.MLP(7, h=16); X = torch.randn(64, 7); y = (torch.rand(64) > 0.5).float()
    g = M.clipped_grad_sum(model, X, y, clip=0.5)
    p = {n: q.detach() for n, q in model.named_parameters()}
    f = lambda p, x, t: torch.nn.functional.binary_cross_entropy_with_logits(functional_call(model, p, (x[None],)), t[None])
    G = vmap(grad(f), in_dims=(None, 0, 0))(p, X, y)
    nrm = torch.sqrt(sum((v.reshape(64, -1) ** 2).sum(1) for v in G.values()))
    c = torch.clamp(0.5 / nrm, max=1.0)
    err = max(float((g[n] - (G[n] * c.view(-1, *[1] * (G[n].dim() - 1))).sum(0)).abs().max()) for n in g)
    assert err < 1e-5, err
    print('ghost clipping ok: max abs diff to vmap %.2e' % err)


def test_empty_release():
    rng = np.random.default_rng(3)
    E = rng.random((30, 2)); r = fh.dp_prototypes(E, np.array([True, True]), 4, rho=1e-4, C2=2, T=1, rng=rng)
    assert len(r['n']) == 0          # tiny class at a tiny budget: everything filtered
    schema = [data.num('b', 0, 1), data.num('u', 0, 1)]; enc = data.Encoder(schema)
    empty = dict(silo=0, y=1, cols=[0], n=np.zeros(0), mu=np.zeros((0, 2)), var=np.zeros((0, 2)))
    full = dict(silo=1, y=1, cols=[0, 1], n=np.ones(3), mu=rng.random((3, 2)), var=np.full((3, 2), .01))
    out = fh.complete_prototypes([empty, full], enc, [0, 1])
    assert len(out) == 1
    print('empty release ok')


if __name__ == '__main__':
    test_empty_release()
    test_ghost_clipping()
    test_accounting(); test_dp_noise_scale(); test_sinkhorn(); test_softjoin_recovers_conditional()
