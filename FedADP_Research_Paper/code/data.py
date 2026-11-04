"""Dataset loading, public encoding and hybrid silo partitioning for Fed-HybridPFN.

Every transform here uses only public schema knowledge (documented code books and
fixed value ranges), so that the encoding itself consumes no privacy budget.
"""
import os
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), '..', 'data_raw')
PROC = os.path.join(os.path.dirname(__file__), '..', 'data_proc')
STATES = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'WA', 'MI']

# public ACS occupation code ranges (2018 OCCP, SOC major groups)
OCCP_EDGES = [0, 500, 1000, 1300, 1600, 2000, 2100, 2200, 2600, 3000, 3600, 3700, 4000,
              4200, 4300, 4700, 5000, 6000, 6200, 7000, 7700, 9000, 9800, 10000]


def num(name, lo, hi):
    return dict(name=name, type='num', lo=float(lo), hi=float(hi))


def cat(name, levels):
    return dict(name=name, type='cat', levels=int(levels))


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------------------
def _load_acs(task):
    from folktables import ACSDataSource, ACSIncome, ACSEmployment
    src = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir=RAW)
    P = ACSIncome if task == 'income' else ACSEmployment
    frames, regions = [], []
    for r, st in enumerate(STATES):
        d = src.get_data(states=[st], download=False)
        X, y, _ = P.df_to_pandas(d)
        X = X.copy(); X['__y'] = y.values.ravel().astype(int)
        frames.append(X); regions.append(np.full(len(X), r))
    df = pd.concat(frames, ignore_index=True)
    reg = np.concatenate(regions)
    y = df.pop('__y').values
    if task == 'income':
        df['OCCP'] = np.digitize(df['OCCP'].values, OCCP_EDGES[1:-1])
        p = df['POBP'].values
        df['POBP'] = np.select([p < 100, p < 200, p < 300, p < 400, p < 500], [0, 1, 2, 3, 4], 5)
        df['COW'] -= 1; df['MAR'] -= 1; df['SEX'] -= 1; df['RAC1P'] -= 1
        schema = [num('AGEP', 17, 95), num('SCHL', 1, 24), num('WKHP', 1, 99),
                  cat('SEX', 2), cat('COW', 8), cat('MAR', 5), cat('OCCP', 23),
                  cat('POBP', 6), cat('RELP', 18), cat('RAC1P', 9)]
        bridge = ['AGEP', 'SEX', 'SCHL']
    else:
        for c in ['MAR', 'DIS', 'CIT', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'SEX', 'RAC1P']:
            df[c] -= 1
        schema = [num('AGEP', 0, 95), num('SCHL', 0, 24), cat('SEX', 2), cat('ESP', 9),
                  cat('MAR', 5), cat('RELP', 18), cat('DIS', 2), cat('CIT', 5), cat('MIG', 4),
                  cat('MIL', 5), cat('ANC', 4), cat('NATIVITY', 2), cat('DEAR', 2),
                  cat('DEYE', 2), cat('DREM', 3), cat('RAC1P', 9)]
        bridge = ['AGEP', 'SEX', 'SCHL']
    return df, y, reg, schema, bridge, None


def _load_taiwan():
    t = pd.read_excel(os.path.join(RAW, 'taiwan', 'default of credit card clients.xls'), header=1)
    y = t.pop('default payment next month').values.astype(int)
    df = pd.DataFrame()
    df['LIMIT'] = np.log(t['LIMIT_BAL'].values)
    df['AGE'] = t['AGE'].values
    df['SEX'] = t['SEX'].values - 1
    df['EDU'] = np.clip(t['EDUCATION'].values, 0, 4)       # 0,5,6 are "unknown" codes
    df.loc[df['EDU'] == 0, 'EDU'] = 4
    df['EDU'] -= 1
    df['MAR'] = np.clip(t['MARRIAGE'].values, 0, 3)
    schema = [num('LIMIT', np.log(1e4), np.log(1e6)), num('AGE', 20, 80), cat('SEX', 2),
              cat('EDU', 4), cat('MAR', 4)]
    groups = {}
    for i, c in enumerate(['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']):
        df['PAY%d' % i] = t[c].values; schema.append(num('PAY%d' % i, -2, 8)); groups['PAY%d' % i] = 'PAY'
    for i in range(1, 7):
        df['BILL%d' % i] = _slog(t['BILL_AMT%d' % i].values); schema.append(num('BILL%d' % i, -14, 14)); groups['BILL%d' % i] = 'BILL'
        df['PAMT%d' % i] = _slog(t['PAY_AMT%d' % i].values); schema.append(num('PAMT%d' % i, 0, 14)); groups['PAMT%d' % i] = 'PAMT'
    # strata used only to induce regional covariate shift (age x credit limit)
    strata = (np.digitize(df['AGE'], [30, 40, 50]) * 3 + np.digitize(df['LIMIT'], [np.log(5e4), np.log(2e5)]))
    bridge = ['AGE', 'SEX', 'EDU', 'MAR']
    return df, y, strata, schema, bridge, groups


def _load_gmsc():
    g = pd.read_parquet(os.path.join(RAW, 'gmsc.pq'))
    y = (g.pop('FinancialDistressNextTwoYears').astype(str).values == 'Yes').astype(int)
    df = pd.DataFrame()
    df['AGE'] = np.clip(g['age'].values, 21, 100)
    inc = g['MonthlyIncome'].values.astype(float)
    df['INCOME'] = np.log1p(np.where(np.isnan(inc), 5000.0, inc))   # fixed public fill value
    dep = g['NumberOfDependents'].values.astype(float)
    df['DEPEND'] = np.clip(np.where(np.isnan(dep), 0, dep), 0, 10)
    df['UTIL'] = np.log1p(np.clip(g['RevolvingUtilizationOfUnsecuredLines'].values, 0, 1e4))
    df['DEBT'] = np.log1p(np.clip(g['DebtRatio'].values, 0, 1e6))
    df['LINES'] = np.clip(g['NumberOfOpenCreditLinesAndLoans'].values, 0, 60)
    df['RE'] = np.clip(g['NumberRealEstateLoansOrLines'].values, 0, 20)
    df['D30'] = np.clip(g['NumberOfTime30-59DaysPastDueNotWorse'].values, 0, 20)
    df['D60'] = np.clip(g['NumberOfTime60-89DaysPastDueNotWorse'].values, 0, 20)
    df['D90'] = np.clip(g['NumberOfTimes90DaysLate'].values, 0, 20)
    schema = [num('AGE', 21, 100), num('INCOME', 0, 14), num('DEPEND', 0, 10), num('UTIL', 0, 9.3),
              num('DEBT', 0, 13.9), num('LINES', 0, 60), num('RE', 0, 20), num('D30', 0, 20),
              num('D60', 0, 20), num('D90', 0, 20)]
    groups = {'D30': 'DELQ', 'D60': 'DELQ', 'D90': 'DELQ'}
    strata = np.digitize(df['AGE'], [35, 50, 65]) * 3 + np.digitize(df['INCOME'], [np.log1p(3500), np.log1p(7500)])
    bridge = ['AGE', 'INCOME', 'DEPEND']
    return df, y, strata, schema, bridge, groups


DATASETS = {'income': lambda: _load_acs('income'), 'employment': lambda: _load_acs('employment'),
            'taiwan': _load_taiwan, 'gmsc': _load_gmsc}


def load(name):
    """Returns dict with raw code matrix Z (n x p, ints for cat, floats for num), y, region/strata, schema."""
    os.makedirs(PROC, exist_ok=True)
    f = os.path.join(PROC, name + '.npz')
    if not os.path.exists(f):
        df, y, reg, schema, bridge, groups = DATASETS[name]()
        Z = np.column_stack([df[c['name']].values.astype(float) for c in schema])
        np.savez_compressed(f, Z=Z, y=y, reg=reg)
    d = np.load(f)
    _, _, _, schema, bridge, groups = _meta(name)
    return dict(Z=d['Z'], y=d['y'], reg=d['reg'], schema=schema, bridge=bridge,
                groups=groups or {}, name=name)


def _meta(name):
    # schema only (cheap): rebuild from the loader definitions without data
    if name in ('income', 'employment'):
        if name == 'income':
            schema = [num('AGEP', 17, 95), num('SCHL', 1, 24), num('WKHP', 1, 99), cat('SEX', 2),
                      cat('COW', 8), cat('MAR', 5), cat('OCCP', 23), cat('POBP', 6), cat('RELP', 18),
                      cat('RAC1P', 9)]
        else:
            schema = [num('AGEP', 0, 95), num('SCHL', 0, 24), cat('SEX', 2), cat('ESP', 9),
                      cat('MAR', 5), cat('RELP', 18), cat('DIS', 2), cat('CIT', 5), cat('MIG', 4),
                      cat('MIL', 5), cat('ANC', 4), cat('NATIVITY', 2), cat('DEAR', 2),
                      cat('DEYE', 2), cat('DREM', 3), cat('RAC1P', 9)]
        return None, None, None, schema, ['AGEP', 'SEX', 'SCHL'], None
    if name == 'taiwan':
        schema = [num('LIMIT', np.log(1e4), np.log(1e6)), num('AGE', 20, 80), cat('SEX', 2),
                  cat('EDU', 4), cat('MAR', 4)]
        groups = {}
        for i in range(6):
            schema.append(num('PAY%d' % i, -2, 8)); groups['PAY%d' % i] = 'PAY'
        for i in range(1, 7):
            schema.append(num('BILL%d' % i, -14, 14)); groups['BILL%d' % i] = 'BILL'
            schema.append(num('PAMT%d' % i, 0, 14)); groups['PAMT%d' % i] = 'PAMT'
        return None, None, None, schema, ['AGE', 'SEX', 'EDU', 'MAR'], groups
    schema = [num('AGE', 21, 100), num('INCOME', 0, 14), num('DEPEND', 0, 10), num('UTIL', 0, 9.3),
              num('DEBT', 0, 13.9), num('LINES', 0, 60), num('RE', 0, 20), num('D30', 0, 20),
              num('D60', 0, 20), num('D90', 0, 20)]
    return None, None, None, schema, ['AGE', 'INCOME', 'DEPEND'], {'D30': 'DELQ', 'D60': 'DELQ', 'D90': 'DELQ'}


# ---------------------------------------------------------------------------
class Encoder:
    """Public encoder: numeric -> [0,1] by fixed bounds, categorical -> one hot.
    For a record restricted to available columns A, ||e(x)||_2^2 <= |A| (bounded sensitivity)."""

    def __init__(self, schema):
        self.schema = schema
        self.slices, s = [], 0
        for c in schema:
            w = 1 if c['type'] == 'num' else c['levels']
            self.slices.append(slice(s, s + w)); s += w
        self.dim = s
        self.names = [c['name'] for c in schema]

    def encode(self, Z, cols=None):
        cols = range(len(self.schema)) if cols is None else cols
        out = np.zeros((len(Z), self.dim))
        for j in cols:
            c = self.schema[j]
            if c['type'] == 'num':
                out[:, self.slices[j]] = np.clip((Z[:, j:j + 1] - c['lo']) / (c['hi'] - c['lo']), 0, 1)
            else:
                k = np.clip(Z[:, j].astype(int), 0, c['levels'] - 1)
                out[np.arange(len(Z)), self.slices[j].start + k] = 1.0
        return out

    def dims(self, cols):
        return np.concatenate([np.arange(self.slices[j].start, self.slices[j].stop) for j in cols])

    def to_model(self, Z):
        """Model input for TFM / GBDT / MLP: numeric scaled to [0,1], categorical integer code."""
        X = np.array(Z, dtype=float)
        for j, c in enumerate(self.schema):
            if c['type'] == 'num':
                X[:, j] = np.clip((X[:, j] - c['lo']) / (c['hi'] - c['lo']), 0, 1)
        return X


# ---------------------------------------------------------------------------
def make_silos(D, K=20, n_targets=8, p_feat=0.6, overlap=0.5, n_lab=64, n_test=1000,
               cap=30000, sel_strength=1.0, seed=0, disjoint=False):
    """Hybrid vertical/horizontal partition.

    Regions (US states for ACS; Dirichlet strata mixtures for credit data) give horizontal
    covariate shift. Each region hosts K/G agencies. Agency a>0 shares a fraction `overlap`
    of its entities with an earlier agency of the same region (the same individuals) and
    draws the rest from the region with a selection bias on a random bridge direction, so
    record sets overlap only partially and there is no linkage key. Each silo observes the
    bridge columns plus each other column (or column group) with probability p_feat.
    """
    rng = np.random.default_rng(seed)
    Z, y, reg, schema = D['Z'], D['y'], D['reg'], D['schema']
    names = [c['name'] for c in schema]
    bidx = [names.index(b) for b in D['bridge']]
    nonb = [j for j in range(len(schema)) if j not in bidx]
    grp = {}
    for j in nonb:
        grp.setdefault(D['groups'].get(names[j], names[j]), []).append(j)
    glist = list(grp.values())
    enc = Encoder(schema)

    if D['name'] in ('income', 'employment'):
        regions = [np.where(reg == r)[0] for r in range(len(STATES))]
    else:  # build G synthetic regions as Dirichlet mixtures over strata
        G = max(K // 2, 2)
        strata = np.unique(reg)
        pools = {s: rng.permutation(np.where(reg == s)[0]).tolist() for s in strata}
        props = rng.dirichlet(0.5 * np.ones(len(strata)), size=G)
        size = len(Z) // G
        regions = []
        for g in range(G):
            idx = []
            for s, pr in zip(strata, props[g]):
                take = int(round(pr * size)); idx += pools[s][:take]; pools[s] = pools[s][take:]
            regions.append(np.array(idx, dtype=int))
    G = len(regions)
    per = int(np.ceil(K / G))
    silos = []
    Bz = enc.encode(Z, bidx)[:, enc.dims(bidx)]
    for g in range(G):
        R = rng.permutation(regions[g])
        if len(silos) >= K:
            break
        n_agency = min(per, K - len(silos))
        base = min(cap, int(len(R) / max(n_agency, 2) * rng.uniform(0.6, 1.0)))
        used = []
        for a in range(n_agency):
            n_a = max(1500, int(base * rng.uniform(0.5, 1.0)))
            if a == 0:
                idx = R[:n_a]
            else:
                prev = rng.choice(len(used))
                n_sh = int(overlap * n_a)
                shared = rng.choice(used[prev], size=min(n_sh, len(used[prev])), replace=False)
                rest = np.setdiff1d(R, shared)
                w = rng.normal(size=Bz.shape[1]); w /= np.linalg.norm(w)
                s = Bz[rest] @ w; s = (s - s.mean()) / (s.std() + 1e-9) if len(rest) else s
                pr = np.exp(sel_strength * s); pr /= max(pr.sum(), 1e-300)
                own = (rng.choice(rest, size=min(n_a - len(shared), len(rest)), replace=False, p=pr)
                       if len(rest) else np.zeros(0, dtype=R.dtype))
                idx = np.concatenate([shared, own])
            used.append(idx)
            avail = list(bidx)
            chosen = [gl for gl in glist if rng.random() < p_feat]
            if not chosen:
                chosen = [glist[rng.integers(len(glist))]]
            for gl in chosen:
                avail += gl
            silos.append(dict(idx=np.array(idx), region=g, cols=sorted(avail)))
    K = len(silos)
    tg = rng.choice(K, size=min(n_targets, K), replace=False)
    for k, s in enumerate(silos):
        s['Z'] = Z[s['idx']]; s['y'] = y[s['idx']]
        n = len(s['idx'])
        s['target'] = k in tg
        perm = rng.permutation(n)
        if s['target']:
            nt = min(n_test, n // 4)
            s['test'] = perm[:nt]
            pool = perm[nt:]
            # stratified few shot labels with at least 2 per class
            pos = pool[s['y'][pool] == 1]; neg = pool[s['y'][pool] == 0]
            npos = int(np.clip(round(n_lab * len(pos) / len(pool)), 2, min(len(pos), n_lab - 2)))
            lab = np.concatenate([pos[:npos], neg[:n_lab - npos]])
            s['lab'] = lab
            s['unl'] = np.setdiff1d(pool, lab)
        else:
            s['test'] = np.array([], dtype=int)
            s['lab'] = perm
            s['unl'] = np.array([], dtype=int)
    if disjoint:
        # evaluation control (simulation only): remove every individual of every target test
        # set from all label rich silos. Indices are used here to build the split and are never
        # visible to the algorithm.
        test_ids = np.concatenate([s['idx'][s['test']] for s in silos if s['target']])
        for s in silos:
            if s['target']:
                continue
            keep = ~np.isin(s['idx'], test_ids)
            s['idx'] = s['idx'][keep]; s['Z'] = s['Z'][keep]; s['y'] = s['y'][keep]
            s['lab'] = np.arange(len(s['idx']))
    return silos, enc
