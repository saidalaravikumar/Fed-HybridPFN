"""Version 3 tables. Baselines that do not read the release (Local GBDT/TFM, DP FedAvg(+FT), pooled raw)
come from main.jsonl; methods that read the release (Fed-HybridPFN, pooled DP summary, DP synth GBDT)
were rerun with the final release in main3.jsonl. Seeds are the unit of replication."""
import json, os
import numpy as np
from scipy import stats

R = os.path.join(os.path.dirname(__file__), '..', 'results')
P = os.path.join(os.path.dirname(__file__), '..', 'paper')
NAMES = {'Local-GBDT': 'Local GBDT', 'Local-TFM': 'Local TFM', 'DP-FedAvg': 'DP FedAvg',
         'DP-FedAvg-FT': 'DP FedAvg+FT', 'DP-Proto-TFM': 'Pooled DP summary', 'DP-Synth-GBDT': 'DP synth GBDT',
         'Central-TFM': 'Pooled raw (no DP)', 'Fed-HybridPFN': 'Fed-HybridPFN'}
DS = ['income', 'employment', 'taiwan', 'gmsc']
DN = {'income': 'ACSIncome', 'employment': 'ACSEmployment', 'taiwan': 'Taiwan', 'gmsc': 'GMSC'}
ORDER = ['Local-GBDT', 'Local-TFM', 'DP-FedAvg', 'DP-FedAvg-FT', 'DP-Synth-GBDT', 'DP-Proto-TFM', 'Fed-HybridPFN', 'Central-TFM']
CITED = {'Local-GBDT': 'Local GBDT \\cite{friedman2001gbm}', 'Local-TFM': 'Local TFM \\cite{qu2026tabiclv2}',
         'DP-FedAvg': 'DP FedAvg \\cite{mcmahan2017fedavg,abadi2016dpsgd}', 'DP-FedAvg-FT': 'DP FedAvg+FT \\cite{mcmahan2017fedavg,abadi2016dpsgd}',
         'DP-Synth-GBDT': 'DP synth GBDT \\cite{friedman2001gbm}', 'DP-Proto-TFM': 'Pooled DP summary \\cite{qu2026tabiclv2}',
         'Fed-HybridPFN': 'Fed-HybridPFN (ours)', 'Central-TFM': '\\textit{Pooled raw} \\cite{qu2026tabiclv2}'}
NEW = ('Fed-HybridPFN', 'DP-Proto-TFM', 'DP-Synth-GBDT')


def load(f):
    p = os.path.join(R, f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def merged_main():
    old = [r for r in load('main.jsonl') if r['method'] not in NEW]
    return old + load('main3.jsonl')


def G(rows):
    out = {}
    for r in rows:
        out.setdefault((r['data'], r['method']), {})[r['seed']] = r
    return out


def ci95(v):
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def paired(g, d, a, b, met='auc'):
    A, B = g.get((d, a), {}), g.get((d, b), {})
    s = sorted(set(A) & set(B))
    if len(s) < 3:
        return None
    x = np.array([A[i][met] - B[i][met] for i in s])
    p = stats.ttest_rel([A[i][met] for i in s], [B[i][met] for i in s]).pvalue if np.any(x != 0) else 1.0
    return dict(diff=float(x.mean()), ci=ci95(x), p=float(p), wins=int((x > 0).sum()), n=len(s))


def f3(v):
    return ('%.3f' % v)[1:] if abs(v) < 1 else '%.2f' % v


def main_table():
    rows = merged_main(); g = G(rows)
    mets = [('auc', 1), ('auprc', 1), ('nll', -1), ('ece', -1)]
    summ, L, sds = {}, [], []
    for m in ORDER:
        cells = []
        for d in DS:
            for met, sgn in mets:
                v = [r[met] for r in g.get((d, m), {}).values()]
                mu = float(np.mean(v)); summ.setdefault(d, {}).setdefault(m, {})[met] = (mu, float(np.std(v)), len(v))
                if met == 'auc':
                    sds.append(float(np.std(v)))
                vals = {o: np.mean([r[met] for r in g[(d, o)].values()]) for o in ORDER if o != 'Central-TFM'}
                best = max(vals.values()) if sgn > 0 else min(vals.values())
                c = f3(mu)
                if m != 'Central-TFM' and abs(mu - best) < 1e-12:
                    c = '\\textbf{%s}' % c
                cells.append(c)
        if m == 'Central-TFM':
            L.append('\\midrule')
        L.append('%s & %s\\\\' % (CITED[m], ' & '.join(cells)))
    hdr = ' & '.join('\\multicolumn{4}{c}{%s}' % DN[d] for d in DS)
    cmid = ' '.join('\\cmidrule(lr){%d-%d}' % (2 + 4 * i, 5 + 4 * i) for i in range(4))
    tab = ('\\begin{table*}[!t]\n\\caption{Main Results at $\\varepsilon=1$ with 64 Target Labels, Means over Ten Seeds (Bold: Best Private Method)}\n'
           '\\label{tab:main}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{l*{16}{c}}\n\\toprule\n & %s\\\\\n%s\n'
           'Method & %s\\\\\n\\midrule\n' % (hdr, cmid, ' & '.join(['AUROC & AUPRC & NLL & ECE'] * 4)) + '\n'.join(L) +
           '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n')
    open(os.path.join(P, 'table_main.tex'), 'w').write(tab)
    tests = {d: {m: paired(g, d, 'Fed-HybridPFN', m) for m in ORDER if m != 'Fed-HybridPFN'} for d in DS}
    tests_nll = {d: {m: paired(g, d, 'Fed-HybridPFN', m, 'nll') for m in ORDER if m != 'Fed-HybridPFN'} for d in DS}
    ctx = {d: float(np.mean([np.mean([c for c in r['ctx'] if c is not None]) for r in g[(d, 'Fed-HybridPFN')].values()])) for d in DS}
    comm = {d: float(np.mean([r['comm'] for r in g[(d, 'Fed-HybridPFN')].values()])) for d in DS}
    old = G(load('main.jsonl'))
    shift_v2 = {d: float(np.mean([g[(d, 'Fed-HybridPFN')][s]['auc'] - old[(d, 'Fed-HybridPFN')][s]['auc']
                                  for s in g[(d, 'Fed-HybridPFN')] if s in old[(d, 'Fed-HybridPFN')]])) for d in DS}
    out = dict(summary=summ, tests=tests, tests_nll=tests_nll, ctx=ctx, comm=comm, sd_range=(min(sds), max(sds)), change_vs_v2=shift_v2)
    json.dump(out, open(os.path.join(R, 'main3_summary.json'), 'w'), indent=1)
    return out


def fair_table():
    g = G(merged_main())
    meths = ['Local-TFM', 'DP-FedAvg', 'DP-Proto-TFM', 'Fed-HybridPFN', 'Central-TFM']
    nm = dict(NAMES); nm['Central-TFM'] = 'Pooled raw'; nm['DP-Proto-TFM'] = 'Pooled DP'
    L = []
    for m in meths:
        c = [f3(float(np.nanmean([r[k] for r in g[(d, m)].values()]))) for d in DS for k in ('wg_auc', 'dp_gap', 'eo_gap')]
        L.append('%s & %s\\\\' % (nm[m], ' & '.join(c)))
    hdr = ' & '.join('\\multicolumn{3}{c}{%s}' % DN[d] for d in DS)
    cmid = ' '.join('\\cmidrule(lr){%d-%d}' % (2 + 3 * i, 4 + 3 * i) for i in range(4))
    tab = ('\\begin{table}[!t]\n\\caption{Group Metrics over Ten Seeds: Worst-Group AUROC (WG) and Parity Gaps (DP, EO)}\n'
           '\\label{tab:fair}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{1.6pt}\n\\begin{tabular}{l*{12}{c}}\n\\toprule\n & %s\\\\\n%s\n'
           'Method & %s\\\\\n\\midrule\n' % (hdr, cmid, ' & '.join(['WG & DP & EO'] * 4)) + '\n'.join(L) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    open(os.path.join(P, 'table_fair.tex'), 'w').write(tab)


def _variant_table(rows, full_rows, V, dsets, nseeds, label, caption, mets=('auc', 'nll'), extra_cols=None):
    g = G(rows); gf = G(full_rows)
    res, L = {}, []
    for k, lab in V:
        cells = []
        for d in dsets:
            src = gf if k == 'Fed-HybridPFN' else g
            A = {s: r for s, r in src.get((d, k), {}).items() if s < nseeds[d]}
            F = {s: r for s, r in gf.get((d, 'Fed-HybridPFN'), {}).items() if s < nseeds[d]}
            for met in mets:
                if not A:
                    cells.append('--'); continue
                mu = float(np.mean([r[met] for r in A.values()]))
                res.setdefault(k, {}).setdefault(d, {})[met] = mu
                c = '%.3f' % mu
                if k != 'Fed-HybridPFN':
                    s = sorted(set(A) & set(F))
                    if len(s) >= 3:
                        x = [A[i][met] - F[i][met] for i in s]
                        p = stats.ttest_rel([A[i][met] for i in s], [F[i][met] for i in s]).pvalue if np.any(np.array(x) != 0) else 1
                        res[k][d][met + '_p'] = float(p); res[k][d][met + '_d'] = float(np.mean(x))
                        if p < 0.05:
                            c += '$^{*}$'
                cells.append(c)
        L.append('%s & %s\\\\' % (lab, ' & '.join(cells)))
    hdr = ' & '.join('\\multicolumn{%d}{c}{%s}' % (len(mets), DN[d]) for d in dsets)
    cmid = ' '.join('\\cmidrule(lr){%d-%d}' % (2 + len(mets) * i, 1 + len(mets) * (i + 1)) for i in range(len(dsets)))
    mh = {'auc': 'AUROC', 'nll': 'NLL', 'proto_mse': 'MSE'}
    tab = ('\\begin{table}[!t]\n\\caption{%s}\n\\label{%s}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n'
           '\\begin{tabular}{l%s}\n\\toprule\n & %s\\\\\n%s\nVariant & %s\\\\\n\\midrule\n' %
           (caption, label, 'c' * (len(dsets) * len(mets)), hdr, cmid, ' & '.join([' & '.join(mh[m] for m in mets)] * len(dsets)))
           + '\n'.join(L) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    return tab, res


def ablation():
    rows = load('abl3.jsonl')
    for r in rows:
        c = r['cfg']
        if r['method'] == 'Fed-HybridPFN' and c.get('mix') == 0:
            r['method'] = 'no hedge'
        elif r['method'] == 'Fed-HybridPFN' and c.get('small_T') == 0:
            r['method'] = 'no shortcut'
    V = [('Fed-HybridPFN', 'Fed-HybridPFN (full)'), ('no hedge', 'without local hedge'), ('no shortcut', 'without small-class shortcut'),
         ('w/o SoftJoin', 'pooled mean fill for SoftJoin'), ('balanced OT', 'balanced OT ($\\tau=\\infty$)'),
         ('full context', 'full context ($S_t=1500$)'), ('E=1', 'one context ($E=1$)'), ('no posterior', 'no posterior resampling'),
         ('no local labels', 'no local labels in context')]
    tab, res = _variant_table(rows, load('main3.jsonl'), V, ['taiwan', 'gmsc', 'income'], dict(taiwan=5, gmsc=5, income=3), 'tab:abl',
                              'Ablations at $\\varepsilon=1$ (Five Seeds; Three on ACSIncome); $^{*}$: Paired $t$ Test against the Full Method, $p<0.05$')
    open(os.path.join(P, 'table_abl.tex'), 'w').write(tab)
    return res


def factorial():
    rows = load('fact3.jsonl'); full = [dict(r) for r in load('main3.jsonl') if r['method'] == 'Fed-HybridPFN']
    def key(c):
        sp = tuple(c.get('split', (0.4, 0.15, 0.7, 0.15)))
        return (int(c.get('adaptM', 1)), int(c.get('shrink', 1)), int(abs(sp[2] - 0.7) < 1e-6))
    for r in rows:
        r['method'] = 'F%d%d%d' % key(r['cfg'])
    combos = [(a, sh, sp) for a in (0, 1) for sh in (0, 1) for sp in (0, 1)]
    g = G(rows); gf = G(full)
    nseeds = dict(taiwan=5, gmsc=5, income=3)
    L, res = [], {}
    for a, sh, sp in combos:
        k = 'F%d%d%d' % (a, sh, sp)
        cells = ['\\checkmark' if a else '', '\\checkmark' if sh else '', '\\checkmark' if sp else '']
        for d in ['taiwan', 'gmsc', 'income']:
            src = gf.get((d, 'Fed-HybridPFN'), {}) if (a, sh, sp) == (1, 1, 1) else g.get((d, k), {})
            A = [r for s, r in src.items() if s < nseeds[d]]
            for met in ('auc', 'proto_mse'):
                if not A:
                    cells.append('--'); continue
                mu = float(np.mean([r[met] for r in A]))
                res.setdefault(k, {}).setdefault(d, {})[met] = mu
                cells.append('%.3f' % mu if met == 'auc' else '%.4f' % mu)
        L.append(' & '.join(cells) + '\\\\')
    tab = ('\\begin{table}[!t]\n\\caption{Factorial Ablation of the Release: AUROC and MSE of Released Means (Five Seeds; Three on ACSIncome)}\n\\label{tab:fact}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{ccc*{6}{c}}\n\\toprule\n'
           'adaptive & EB & unequal & \\multicolumn{2}{c}{Taiwan} & \\multicolumn{2}{c}{GMSC} & \\multicolumn{2}{c}{ACSIncome}\\\\\n'
           '\\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}\n'
           'count & shrinkage & budget & AUROC & MSE & AUROC & MSE & AUROC & MSE\\\\\n\\midrule\n' + '\n'.join(L) +
           '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    open(os.path.join(P, 'table_fact.tex'), 'w').write(tab)
    return res


def tfm_table():
    v1 = G(load('tfm3.jsonl')); v2 = G(merged_main())
    nseeds = dict(taiwan=5, gmsc=5, income=3)
    L, res = [], {}
    for m in ['Local-TFM', 'DP-Proto-TFM', 'Fed-HybridPFN']:
        cells = []
        for src, tag in [(v1, 'v1'), (v2, 'v2')]:
            for d in ['taiwan', 'gmsc', 'income']:
                A = [r['auc'] for s, r in src.get((d, m), {}).items() if s < nseeds[d]]
                mu = float(np.mean(A)) if A else float('nan'); res.setdefault(tag, {}).setdefault(d, {})[m] = mu
                cells.append('%.3f' % mu if A else '--')
        L.append('%s & %s\\\\' % (NAMES[m], ' & '.join(cells)))
    tab = ('\\begin{table}[!t]\n\\caption{AUROC with Two Foundation Models at $\\varepsilon=1$ (Five Seeds; Three on ACSIncome)}\n'
           '\\label{tab:tfm}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2.4pt}\n\\begin{tabular}{l*{6}{c}}\n\\toprule\n'
           ' & \\multicolumn{3}{c}{TabICL} & \\multicolumn{3}{c}{TabICLv2}\\\\\n\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n'
           'Method & Taiwan & GMSC & ACSInc. & Taiwan & GMSC & ACSInc.\\\\\n\\midrule\n' + '\n'.join(L) +
           '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    open(os.path.join(P, 'table_tfm.tex'), 'w').write(tab)
    return res


def shift_table():
    d = json.load(open(os.path.join(R, 'audit_shift_income.json')))
    rules = [('unbalanced OT', 'UOT'), ('balanced OT', 'balanced'), ('nearest prototype', 'nearest'), ('pooled mean fill', 'mean fill')]
    L = []
    for st in [0.0, 1.0, 2.0, 3.0]:
        c = []
        for rk, _ in rules:
            c.append('%.4f' % np.mean([x['rmse'] for x in d if x['rule'] == rk and x['strength'] == st and str(x['eps']) == '1.0']))
        u = [x for x in d if x['rule'] == 'unbalanced OT' and x['strength'] == st and str(x['eps']) == '1.0']
        c.append('%.3f' % np.mean([x['mass_kept'] for x in u])); c.append('%.3f' % np.mean([x['marg_viol'] for x in u]))
        L.append('%g & %s\\\\' % (st, ' & '.join(c)))
    tab = ('\\begin{table}[!t]\n\\caption{Completion RMSE and UOT Coupling Statistics on ACSIncome, $\\varepsilon=1$, Five Seeds (Mass for Unit-Mass Marginals)}\n'
           '\\label{tab:shift}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{lcccccc}\n\\toprule\n'
           ' & \\multicolumn{4}{c}{completion RMSE} & \\multicolumn{2}{c}{UOT coupling}\\\\\n\\cmidrule(lr){2-5} \\cmidrule(lr){6-7}\n'
           'strength & UOT & balanced & nearest & mean fill & mass & viol.\\\\\n\\midrule\n' + '\n'.join(L) +
           '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    open(os.path.join(P, 'table_shift.tex'), 'w').write(tab)


def donors():
    rows = load('donors3.jsonl'); g = {}
    for r in rows:
        g.setdefault((r['cfg'].get('single_donor', 0), r['cfg']['K'], r['method']), []).append(r)
    out = {}
    for k, v in g.items():
        out[str(k)] = dict(auc=float(np.mean([r['auc'] for r in v])), nll=float(np.mean([r['nll'] for r in v])),
                           server_t=float(np.mean([r.get('server_t', 0) for r in v])), n_ot=float(np.mean([r.get('n_ot', 0) for r in v])),
                           comm=float(np.mean([r['comm'] for r in v])), n=len(v))
    json.dump(out, open(os.path.join(R, 'donors3_summary.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    for f in (main_table, fair_table, ablation, factorial, tfm_table, shift_table, donors):
        try:
            o = f(); print('ok', f.__name__)
        except Exception as e:
            print('skip', f.__name__, repr(e)[:150])
