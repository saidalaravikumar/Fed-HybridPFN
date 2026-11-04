"""Tables and statistics for version 2. Seed level paired tests: the unit of replication is a
seed (one independently drawn federation); target silos inside a seed are averaged first."""
import json, os
import numpy as np
from scipy import stats

R = os.path.join(os.path.dirname(__file__), '..', 'results')
P = os.path.join(os.path.dirname(__file__), '..', 'paper')
INF = float('inf')
NAMES = {'Local-GBDT': 'Local GBDT', 'Local-TFM': 'Local TFM', 'DP-FedAvg': 'DP FedAvg',
         'DP-FedAvg-FT': 'DP FedAvg+FT', 'DP-Proto-TFM': 'Pooled DP summary', 'DP-Synth-GBDT': 'DP synth GBDT',
         'Central-TFM': 'Pooled raw (no DP)', 'Fed-HybridPFN': 'Fed-HybridPFN'}
DS = ['income', 'employment', 'taiwan', 'gmsc']
DN = {'income': 'ACSIncome', 'employment': 'ACSEmployment', 'taiwan': 'Taiwan', 'gmsc': 'GMSC'}
ORDER = ['Local-GBDT', 'Local-TFM', 'DP-FedAvg', 'DP-FedAvg-FT', 'DP-Synth-GBDT', 'DP-Proto-TFM', 'Fed-HybridPFN', 'Central-TFM']


def load(f):
    p = os.path.join(R, f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def by(rows, *keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] if k in r else r['cfg'].get(k) for k in keys), []).append(r)
    return out


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def paired(rows, d, a, b, metric='auc'):
    A = {r['seed']: r[metric] for r in rows if r['data'] == d and r['method'] == a}
    B = {r['seed']: r[metric] for r in rows if r['data'] == d and r['method'] == b}
    s = sorted(set(A) & set(B))
    if len(s) < 3:
        return None
    x = np.array([A[i] - B[i] for i in s])
    p = stats.ttest_rel([A[i] for i in s], [B[i] for i in s]).pvalue if np.any(x != 0) else 1.0
    return dict(diff=float(x.mean()), ci=ci95(x), p=float(p), wins=int((x > 0).sum()), n=len(s))


def main_table():
    rows = load('main.jsonl')
    G = by(rows, 'data', 'method')
    summ, lines = {}, []
    metrics = [('auc', 'AUROC', 1), ('auprc', 'AUPRC', 1), ('nll', 'NLL', -1), ('ece', 'ECE', -1)]
    sds = []
    for m in ORDER:
        cells = []
        for d in DS:
            for met, lab, sgn in metrics:
                v = [r[met] for r in G.get((d, m), [])]
                mu, sd = float(np.mean(v)), float(np.std(v))
                summ.setdefault(d, {}).setdefault(m, {})[met] = (mu, sd, len(v))
                if met == 'auc':
                    sds.append(sd)
                priv = [o for o in ORDER if o != 'Central-TFM' and G.get((d, o))]
                vals = {o: np.mean([r[met] for r in G[(d, o)]]) for o in priv}
                best = max(vals.values()) if sgn > 0 else min(vals.values())
                c = ('%.3f' % mu)[1:] if mu < 1 else '%.2f' % mu
                if m != 'Central-TFM' and abs(mu - best) < 1e-12:
                    c = '\\textbf{%s}' % c
                cells.append(c)
        nm = NAMES[m] if m != 'Central-TFM' else '\\textit{Pooled raw}'
        if m == 'Central-TFM':
            lines.append('\\midrule')
        lines.append('%s & %s\\\\' % (nm, ' & '.join(cells)))
    hdr = ' & '.join('\\multicolumn{4}{c}{%s}' % DN[d] for d in DS)
    cmid = ' '.join('\\cmidrule(lr){%d-%d}' % (2 + 4 * i, 5 + 4 * i) for i in range(4))
    sub = ' & '.join(['AUROC & AUPRC & NLL & ECE'] * 4)
    tests = {d: {m: paired(rows, d, 'Fed-HybridPFN', m) for m in ORDER if m not in ('Fed-HybridPFN',)} for d in DS}
    tests_nll = {d: {m: paired(rows, d, 'Fed-HybridPFN', m, 'nll') for m in ORDER if m != 'Fed-HybridPFN'} for d in DS}
    fair = {d: {m: {k: float(np.nanmean([r[k] for r in G.get((d, m), [])])) for k in ('dp_gap', 'eo_gap', 'gap_auc')}
                for m in ORDER if G.get((d, m))} for d in DS}
    ctx = {d: float(np.mean([np.mean([c for c in r['ctx'] if c is not None]) for r in G.get((d, 'Fed-HybridPFN'), []) if r.get('ctx')]))
           for d in DS if G.get((d, 'Fed-HybridPFN'))}
    comm = {d: float(np.mean([r['comm'] for r in G.get((d, 'Fed-HybridPFN'), [])])) for d in DS if G.get((d, 'Fed-HybridPFN'))}
    tab = ('\\begin{table*}[!t]\n\\caption{Main Results at $\\varepsilon=1$, $\\delta=10^{-5}$ per Label Rich Agency and 64 Labels per '
           'Target Agency, Means over Ten Seeds. AUROC and AUPRC: Higher Is Better; NLL and ECE: Lower Is Better. Bold: Best Private Method. '
           'Seed Standard Deviations of AUROC Range from %.3f to %.3f.}\n'
           '\\label{tab:main}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{l*{16}{c}}\n\\toprule\n & %s\\\\\n%s\n'
           'Method & %s\\\\\n\\midrule\n' % (min(sds), max(sds), hdr, cmid, sub) + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n')
    open(os.path.join(P, 'table_main.tex'), 'w').write(tab)
    out = dict(summary=summ, tests=tests, tests_nll=tests_nll, fair=fair, ctx=ctx, comm=comm)
    json.dump(out, open(os.path.join(R, 'main_summary.json'), 'w'), indent=1)
    return out


def simple_table(fname, variants, datasets, label, caption, metrics=(('auc', 'AUROC'), ('nll', 'NLL')), keyfun=None):
    rows = load(fname)
    if keyfun:
        for r in rows:
            r['method'] = keyfun(r)
    G = by(rows, 'data', 'method')
    head = ' & '.join(('%s %s' % (DN[d], ml)) if len(metrics) > 1 else DN[d] for d in datasets for _, ml in metrics)
    L = []
    res = {}
    for k, lab in variants:
        cells = []
        for d in datasets:
            for met, _ in metrics:
                v = [r[met] for r in G.get((d, k), [])]
                if v:
                    res.setdefault(k, {}).setdefault(d, {})[met] = (float(np.mean(v)), float(np.std(v)), len(v))
                cells.append('%.3f' % np.mean(v) if v else '--')
        L.append('%s & %s\\\\' % (lab, ' & '.join(cells)))
    tab = ('\\begin{table}[!t]\n\\caption{%s}\n\\label{%s}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n'
           '\\begin{tabular}{l%s}\n\\toprule\nVariant & %s\\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
           % (caption, label, 'c' * (len(datasets) * len(metrics)), head, '\n'.join(L)))
    return tab, res


def ablation():
    def key(r):
        c = r['cfg']
        if r['method'] == 'Fed-HybridPFN' and c.get('equal_split'):
            return 'equal split'
        if r['method'] == 'Fed-HybridPFN' and c.get('adaptM') == 0:
            return 'fixed M'
        return r['method']
    main = [dict(r) for r in load('main.jsonl') if r['method'] == 'Fed-HybridPFN' and r['seed'] < 5 and r['data'] in ('income', 'taiwan')]
    rows = load('ablation.jsonl') + main
    for r in rows:
        r['method'] = key(r)
    open(os.path.join(R, '_abl_tmp.jsonl'), 'w').write('\n'.join(json.dumps(r) for r in rows))
    V = [('Fed-HybridPFN', 'Fed-HybridPFN (full)'), ('w/o SoftJoin', 'pooled mean fill instead of SoftJoin'),
         ('balanced OT', 'balanced instead of unbalanced OT'), ('full context', 'full context ($S_t=1500$)'),
         ('E=1', 'one context ($E=1$)'), ('no posterior', '$E=4$ without posterior resampling'),
         ('fixed M', 'fixed $M=32$, no shrinkage'), ('equal split', 'equal budget split, fixed $M$'),
         ('no local labels', 'no local labels in the context')]
    rows = load('_abl_tmp.jsonl')
    G = by(rows, 'data', 'method')
    res, L = {}, []
    for k, lab in V:
        cells = []
        for d in ['income', 'taiwan']:
            for met in ('auc', 'nll'):
                v = {r['seed']: r[met] for r in G.get((d, k), [])}
                f = {r['seed']: r[met] for r in G.get((d, 'Fed-HybridPFN'), [])}
                mu = float(np.mean(list(v.values())))
                res.setdefault(k, {}).setdefault(d, {})[met] = mu
                c = '%.3f' % mu
                if k != 'Fed-HybridPFN':
                    sd = sorted(set(v) & set(f))
                    x = np.array([v[i] - f[i] for i in sd])
                    p = stats.ttest_rel([v[i] for i in sd], [f[i] for i in sd]).pvalue if np.any(x != 0) else 1.0
                    res[k][d][met + '_p'] = float(p)
                    if p < 0.05:
                        c += '$^{*}$'
                cells.append(c)
        L.append('%s & %s\\\\' % (lab, ' & '.join(cells)))
    tab = ('\\begin{table}[!t]\n\\caption{Ablations at $\\varepsilon=1$, Means over Five Seeds. $^{*}$: Paired $t$ Test against the Full '
           'Method over Seeds, $p<0.05$}\n\\label{tab:abl}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n'
           '\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{ACSIncome} & \\multicolumn{2}{c}{Taiwan}\\\\\n'
           '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\nVariant & AUROC & NLL & AUROC & NLL\\\\\n\\midrule\n' + '\n'.join(L) +
           '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    open(os.path.join(P, 'table_abl.tex'), 'w').write(tab)
    return res


def gates():
    main = [r for r in load('main.jsonl') if r['seed'] < 5 and r['data'] in ('income', 'taiwan', 'gmsc') and r['method'] in ('Fed-HybridPFN', 'Local-TFM')]
    abl = [r for r in load('ablation.jsonl') if r['method'] == 'full context' and r['data'] in ('income', 'taiwan', 'gmsc')]
    epc = [dict(r) for r in load('eps_credit.jsonl') if r['method'] == 'full context' and r['cfg']['eps'] == 1.0]
    rows = load('gates.jsonl') + main + abl + epc
    open(os.path.join(R, '_gate_tmp.jsonl'), 'w').write('\n'.join(json.dumps(r) for r in rows))
    V = [('full context', 'no gate (full context)'), ('CV gate', 'threefold CV gate, 64 labels'),
         ('lambda gate', 'reliability gate'), ('ARRG', 'reliability $\\times$ compatibility'),
         ('Fed-HybridPFN', 'resolution rule, $\\kappa=4$ (ours)'), ('Local-TFM', 'Local TFM'),
         ('oracle gate', 'oracle (test labels)')]
    tab, res = simple_table('_gate_tmp.jsonl', V, ['income', 'taiwan', 'gmsc'], 'tab:gates',
                            'Context Size Rules at $\\varepsilon=1$: AUROC, Mean over Five Seeds. The Oracle Picks, per Target and with One Context, the Best of 0, 500, 1000 or 1500 Synthetic Rows by Test AUROC', metrics=(('auc', 'AUROC'),))
    open(os.path.join(P, 'table_gates.tex'), 'w').write(tab)
    return res


def disjoint():
    rows = load('disjoint.jsonl')
    main = [r for r in load('main.jsonl') if r['seed'] < 5 and r['data'] in ('income', 'employment')]
    G1, G0 = by(rows, 'data', 'method'), by(main, 'data', 'method')
    out = {}
    for d in ('income', 'employment'):
        for m in ['Local-TFM', 'DP-FedAvg', 'DP-Proto-TFM', 'Fed-HybridPFN', 'Central-TFM']:
            a = [r['auc'] for r in G0.get((d, m), [])]; b = [r['auc'] for r in G1.get((d, m), [])]
            out.setdefault(d, {})[m] = (float(np.mean(a)) if a else None, float(np.mean(b)) if b else None, len(b))
    json.dump(out, open(os.path.join(R, 'disjoint_summary.json'), 'w'), indent=1)
    L = []
    for m in ['Local-TFM', 'DP-FedAvg', 'DP-Proto-TFM', 'Fed-HybridPFN', 'Central-TFM']:
        c = []
        for d in ('income', 'employment'):
            a, b, _ = out[d][m]
            c += ['%.4f' % a if a is not None else '--', '%.4f' % b if b is not None else '--']
        L.append('%s & %s\\\\' % (NAMES[m], ' & '.join(c)))
    tab = ('\\begin{table}[!t]\n\\caption{Overlapping Versus Disjoint Evaluation (AUROC, Five Seeds)}\n\\label{tab:disjoint}\n'
           '\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccc}\n\\toprule\n'
           ' & \\multicolumn{2}{c}{ACSIncome} & \\multicolumn{2}{c}{ACSEmployment}\\\\\nMethod & overlap & disjoint & overlap & disjoint\\\\\n'
           '\\midrule\n' + '\n'.join(L) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    open(os.path.join(P, 'table_disjoint.tex'), 'w').write(tab)
    return out


if __name__ == '__main__':
    o = main_table()
    for d in DS:
        print(d, {m: (round(v['diff'], 4), '%.1e' % v['p'], v['wins'], v['n']) for m, v in o['tests'][d].items() if v})
    print('ctx', o['ctx'], 'comm', o['comm'])
    for f in (ablation, gates, disjoint):
        try:
            print(f.__name__, json.dumps(f(), indent=0)[:1500])
        except Exception as e:
            print('skip', f.__name__, e)


def fair_table():
    rows = load('main.jsonl'); G = by(rows, 'data', 'method')
    meths = ['Local-TFM', 'DP-FedAvg', 'DP-Proto-TFM', 'Fed-HybridPFN', 'Central-TFM']
    nm = dict(NAMES); nm['Central-TFM'] = 'Pooled raw'; nm['DP-Proto-TFM'] = 'Pooled DP'
    L = []
    for m in meths:
        c = []
        for d in DS:
            for k in ('wg_auc', 'dp_gap', 'eo_gap'):
                v = float(np.nanmean([r[k] for r in G[(d, m)]])); c.append(('%.3f' % v)[1:])
        L.append('%s & %s\\\\' % (nm[m], ' & '.join(c)))
    hdr = ' & '.join('\\multicolumn{3}{c}{%s}' % DN[d] for d in DS)
    cmid = ' '.join('\\cmidrule(lr){%d-%d}' % (2 + 3 * i, 4 + 3 * i) for i in range(4))
    tab = ('\\begin{table}[!t]\n\\caption{Group Metrics at $\\varepsilon=1$ over Ten Seeds: Worst Group AUROC (WG), Demographic Parity Gap (DP) '
           'and Equal Opportunity Gap (EO) at Threshold 0.5}\n\\label{tab:fair}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{1.6pt}\n'
           '\\begin{tabular}{l*{12}{c}}\n\\toprule\n & %s\\\\\n%s\nMethod & %s\\\\\n\\midrule\n' % (hdr, cmid, ' & '.join(['WG & DP & EO'] * 4))
           + '\n'.join(L) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    open(os.path.join(P, 'table_fair.tex'), 'w').write(tab)


if __name__ == '__main__':
    fair_table()
