"""One JSON per figure panel (matlab_fig_1 format) from the version 2 logs. Values are seed
means exactly as logged, on the seeds stated in each caption; nothing is smoothed."""
import json, os
import numpy as np

R = os.path.join(os.path.dirname(__file__), '..', 'results')
OUT = os.path.join(os.path.dirname(__file__), '..', 'matlab', 'data')
os.makedirs(OUT, exist_ok=True)
INF = float('inf')
BLUE, VERM, GREEN, PURP, PINK, AMBER, SKY, GREY = ([0.000, 0.447, 0.698], [0.835, 0.369, 0.000], [0.000, 0.620, 0.451],
                                                  [0.482, 0.196, 0.580], [0.800, 0.475, 0.655], [0.902, 0.624, 0.000],
                                                  [0.337, 0.706, 0.914], [0.350, 0.350, 0.350])
STY = {'Fed-HybridPFN': dict(color=BLUE, marker='o', line='-', proposed=True),
       'full context': dict(color=GREEN, marker='d', line='-.'),
       'DP-Proto-TFM': dict(color=VERM, marker='s', line='--'),
       'DP-FedAvg': dict(color=GREEN, marker='d', line='-.'),
       'DP-FedAvg-FT': dict(color=PURP, marker='^', line=':'),
       'Local-TFM': dict(color=PINK, marker='v', line='-')}
LAB = {'Fed-HybridPFN': 'Fed-HybridPFN', 'full context': 'full context', 'DP-Proto-TFM': 'Pooled DP summary',
       'DP-FedAvg': 'DP FedAvg', 'DP-FedAvg-FT': 'DP FedAvg+FT', 'Local-TFM': 'Local TFM'}
S5 = set(range(5))
S3 = set(range(3))


def load(f):
    p = os.path.join(R, f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def mean_by(rows, key, data, method, metric='auc', seeds=S3):
    d = {}
    for r in rows:
        if r['data'] == data and r['method'] == method and r['seed'] in seeds:
            d.setdefault(r['cfg'][key], []).append(r[metric])
    ks = sorted(d)
    return ks, [float(np.mean(d[k])) for k in ks]


def main_mean(data, method, metric='auc', seeds=S3):
    v = [r[metric] for r in load('main.jsonl') if r['data'] == data and r['method'] == method and r['seed'] in seeds]
    return float(np.mean(v)) if v else None


def ser(name, x, y, label=None, **kw):
    s = dict(STY[name]); s.update(kw); s.update(label=LAB[name] if label is None else label, x=list(x), y=list(y))
    return s


def hline(y, label, color=GREY, line=':'):
    return dict(kind='hline', y=y, line=line, label=label, color=color)


def write(panel):
    """Reference lines carry no legend entry; their name is written on the plot."""
    ann = list(panel.get('annotations', []))
    xr = panel.get('xlim')
    for s in panel['series']:
        if s.get('kind') == 'hline' and s.get('label') and xr and not s.pop('in_legend', False):
            lg = panel.get('xscale') == 'log'
            pos = lambda f: xr[0] * (xr[1] / xr[0]) ** f if lg else xr[0] + f * (xr[1] - xr[0])
            dy = panel.get('ann_dy', 0.006)
            right = s['label'].startswith('Local')
            ann.append(dict(x=pos(0.97 if right else 0.03), y=s['y'] + (-dy if right else dy), text=s['label'],
                            italic=True, halign='right' if right else 'left'))
            s['label'] = ''
    panel['annotations'] = ann
    panel.setdefault('margins_in', [0.56, 0.42, 0.10, 0.08])
    json.dump(panel, open(os.path.join(OUT, panel['name'] + '.json'), 'w'), indent=1)


EPSX = lambda e: 32.0 if not np.isfinite(e) else e


def fig_eps():
    rows = load('eps.jsonl') + [r for r in load('main.jsonl') if r['data'] == 'income']
    S = []
    for m in ['Fed-HybridPFN', 'DP-Proto-TFM', 'DP-FedAvg']:
        ks, mu = mean_by(rows, 'eps', 'income', m)
        S.append(ser(m, [EPSX(k) for k in ks], mu))
    S += [hline(main_mean('income', 'Local-TFM'), 'Local TFM', PINK)]
    write(dict(name='fig_eps_income', layout='trio', xlabel='Privacy budget \\epsilon per agency', ylabel='AUROC', xscale='log',
               xticks=[0.5, 1, 2, 4, 8, 32], xticklabels=['0.5', '1', '2', '4', '8', '\\infty'], xlim=[0.42, 38],
               ylim=[0.74, 0.86], lock_ylim=True, legend='southeast', ann_dy=0.004, series=S))
    for d in ['taiwan', 'gmsc']:
        rows = load('epsc3.jsonl') + load('main3.jsonl')
        S = []
        for m in ['Fed-HybridPFN', 'DP-Proto-TFM']:
            ks, mu = mean_by(rows, 'eps', d, m)
            if ks:
                S.append(ser(m, [EPSX(k) for k in ks], mu))
        S += [hline(main_mean(d, 'Local-TFM'), 'Local TFM', PINK)]
        write(dict(name='fig_eps_%s' % d, layout='trio', xlabel='Privacy budget \\epsilon per agency', ylabel='AUROC', xscale='log',
                   xticks=[1, 2, 4, 8, 16, 32], xticklabels=['1', '2', '4', '8', '16', '\\infty'], xlim=[0.85, 38],
                   ylim={'taiwan': [0.50, 0.70], 'gmsc': [0.52, 0.80]}[d], lock_ylim=True, legend='southeast', series=S))


def fig_nlab():
    rows = load('nlab.jsonl') + load('main.jsonl')
    S = []
    for m in ['Fed-HybridPFN', 'DP-Proto-TFM', 'DP-FedAvg-FT', 'Local-TFM']:
        ks, mu = mean_by(rows, 'n_lab', 'income', m)
        S.append(ser(m, ks, mu))
    write(dict(name='fig_nlab', layout='trio', xlabel='Labelled records at the target', ylabel='AUROC', xscale='log',
               xticks=[16, 32, 64, 128, 256], xticklabels=['16', '32', '64', '128', '256'], xlim=[13, 310], ylim=[0.45, 0.88], lock_ylim=True, legend='southeast', series=S))


def fig_kappa():
    rows = load('kappa3.jsonl') + load('main3.jsonl') + [r for r in load('kappa.jsonl') + load('main.jsonl') if r['data'] == 'income' and r['method'] == 'Fed-HybridPFN']
    full = load('abl3.jsonl') + [r for r in load('ablation.jsonl') if r['data'] == 'income']
    S = []
    sty = {'income': dict(color=BLUE, marker='o'), 'taiwan': dict(color=VERM, marker='s'), 'gmsc': dict(color=GREEN, marker='d')}
    nm = {'income': 'ACSIncome', 'taiwan': 'Taiwan', 'gmsc': 'GMSC'}
    for d in ['income', 'taiwan', 'gmsc']:
        src = [r for r in rows if (r['data'] != 'income') or ('mix' not in r['cfg'])]   # census curve: v2 core runs
        ks, mu = mean_by(src, 'kappa', d, 'Fed-HybridPFN')
        fc = [r['auc'] for r in full if r['data'] == d and r['method'] == 'full context' and r['seed'] in S3
              and ((d != 'income') or ('mix' not in r['cfg']))]
        loc = main_mean(d, 'Local-TFM')
        x = list(ks) + ([512] if fc else []); y = list(mu) + ([float(np.mean(fc))] if fc else [])
        s_ = dict(sty[d]); s_.update(label=nm[d], x=x, y=[v - loc for v in y])
        S.append(s_)
    S.append(dict(kind='hline', y=0.0, label='', color=GREY))
    write(dict(name='fig_kappa', layout='quad', xlabel='Rows per prototype \\kappa', ylabel='Gain over Local TFM',
               xscale='log', xticks=[1, 2, 4, 8, 16, 512], xticklabels=['1', '2', '4', '8', '16', 'full'], xlim=[0.85, 600], margins_in=[0.66, 0.42, 0.08, 0.08],
               lock_ylim=False, legend='southwest', series=S))


def fig_ens():
    rows = load('ens.jsonl') + load('main.jsonl')
    S = []
    for d, sty in [('income', dict(color=BLUE, marker='o', line='-')), ('taiwan', dict(color=VERM, marker='s', line='--'))]:
        for met, ls in [('nll', None)]:
            ks, mu = mean_by(rows, 'ens', d, 'Fed-HybridPFN', 'nll', S5)
            base = mu[0]
            s = dict(sty); s.update(label='NLL, %s' % ('ACSIncome' if d == 'income' else 'Taiwan'), x=ks, y=[100 * (v - base) / base for v in mu])
            S.append(s)
            ks, mu = mean_by(rows, 'ens', d, 'Fed-HybridPFN', 'brier', S5)
            base = mu[0]
            s = dict(sty); s.update(label='Brier, %s' % ('ACSIncome' if d == 'income' else 'Taiwan'), x=ks,
                                    y=[100 * (v - base) / base for v in mu], open=True, line=':')
            S.append(s)
    write(dict(name='fig_ens', layout='quad', xlabel='Number of contexts E', ylabel='Change from E = 1 (%)', xscale='log',
               xticks=[1, 2, 4, 8], xticklabels=['1', '2', '4', '8'], xlim=[0.9, 9], ylim=[-11, 1], lock_ylim=True, legend='southwest', margins_in=[0.6, 0.42, 0.08, 0.08], series=S))


def fig_shrink():
    d = json.load(open(os.path.join(R, 'audit_shrink_income.json')))
    base = {x['n']: x['mse'] for x in d if x['mode'] == 'none'}
    sty = {'pooled': ('full pooling', VERM, 's', '--'), 'js': ('James-Stein', GREEN, 'd', '-.'),
           'robust': ('robust empirical Bayes', PURP, '^', ':'), 'eb': ('empirical Bayes (ours)', BLUE, 'o', '-')}
    S = []
    for m in ['pooled', 'js', 'robust', 'eb']:
        pts = sorted((x['n'], x['mse'] / base[x['n']]) for x in d if x['mode'] == m)
        lab, c, mk, ls = sty[m]
        S.append(dict(label=lab, x=[p[0] for p in pts], y=[p[1] for p in pts], color=c, marker=mk, line=ls, proposed=(m == 'eb')))
    S.append(dict(hline(1.0, 'raw DP means (ratio 1)', color=[0, 0, 0]), in_legend=True))
    write(dict(name='fig_shrink', layout='trio', xlabel='Records in the class', ylabel='MSE relative to raw DP means', xscale='log',
               xticks=[50, 100, 300, 1000, 4000], xticklabels=['50', '100', '300', '1000', '4000'], xlim=[45, 4500], yscale='log', yticks=[0.5, 1, 2, 4, 8, 16, 32], yticklabels=['0.5', '1', '2', '4', '8', '16', '32'], ylim=[0.45, 60], legend='northwest', lock_ylim=True, ann_dy=0.0, series=S))


def fig_mrule():
    d = json.load(open(os.path.join(R, 'audit_mrule_income.json')))
    orc = {(x['eps'], x['n']): x['err'] for x in d if x['rule'] == 'oracle'}
    sty = {'fixed (M=32)': ('fixed M = 32', VERM, 's', '--'), 'dimension adaptive': ('dimension-adaptive', GREEN, 'd', '-.'),
           'linear (ours)': ('linear rule (ours)', BLUE, 'o', '-')}
    S = []
    for k in ['fixed (M=32)', 'dimension adaptive', 'linear (ours)']:
        pts = sorted((x['n'], 100 * (x['err'] / orc[(x['eps'], x['n'])] - 1)) for x in d if x['rule'] == k and x['eps'] == 1.0)
        lab, c, mk, ls = sty[k]
        S.append(dict(label=lab, x=[p[0] for p in pts], y=[p[1] for p in pts], color=c, marker=mk, line=ls, proposed=(k == 'linear (ours)')))
    write(dict(name='fig_mrule', layout='trio', xlabel='Records in the class', ylabel='Release error above oracle M (%)', xscale='log',
               xticks=[300, 1000, 3000, 10000, 20000], xticklabels=['300', '1k', '3k', '10k', '20k'], xlim=[270, 22000], ylim=[-1, 17], lock_ylim=True, legend='north', series=S))


def fig_msweep():
    rows = load('msweep.jsonl')
    S = []
    for d, c, mk in [('income', BLUE, 'o'), ('taiwan', VERM, 's')]:
        for e, ls in [(0.5, '--'), (4.0, '-')]:
            sub = [r for r in rows if r['cfg']['eps'] == e]
            ks, mu = mean_by(sub, 'M', d, 'Ours E1')
            if not ks:
                continue
            base = max(mu)
            S.append(dict(label='%s, \\epsilon = %g' % ('ACSIncome' if d == 'income' else 'Taiwan', e), x=[k // 2 for k in ks],
                          y=[v - base for v in mu], color=c, marker=mk, line=ls, open=(e == 0.5)))
    write(dict(name='fig_msweep', layout='trio', xlabel='Prototypes per class M', ylabel='AUROC minus its maximum', xscale='log',
               xticks=[2, 4, 8, 16, 32, 64], xticklabels=['2', '4', '8', '16', '32', '64'], xlim=[1.7, 76], series=S))


def fig_shift():
    d = json.load(open(os.path.join(R, 'audit_shift_income.json')))
    def agg(rule, eps, key):
        g = {}
        for x in d:
            if x['rule'] == rule and str(x['eps']) == eps:
                g.setdefault(x['strength'], []).append(x[key])
        ks = sorted(g)
        return ks, [float(np.mean(g[k])) for k in ks]
    sty = {'unbalanced OT': ('unbalanced OT (ours)', BLUE, 'o', '-', True), 'balanced OT': ('balanced OT', VERM, 's', '--', False),
           'nearest prototype': ('nearest prototype', GREEN, 'd', '-.', False)}
    for eps, nm in [('1.0', 'fig_shift_dp'), ('inf', 'fig_shift_nodp')]:
        kb, base = agg('pooled mean fill', eps, 'rmse')
        S = []
        for k, (lab, c, mk, ls, pr) in sty.items():
            ks, mu = agg(k, eps, 'rmse')
            S.append(dict(label=lab, x=ks, y=[m / b for m, b in zip(mu, base)], color=c, marker=mk, line=ls, proposed=pr))
        S.append(hline(1.0, 'pooled mean fill'))
        write(dict(name=nm, layout='trio', xlabel='Selection strength between agencies', ylabel='Completion RMSE / mean fill',
                   xticks=[0, 1, 2, 3], xlim=[-0.15, 3.15], ylim=[0.5, 2.1], lock_ylim=True, legend='north', ann_dy=0.04, series=S))
    S = []
    for eps, lab, c, mk, ls in [('1.0', '\\epsilon = 1', BLUE, 'o', '-'), ('inf', 'no DP', VERM, 's', '--')]:
        ks, mu = agg('unbalanced OT', eps, 'marg_viol')
        S.append(dict(label='marginal violation, ' + lab, x=ks, y=mu, color=c, marker=mk, line=ls))
    write(dict(name='fig_shift_mass', layout='trio', xlabel='Selection strength between agencies', ylabel='Relative marginal violation',
               xticks=[0, 1, 2, 3], xlim=[-0.15, 3.15], ylim=[0, 0.42], lock_ylim=True, legend='north', series=S))


def fig_cmi():
    S = []
    allv = []
    for f, lab, c, mk in [('audit_cmi_income.json', 'ACSIncome', BLUE, 'o'), ('audit_cmi_employment.json', 'ACSEmployment', VERM, 's')]:
        d = json.load(open(os.path.join(R, f)))
        x = [max(v['I_S'], 1e-4) for v in d]; y = [max(v['I_SY'], 1e-4) for v in d]
        allv += x + y
        order = np.argsort(x)
        S.append(dict(label=lab, x=[x[i] for i in order], y=[y[i] for i in order], color=c, marker=mk, line='none'))
    lo, hi = 1e-4, max(allv) * 1.5
    S.append(dict(kind='ref', x=[lo, hi], y=[lo, hi]))
    write(dict(name='fig_cmi', layout='trio', xlabel='{\\itI}({\\itU};{\\itV} | {\\itS})', ylabel='{\\itI}({\\itU};{\\itV} | {\\itS},{\\itY})',
               xscale='log', yscale='log', xlim=[lo, hi], ylim=[lo, hi], series=S,
               annotations=[dict(x=lo * 3, y=hi / 3, text='class matching raises dependence', italic=True, halign='left'),
                            dict(x=hi / 2.5, y=lo * 2, text='lowers', italic=True, halign='right')]))


def fig_phase():
    rows = load('phase.jsonl')
    caps, epss = [2000, 6000, 30000], [0.5, 1.0, 4.0, 16.0]
    Z, T = [], []
    for c in caps:
        zr, tr = [], []
        for e in epss:
            m = lambda meth: float(np.mean([r['auc'] for r in rows if r['method'] == meth and r['cfg']['cap'] == c and r['cfg']['eps'] == e]))
            ours, loc, pool = m('Ours E1'), m('Local-TFM'), m('DP-Proto-TFM')
            zr.append(ours - loc)
            tr.append(('%+.3f' % (ours - loc)).replace('0.', '.'))
        Z.append(zr); T.append(tr)
    write(dict(name='fig_phase', layout='trio', xlabel='Privacy budget \\epsilon per agency', ylabel='Records per agency',
               xticks=[1, 2, 3, 4], xticklabels=['0.5', '1', '4', '16'], yticks=[1, 2, 3], yticklabels=['2k', '6k', '30k'],
               xlim=[0.5, 4.5], ylim=[0.5, 3.5], margins_in=[0.56, 0.42, 0.74, 0.08],
               series=[dict(kind='image', x=[1, 2, 3, 4], y=[1, 2, 3], z=Z, clim=[-0.06, 0.06], cblabel='gain over Local TFM', txt=T)]))


def fig_scale():
    rows = load('scale.jsonl') + load('main.jsonl')
    S = []
    for m in ['Fed-HybridPFN', 'DP-Proto-TFM', 'DP-FedAvg']:
        ks, mu = mean_by(rows, 'K', 'income', m)
        S.append(ser(m, ks, mu))
    write(dict(name='fig_scale', layout='trio', xlabel='Number of agencies K', ylabel='AUROC', xticks=[20, 50, 100],
               xlim=[15, 105], series=S))


def fig_tau():
    d = json.load(open(os.path.join(R, 'audit_tau_income.json')))
    TX = {'0.1': 0.1, '0.25': 0.25, '0.5': 0.5, '1.0': 1.0, '2.0': 2.0, '5.0': 5.0, 'inf': 20.0}
    sty = {(0.5, 0.0): ('overlap 0.5, strength 0', BLUE, 'o', True), (0.5, 3.0): ('overlap 0.5, strength 3', SKY, 's', False),
           (0.1, 0.0): ('overlap 0.1, strength 0', VERM, 'd', False), (0.1, 3.0): ('overlap 0.1, strength 3', AMBER, '^', False)}
    S = []
    for (ov, st), (lab, c, mk, pr) in sty.items():
        xs, ys = [], []
        for t in ['0.1', '0.25', '0.5', '1.0', '2.0', '5.0', 'inf']:
            v = [x['rmse'] for x in d if x['overlap'] == ov and x['strength'] == st and x['tau'] == t]
            if v:
                xs.append(TX[t]); ys.append(float(np.mean(v)))
        if xs:
            S.append(dict(label=lab, x=xs, y=ys, color=c, marker=mk, proposed=False))
    write(dict(name='fig_tau', layout='trio', xlabel='Marginal penalty \\tau', ylabel='Completion RMSE', xscale='log',
               xticks=[0.1, 0.25, 0.5, 1, 2, 5, 20], xticklabels=['0.1', '.25', '0.5', '1', '2', '5', '\\infty'], xlim=[0.085, 24],
               legend='auto', series=S))


def fig_donors():
    rows = load('donors3.jsonl') + [r for r in load('main3.jsonl') + load('main.jsonl') if r['data'] == 'income' and r['seed'] in S3]
    S = []
    for m in ['Fed-HybridPFN', 'DP-Proto-TFM']:
        xs, ys = [], []
        for nd in [1, 3, 6, 12]:
            if nd == 12:
                src = load('main3.jsonl') if m != 'Central-TFM' else load('main.jsonl')
                v = [r['auc'] for r in src if r['data'] == 'income' and r['method'] == m and r['seed'] in S3]
            else:
                v = [r['auc'] for r in rows if r['method'] == m and r['cfg'].get('single_donor') == nd and r['seed'] in S3]
            if v:
                xs.append(nd); ys.append(float(np.mean(v)))
        if m == 'Central-TFM':
            S.append(dict(label='pooled raw, no DP', x=xs, y=ys, color=GREY, marker='v'))
        else:
            S.append(ser(m, xs, ys))
    S.append(hline(main_mean('income', 'Local-TFM'), 'Local TFM', PINK))
    write(dict(name='fig_donors', layout='trio', xlabel='Label-rich agencies in the federation', ylabel='AUROC', xscale='log',
               xticks=[1, 3, 6, 12], xticklabels=['1', '3', '6', '12'], xlim=[0.85, 14], legend='southeast', series=S))


if __name__ == '__main__':
    for f in [fig_eps, fig_nlab, fig_kappa, fig_ens, fig_shrink, fig_mrule, fig_phase, fig_tau, fig_donors]:
        try:
            f(); print('ok', f.__name__)
        except Exception as ex:
            print('skip', f.__name__, repr(ex)[:120])
