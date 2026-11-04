"""Version 6 quick suites: temperature scaling (calib6), GMSC seeds 10-19 (gmsc20), pruned matching (prune6)."""
import sys, json, os
import run, data
import experiments  # noqa: F401  (registers ablation variants)


def jobs(suite):
    J = []
    if suite == 'calib6':
        for sd in range(10):
            for d in ['taiwan', 'gmsc']:
                J.append((d, sd, dict(tscale=1), ['Fed-HybridPFN'], 'calib6'))
    elif suite == 'gmsc20':
        for sd in range(10, 20):
            J.append(('gmsc', sd, {}, ['Fed-HybridPFN', 'Local-TFM', 'DP-Proto-TFM'], 'gmsc20'))
    elif suite == 'prune6':
        for sd in range(3):
            for K, nt in [(10, 4), (20, 8), (40, 8)]:
                J.append(('income', sd, dict(K=K, n_targets=nt, topk=3), ['Fed-HybridPFN'], 'prune6'))
    return J


if __name__ == '__main__':
    suite, w, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cache = {}
    for i, (d, sd, ov, meths, out) in enumerate(jobs(suite)):
        if i % nw != w:
            continue
        f = '../results/%s.jsonl' % out
        done = set()
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if r['data'] == d and r['seed'] == sd and all(r['cfg'].get(k) == v for k, v in ov.items()):
                    done.add(r['method'])
        meths = [m for m in meths if m not in done]
        if not meths:
            continue
        if d not in cache:
            cache = {d: data.load(d)}
        run.run_one(cache[d], run.cfg_for(d, **ov), sd, meths, log=f)
