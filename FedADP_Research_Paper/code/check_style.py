"""Scan LaTeX prose for hyphens/dashes outside math, commands and fixed names, and for banned words."""
import re, sys, glob
BANNED = ['delve','leverage','harness','pivotal','crucial','paramount','groundbreaking','seamless','comprehensive',
          'intricate','multifaceted','landscape','realm','tapestry','paradigm','holistic','synergy','cutting edge',
          'notably','furthermore','moreover','sheds light','paves the way','testament','robust','novel']
FIXED = ['Fed-HybridPFN','TabICLv2','IEEE','zCDP']
for f in sorted((glob.glob(sys.argv[1] + '/sec_*.tex') + glob.glob(sys.argv[1] + '/main.tex'))):
    t = open(f).read()
    t = re.sub(r'\\begin\{(equation|tikzpicture|algorithmic)\}.*?\\end\{\1\}', ' ', t, flags=re.S)
    t = re.sub(r'\$\$.*?\$\$', ' ', t, flags=re.S)
    t = re.sub(r'\$.*?\$', ' ', t, flags=re.S)
    t = re.sub(r'\\(cite|ref|label|eqref|input|includegraphics|url|begin|end)\{[^}]*\}', ' ', t)
    t = re.sub(r'%.*', ' ', t)
    for w in FIXED:
        t = t.replace(w, ' ')
    for i, line in enumerate(t.split('\n')):
        for m in re.finditer(r'[A-Za-z0-9\)\]]\s*[-\u2013\u2014]+\s*[A-Za-z0-9\(]|\u2013|\u2014|--', line):
            print('%s: dash: ...%s...' % (f.split('/')[-1], line[max(0, m.start()-30):m.end()+30]))
        low = line.lower()
        for b in BANNED:
            if re.search(r'\b' + b + r'\b', low):
                print('%s: banned "%s": ...%s...' % (f.split('/')[-1], b, line[max(0, low.find(b)-40):low.find(b)+40]))
