"""Assign each SPAM run in a DL-S2 dataset to one of K folds.

Folds are grouped by run (all images of a run share one fold, so no run is in
both training and validation) and stratified by score: runs are sorted by
score, cut into consecutive blocks of K, and each block's runs are dealt
randomly to the K folds.

Usage: python make_folds.py data/dl-s2-10.json data/dl-s2-10-folds.json [-k 5] [-seed 2026]
"""
import argparse, json, random, statistics as st
from collections import Counter, defaultdict

p = argparse.ArgumentParser()
p.add_argument('info_json'); p.add_argument('out_json')
p.add_argument('-k', type=int, default=5); p.add_argument('-seed', type=int, default=2026)
a = p.parse_args()

info = json.load(open(a.info_json))['info']
score = {d['run_id']: d['score'] for d in info}
nImg = Counter(d['run_id'] for d in info)

rng = random.Random(a.seed)
runs = sorted(score, key=lambda r: (score[r], r))
fold = {}
for i in range(0, len(runs), a.k):
    block = runs[i:i + a.k]
    slots = list(range(a.k)); rng.shuffle(slots)
    for r, f in zip(block, slots): fold[r] = f

json.dump({'source': a.info_json, 'k': a.k, 'seed': a.seed,
           'method': 'grouped by run_id, stratified by score (sorted blocks of k)',
           'fold': fold}, open(a.out_json, 'w'), indent=1)

# Report balance
by = defaultdict(list)
for r, f in fold.items(): by[f].append(r)
print('fold  runs  images  mean_score  sd_score   (image-weighted mean)')
for f in range(a.k):
    rs = by[f]; imgs = sum(nImg[r] for r in rs)
    wmean = sum(score[r] * nImg[r] for r in rs) / imgs
    print('%4d %5d %7d %11.3f %9.3f   %.3f' % (f, len(rs), imgs, st.mean(score[r] for r in rs), st.pstdev(score[r] for r in rs), wmean))
