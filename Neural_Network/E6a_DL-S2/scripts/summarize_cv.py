"""Summarise a 5-fold run of train-fitness-score.py.

Reads results/<name>-f<k>-best-pred.npz (validation predictions at each fold's
best epoch) and prints per-fold correlation, mean +- sd, and the pooled
out-of-fold correlation (every image predicted once, by a model that never saw
its run).  Also writes results/<name>-oof.png, a scatter like the 2023 plot.

Usage (inside the Docker image, which has numpy/matplotlib):  python scripts/summarize_cv.py cv5
"""
import sys
import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else 'cv5'
ys, ps, rows = [], [], []
for k in range(5):
    try:
        d = np.load('results/%s-f%d-best-pred.npz' % (name, k))
    except FileNotFoundError:
        print('fold %d: missing' % k); continue
    r = np.corrcoef(d['yval'], d['pval'])[0, 1]
    mse = np.mean((d['yval'] - d['pval'])**2)
    rows.append(r); ys.append(d['yval']); ps.append(d['pval'])
    print('fold %d: best epoch %2d  val r %.4f  val MSE %.5f  (train r %.4f)' % (k, d['epoch'], r, mse, d['train_corr']))

if rows:
    y, p = np.concatenate(ys), np.concatenate(ps)
    print('mean r %.4f +- %.4f over %d folds' % (np.mean(rows), np.std(rows), len(rows)))
    print('pooled out-of-fold r %.4f  (n = %d images)' % (np.corrcoef(y, p)[0, 1], len(y)))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4), dpi=200)
    plt.scatter(y, p, s=4, alpha=0.4)
    plt.xlabel('Human Fitness Scores'); plt.ylabel('Predictions')
    plt.title('%s  out-of-fold\nCorrelation: %.4f' % (name, np.corrcoef(y, p)[0, 1]))
    plt.xlim([0, 1]); plt.ylim([0, 1]); plt.gca().set_aspect('equal')
    plt.tight_layout(); plt.savefig('results/%s-oof.png' % name)
    print('wrote results/%s-oof.png' % name)
