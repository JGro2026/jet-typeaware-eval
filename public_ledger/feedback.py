"""Feedback protocol (4 rounds x 25 reviewed journal entries) on the DIB ledger.
Usage: python feedback.py linear <seed> | deepsad <seed>"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from scipy.optimize import minimize
from dib_data import *
from evalx import order

DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb', 'DECOMP']
from dibpaths import SCORES as SC, RESULTS as RES
B, R = 25, 4


def det_file(d, seed):
    f = f'{SC}/{d}_s{seed}.npy'
    return f if os.path.exists(f) else f'{SC}/{d}_s0.npy'


def entry_context(df, seed):
    cols = []
    for d in DETS:
        s = entry_scores(np.load(det_file(d, seed)), df.eid.values); cols.append(pct(s))
    Xc = np.stack(cols, 1)
    return np.hstack([Xc, np.ones((len(Xc), 1))])          # intercept column


class Oracle:
    def __init__(s, y): s._y = y; s.reviewed = np.zeros(len(y), bool)
    def reveal(s, idx):
        assert not s.reviewed[idx].any(); s.reviewed[idx] = True
        return s._y[idx]


def run_loop(score_fn, y, mk):
    """score_fn(round, reviewed_idx, reviewed_labels) -> entry scores."""
    orc = Oracle(y); rev_idx = np.array([], int); rev_y = np.array([], int); found = []; comp = []
    for r in range(R):
        s = score_fn(r, rev_idx, rev_y)
        o = order(s); o = o[~orc.reviewed[o]][:B]
        lab = orc.reveal(o); rev_idx = np.r_[rev_idx, o]; rev_y = np.r_[rev_y, lab]
        found.append(int(rev_y.sum()))
    s = score_fn(R, rev_idx, rev_y)                       # final ranking after all feedback
    rem = ~orc.reviewed; o = order(np.where(rem, s, -np.inf)); o = o[rem[o]]
    out = {f'found@{25*(i+1)}': found[i] for i in range(R)}
    for k in (25, 50, 100): out[f'final P@{k}'] = float(y[o[:k]].mean())
    out['adaptive P@100'] = found[-1] / 100
    for m in range(1, 9): out[f'rev_M{m}'] = int(((mk[rev_idx] == m)).sum())
    return out


def lints(X, theta0, mu0, prior_var, seed):
    L0 = np.diag(1.0 / np.asarray(prior_var))
    def f(r, idx, yy):
        if r == 0: return X @ theta0
        Xr = X[idx]; A = L0 + Xr.T @ Xr; Ainv = np.linalg.inv(A)
        mu = Ainv @ (L0 @ mu0 + Xr.T @ yy)
        rng = np.random.default_rng([seed, r, 0x75])
        th = rng.multivariate_normal(mu, (Ainv + Ainv.T) / 2)
        return X @ th
    return f


def greedy_lr(X, theta0, mu0, pi, C=1.0):
    def f(r, idx, yy):
        if r == 0 or len(np.unique(yy)) < 2: return X @ theta0
        Xr = X[idx]; t = np.where(yy == 1, 1.0, -1.0)
        def obj(th):
            z = t * (Xr @ th); d = th - mu0
            l = np.logaddexp(0, -z).sum() + 0.5 / C * (pi * d * d).sum()
            g = -(Xr * (t * (1 / (1 + np.exp(z))))[:, None]).sum(0) + (pi * d) / C
            return l, g
        th = minimize(obj, mu0.copy(), jac=True, method='L-BFGS-B').x
        return X @ th
    return f


if __name__ == '__main__':
    mode, seed = sys.argv[1], int(sys.argv[2])
    df = load(); e = entries(df); y = e.label.values; mk = e.marking.values
    X = entry_context(df, seed); k = X.shape[1]
    theta0 = np.r_[np.full(k - 1, 1.0 / (k - 1)), 0.0]
    res = {}
    if mode == 'linear':
        res['STATIC-AVG'] = run_loop(lambda r, i, yy: X @ theta0, y, mk)
        res['LinTS'] = run_loop(lints(X, theta0, theta0, np.r_[np.full(k - 1, 0.01), 1e6], seed), y, mk)
        res['LinTS-spec'] = run_loop(lints(X, theta0, np.zeros(k), np.ones(k), seed), y, mk)
        res['GREEDY-LR'] = run_loop(greedy_lr(X, theta0, theta0, np.r_[np.ones(k - 1), 0.0]), y, mk)
        res['GREEDY-LR-spec'] = run_loop(greedy_lr(X, theta0, np.zeros(k), np.ones(k)), y, mk)
    elif mode == 'deepsad':
        import torch
        from run_dets import ae_emb, seed_everything
        t0 = time.time()
        _, m, C, Xn = ae_emb(df, seed, epochs=0, return_model=True)
        m.load_state_dict(torch.load(f'{SC}/ae_emb_s{seed}.pt'))
        state = {kk: v.clone() for kk, v in m.state_dict().items()}
        eid = df.eid.values; n = len(Xn)
        with torch.no_grad():
            z0 = torch.cat([m.encode(C[i:i+8192], Xn[i:i+8192]) for i in range(0, n, 8192)])
        c = z0.mean(0); c[(c.abs() < 0.1) & (c < 0)] = -0.1; c[(c.abs() < 0.1) & (c >= 0)] = 0.1
        ae_line = np.load(f'{SC}/AE-emb_s{seed}.npy')

        def f(r, idx, yy):
            if r == 0: return entry_scores(ae_line, eid)
            seed_everything(seed * 100 + r); m.load_state_dict(state)
            ylab = np.zeros(len(e)); ylab[idx] = np.where(yy == 1, -1.0, 1.0)
            yl = torch.tensor(ylab[eid], dtype=torch.float32)
            params = list(m.emb.parameters()) + list(m.enc.parameters())
            opt = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-6)
            gen = torch.Generator().manual_seed(seed * 100 + r)
            for ep in range(50):
                perm = torch.randperm(n, generator=gen)
                for i in range(0, n, 128):
                    b = perm[i:i+128]; d2 = ((m.encode(C[b], Xn[b]) - c) ** 2).sum(1); lb = yl[b]
                    loss = torch.where(lb == 0, d2, (d2 + 1e-6) ** lb).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                d2 = torch.cat([((m.encode(C[i:i+8192], Xn[i:i+8192]) - c) ** 2).sum(1) for i in range(0, n, 8192)]).numpy()
            return entry_scores(d2, eid)
        res['DeepSAD'] = run_loop(f, y, mk); res['DeepSAD']['runtime_s'] = round(time.time() - t0, 1)
    with open(f'{RES}/feedback_{mode}_s{seed}.json', 'w') as fh: json.dump(res, fh, indent=1)
    print(json.dumps(res))
