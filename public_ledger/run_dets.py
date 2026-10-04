"""Run the Paper 1 detectors on the DIB ledger (line level). Scores saved to scores/<det>_s<seed>.npy.
Usage: python run_dets.py deterministic | stochastic <seed> | ceiling <seed>"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from dib_data import *

from dibpaths import SCORES as OUT, RESULTS as _RES
LOG = os.path.join(_RES, 'runtimes.jsonl')


def save(name, seed, s, t0, info=None):
    np.save(f'{OUT}/{name}_s{seed}.npy', np.asarray(s, np.float64))
    rec = {'det': name, 'seed': seed, 'runtime_s': round(time.time() - t0, 2)}
    if info: rec.update(info)
    with open(LOG, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(rec, flush=True)


def seed_everything(seed):
    import random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True); torch.set_num_threads(1)


def ae_emb(df, seed, epochs=30, batch=256, return_model=False):
    import torch, torch.nn as nn
    seed_everything(seed)
    codes, num, vocab = e_emb(df)
    C = torch.tensor(codes); Xn = torch.tensor(num)

    class AEE(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.ModuleList([nn.Embedding(v, d) for v, d in zip(vocab, EMB_DIMS)])
            din = sum(EMB_DIMS) + Xn.shape[1]
            s.enc = nn.Sequential(nn.Linear(din, 64), nn.LeakyReLU(0.01), nn.Linear(64, 32), nn.LeakyReLU(0.01), nn.Linear(32, 8))
            s.dec = nn.Sequential(nn.Linear(8, 32), nn.LeakyReLU(0.01), nn.Linear(32, 64), nn.LeakyReLU(0.01))
            s.num_head = nn.Linear(64, Xn.shape[1])
            s.cat_heads = nn.ModuleList([nn.Linear(64, v) for v in vocab])
        def embed(s, c, x):
            return torch.cat([e(c[:, i]) for i, e in enumerate(s.emb)] + [x], 1)
        def encode(s, c, x):
            return s.enc(s.embed(c, x))
        def forward(s, c, x):
            h = s.dec(s.encode(c, x))
            mse = ((s.num_head(h) - x) ** 2).mean(1)
            ce = sum(nn.functional.cross_entropy(hd(h), c[:, i], reduction='none') for i, hd in enumerate(s.cat_heads))
            return mse + ce
    m = AEE(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed); n = len(Xn)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            b = perm[i:i + batch]; loss = m(C[b], Xn[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, n, 8192): out.append(m(C[i:i + 8192], Xn[i:i + 8192]))
    s = torch.cat(out).numpy()
    return (s, m, C, Xn) if return_model else s


if __name__ == '__main__':
    mode = sys.argv[1]; df = load(); X, cols = e_base(df)
    if mode == 'deterministic':
        from pyod.models.lof import LOF
        from pyod.models.knn import KNN
        from pyod.models.hbos import HBOS
        from pyod.models.ecod import ECOD
        from pyod.models.pca import PCA
        t = time.time(); s, P = decomp(df); save('DECOMP', 0, s, t)
        np.save(f'{OUT}/DECOMP_components.npy', np.stack(list(P.values()), 1))
        t = time.time(); s, P = decomp(df, extended=True); save('DECOMP+ctx', 0, s, t)
        np.save(f'{OUT}/DECOMPctx_components.npy', np.stack(list(P.values()), 1))
        t = time.time(); save('HBOS', 0, HBOS().fit(X).decision_scores_, t)
        t = time.time(); save('ECOD', 0, ECOD(n_jobs=1).fit(X).decision_scores_, t)
        t = time.time(); m = PCA(n_components=0.90, svd_solver='covariance_eigh').fit(X)
        save('PCA', 0, m.decision_scores_, t, {'n_components': int(m.n_components_)})
        if os.environ.get('DIB_SKIP_SLOW'):
            print('DIB_SKIP_SLOW set: skipping kNN and LOF (about 16 min together)', flush=True)
        else:
            t = time.time(); save('kNN', 0, KNN(n_neighbors=5, method='largest').fit(X).decision_scores_, t)
            t = time.time(); save('LOF', 0, LOF(n_neighbors=20).fit(X).decision_scores_, t)
    elif mode == 'stochastic':
        seed = int(sys.argv[2])
        from pyod.models.iforest import IForest
        if not os.path.exists(f'{OUT}/IF_s{seed}.npy'):
          t = time.time(); save('IF', seed, IForest(n_estimators=200, max_samples=256, random_state=seed).fit(X).decision_scores_, t)
        from pyod.models.auto_encoder import AutoEncoder
        from pyod.models.vae import VAE
        done = lambda d: os.path.exists(f'{OUT}/{d}_s{seed}.npy')
        seed_everything(seed); t = time.time()
        if not done('AE-pyod'):
          m = AutoEncoder(hidden_neuron_list=[64, 32, 8], epoch_num=30, batch_size=256, random_state=seed, verbose=0)
          m.fit(X); save('AE-pyod', seed, m.decision_scores_, t)
        seed_everything(seed); t = time.time()
        if not done('VAE-pyod'):
          m = VAE(encoder_neuron_list=[64, 32], decoder_neuron_list=[32, 64], latent_dim=8,
                epoch_num=30, batch_size=256, random_state=seed, verbose=0)
          m.fit(X); save('VAE-pyod', seed, m.decision_scores_, t)
        if not done('AE-emb'):
            t = time.time(); s, m, C, Xn = ae_emb(df, seed, return_model=True)
            import torch; torch.save(m.state_dict(), f'{OUT}/ae_emb_s{seed}.pt'); save('AE-emb', seed, s, t)
    elif mode == 'ceiling':
        seed = int(sys.argv[2])
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedGroupKFold
        y = df.label.values; grp = df.eid.values; oof = np.zeros(len(y)); t = time.time()
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, grp):
            clf = lgb.LGBMClassifier(scale_pos_weight=(y[tr] == 0).sum() / (y[tr] == 1).sum(),
                                     random_state=seed, verbose=-1, n_jobs=1)
            clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        save('LGBM-ceiling', seed, oof, t)
