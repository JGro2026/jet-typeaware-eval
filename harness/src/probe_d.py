"""Table-D runtime probe for the WP2 checkpoint (Section 3.7).

Times the cheap detectors and the ceiling directly, runs one epoch of each
neural model and extrapolates, and times full LOF and kNN. Writes
reports/wp2_table_d_probe.json. Nothing here is a benchmark result.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .detectors import build_ae_emb, run_ceiling, run_decomp, run_pyod, seed_everything
from .encode import encode_base, encode_emb
from .inject import inject
from .load import ROOT, load_table

out: dict = {}
t0 = time.perf_counter()
df = load_table("D")
inj = inject(df, "D", 0.015, "medium", 0)
out["inject_s"] = time.perf_counter() - t0
t0 = time.perf_counter()
base = encode_base(inj.data)
emb = encode_emb(inj.data)
out["encode_s"] = time.perf_counter() - t0
out["E_base_dim"] = int(base.X.shape[1])
del df
import gc; gc.collect()
print(f"inject {out['inject_s']:.1f}s, encode {out['encode_s']:.1f}s, E_base dim {base.X.shape[1]}", flush=True)

for det in ["HBOS", "ECOD", "PCA", "IF"]:
    r = run_pyod(det, base, 0)
    out[f"{det}_s"] = r.runtime_s
    print(f"{det:<10s} {r.runtime_s:8.1f}s", flush=True)
r = run_decomp(inj.data)
out["DECOMP_s"] = r.runtime_s
print(f"DECOMP     {r.runtime_s:8.1f}s", flush=True)
r = run_ceiling(base, inj.labels, 0)
out["ceiling_s"] = r.runtime_s
print(f"ceiling    {r.runtime_s:8.1f}s", flush=True)

# One-epoch probes of the PyOD networks (epoch_num=1, everything else as in the protocol)
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
X = base.X.astype(np.float64)
for name, ctor in [("AE-pyod", lambda: AutoEncoder(hidden_neuron_list=[64, 32, 8], epoch_num=1, batch_size=64, random_state=0, verbose=0)),
                   ("VAE-pyod", lambda: VAE(encoder_neuron_list=[64, 32], decoder_neuron_list=[32, 64], latent_dim=8, epoch_num=1, batch_size=64, random_state=0, verbose=0))]:
    seed_everything(0)
    t0 = time.perf_counter()
    ctor().fit(X)
    dt = time.perf_counter() - t0
    out[f"{name}_1epoch_s"] = dt
    out[f"{name}_100epochs_proj_s"] = dt * 100
    print(f"{name:<10s} 1 epoch {dt:7.1f}s  -> 100 epochs {dt * 100 / 60:6.1f} min", flush=True)

# AE-emb: one epoch at batch 256 (Table-D setting) and at batch 64 (protocol default)
cat = torch.as_tensor(emb.cat_codes); num = torch.as_tensor(emb.num)
for bs, epochs in [(256, 30), (64, 100)]:
    seed_everything(0)
    model = build_ae_emb(emb.vocab_sizes, num.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0); N = len(cat); t0 = time.perf_counter()
    perm = torch.randperm(N, generator=g)
    for s in range(0, N, bs):
        b = perm[s: s + bs]; opt.zero_grad(); loss = model.row_loss(cat[b], num[b]).mean(); loss.backward(); opt.step()
    dt = time.perf_counter() - t0
    out[f"AE-emb_bs{bs}_1epoch_s"] = dt
    out[f"AE-emb_bs{bs}_{epochs}epochs_proj_s"] = dt * epochs
    print(f"AE-emb bs{bs:<4d} 1 epoch {dt:7.1f}s  -> {epochs} epochs {dt * epochs / 60:6.1f} min", flush=True)

# DeepSAD: one epoch at batch 128, times 50 epochs times 4 rounds
seed_everything(0)
model = build_ae_emb(emb.vocab_sizes, num.shape[1])
params = list(model.embs.parameters()) + list(model.encoder.parameters())
opt = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-6)
c = torch.zeros(8) + 0.1; semi = torch.zeros(N); g = torch.Generator().manual_seed(0); t0 = time.perf_counter()
perm = torch.randperm(N, generator=g)
for s in range(0, N, 128):
    b = perm[s: s + 128]; opt.zero_grad()
    dist = ((model.encode(cat[b], num[b]) - c) ** 2).sum(dim=1); st = semi[b]
    loss = torch.where(st == 0, dist, 1.0 * ((dist + 1e-6) ** st)).mean(); loss.backward(); opt.step()
dt = time.perf_counter() - t0
out["DeepSAD_1epoch_s"] = dt
out["DeepSAD_4rounds_50epochs_proj_s"] = dt * 50 * 4
print(f"DeepSAD    1 epoch {dt:7.1f}s  -> 50 epochs x 4 rounds {dt * 200 / 60:6.1f} min", flush=True)

# Full LOF and kNN on all rows (brute force in 60+ dims)
for det in ["kNN", "LOF"]:
    r = run_pyod(det, base, 0)
    out[f"{det}_full_s"] = r.runtime_s
    print(f"{det:<10s} full {r.runtime_s:8.1f}s", flush=True)

(ROOT / "reports" / "wp2_table_d_probe.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
