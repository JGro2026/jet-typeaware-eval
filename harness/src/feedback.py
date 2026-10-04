"""Simulated-auditor feedback protocol (Section 3.5).

The oracle is the only object that holds labels; a method can obtain a label
exclusively through `Oracle.review`, which asserts on the reviewed mask every
round. Methods: STATIC-AVG, DeepSAD, LinTS, GREEDY-LR.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .evaluate import precision_at_k, ranking

ROUNDS = 4
BUDGET = 25
REVIEW_KS = [25, 50, 100]


class LabelLeak(AssertionError):
    pass


class Oracle:
    """Holds the hidden labels. `review(rows)` reveals labels for exactly the
    given unreviewed rows and records them in the reviewed mask. Any attempt
    to read a label of a row that is not in the reviewed mask raises."""

    def __init__(self, labels: np.ndarray):
        self._labels = np.asarray(labels).astype(int)
        self.reviewed = np.zeros(len(self._labels), dtype=bool)
        self.history: list[np.ndarray] = []

    @property
    def n(self) -> int:
        return len(self._labels)

    def review(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=int)
        if len(np.unique(rows)) != len(rows):
            raise LabelLeak("duplicate rows in a review batch")
        if self.reviewed[rows].any():
            raise LabelLeak("a row was presented for review twice")
        self.reviewed[rows] = True
        self.history.append(rows.copy())
        return self._labels[rows].copy()

    def labels_for(self, rows: np.ndarray) -> np.ndarray:
        """Labels of already reviewed rows only; asserts on the mask."""
        rows = np.asarray(rows, dtype=int)
        if not self.reviewed[rows].all():
            raise LabelLeak("label requested for an unreviewed row")
        return self._labels[rows].copy()

    def feedback(self) -> tuple[np.ndarray, np.ndarray]:
        """(rows, labels) of everything reviewed so far."""
        rows = np.flatnonzero(self.reviewed)
        return rows, self._labels[rows].copy()

    def assert_mask(self, rows_used: np.ndarray) -> None:
        rows_used = np.asarray(rows_used, dtype=int)
        if not self.reviewed[rows_used].all():
            raise LabelLeak("method used labels outside the reviewed rows")

    # Evaluation helpers, called by the harness after the last round only.
    def final_metrics(self, scores: np.ndarray) -> dict[str, float]:
        unl = ~self.reviewed
        s = np.asarray(scores, dtype=float)[unl]
        y = self._labels[unl]
        out = {f"p_at_{k}": precision_at_k(s, y, k) for k in REVIEW_KS}
        out["remaining_anomalies"] = float(y.sum())
        return out

    def found_so_far(self) -> int:
        return int(self._labels[self.reviewed].sum())


class FeedbackMethod:
    """Interface. `initial_scores` gives the round-1 ranking; `update` receives
    all feedback so far (rows, labels) and returns new scores for all rows."""
    name = "base"

    def initial_scores(self) -> np.ndarray:
        raise NotImplementedError

    def update(self, rows: np.ndarray, labels: np.ndarray, round_idx: int) -> np.ndarray:
        raise NotImplementedError


class StaticAvg(FeedbackMethod):
    name = "STATIC-AVG"

    def __init__(self, avg_scores: np.ndarray):
        self.scores = np.asarray(avg_scores, dtype=float)

    def initial_scores(self) -> np.ndarray:
        return self.scores

    def update(self, rows, labels, round_idx) -> np.ndarray:
        return self.scores


@dataclass
class FeedbackRecord:
    method: str
    round: int
    reviews_cumulative: int
    anomalies_found: int
    p_at_25: float = float("nan")
    p_at_50: float = float("nan")
    p_at_100: float = float("nan")
    extra: dict = field(default_factory=dict)


def run_rounds(method: FeedbackMethod, oracle: Oracle, rounds: int = ROUNDS,
               budget: int = BUDGET) -> list[FeedbackRecord]:
    """Round r: present the top-b unreviewed rows of the current ranking,
    reveal their labels, update the method on all feedback so far, re-rank.
    P@k of the final ranking over the unreviewed remainder is attached to the
    last record."""
    scores = np.asarray(method.initial_scores(), dtype=float)
    records: list[FeedbackRecord] = []
    for r in range(1, rounds + 1):
        order = ranking(scores)
        unreviewed = order[~oracle.reviewed[order]]
        batch = unreviewed[:budget]
        oracle.review(batch)
        rows, labels = oracle.feedback()
        oracle.assert_mask(rows)
        scores = np.asarray(method.update(rows, labels, r), dtype=float)
        if scores.shape != (oracle.n,):
            raise ValueError("method must return one score per row")
        records.append(FeedbackRecord(method.name, r, int(oracle.reviewed.sum()), oracle.found_so_far()))
    final = oracle.final_metrics(scores)
    records[-1].p_at_25 = final["p_at_25"]
    records[-1].p_at_50 = final["p_at_50"]
    records[-1].p_at_100 = final["p_at_100"]
    records[-1].extra["remaining_anomalies"] = final["remaining_anomalies"]
    return records


# ---------------------------------------------------------------- LinTS

class LinTS(FeedbackMethod):
    """Linear Thompson sampling on the context x_i (rank-normalised scores of
    detectors 1 to 10 plus an intercept column). Bayesian linear regression
    with Gaussian prior and noise variance 1. Protocol setting: prior N(0, I)
    (center="zero", intercept="penalised", prior_var=1.0). Round 1 always
    uses uniform weights on the ten detector columns (the AVG ranking); every
    later round samples theta from the posterior given all feedback so far
    and ranks by x . theta.

    Variants offered at the WP2 checkpoint: center="uniform" puts the prior
    mean at the uniform weights theta0; intercept="free" gives the intercept
    coordinate a flat prior (variance 1e6); prior_var scales the prior
    variance of the detector weights."""
    name = "LinTS"

    def __init__(self, context: np.ndarray, seed: int, prior_var: float = 1.0, noise_var: float = 1.0,
                 center: str = "zero", intercept: str = "penalised"):
        self.X = np.asarray(context, dtype=np.float64)
        self.seed = int(seed)
        self.noise_var = noise_var
        d = self.X.shape[1]
        self.theta0 = np.r_[np.full(d - 1, 1.0 / (d - 1)), 0.0]
        self.mu0 = np.zeros(d) if center == "zero" else self.theta0.copy()
        pv = np.full(d, float(prior_var))
        if intercept == "free":
            pv[-1] = 1e6
        self.prior_prec = 1.0 / pv
        self.settings = dict(prior_var=prior_var, noise_var=noise_var, center=center, intercept=intercept)

    def initial_scores(self) -> np.ndarray:
        return self.X @ self.theta0

    def update(self, rows, labels, round_idx) -> np.ndarray:
        Xr = self.X[rows]
        y = np.asarray(labels, dtype=np.float64)
        A = np.diag(self.prior_prec) + Xr.T @ Xr / self.noise_var
        cov = np.linalg.inv(A)
        mean = cov @ (self.prior_prec * self.mu0 + Xr.T @ y / self.noise_var)
        rng = np.random.default_rng([self.seed, 0x715, round_idx])
        theta = rng.multivariate_normal(mean, cov, method="cholesky")
        self.last_theta = theta
        return self.X @ theta


# ---------------------------------------------------------------- GREEDY-LR

class GreedyLR(FeedbackMethod):
    """L2 logistic regression (C = 1.0) on the same context, refit every round
    on all feedback so far, ranking by predicted probability. Round 1 is the
    AVG ranking. Protocol setting: center="zero", intercept="penalised", i.e.
    sklearn's LogisticRegression(C=1.0, fit_intercept=False) on the context
    including its intercept column (verified to match the closed-form
    objective below). Variants offered at the WP2 checkpoint: center="uniform"
    penalises theta - theta0 instead of theta; intercept="free" leaves the
    intercept coefficient unpenalised. If the feedback so far contains a single
    class the AVG ranking is kept for that round (recorded)."""
    name = "GREEDY-LR"

    def __init__(self, context: np.ndarray, seed: int, C: float = 1.0, center: str = "zero",
                 intercept: str = "penalised"):
        self.X = np.asarray(context, dtype=np.float64)
        self.seed = int(seed)
        self.C = C
        d = self.X.shape[1]
        self.theta0 = np.r_[np.full(d - 1, 1.0 / (d - 1)), 0.0]
        self.mu0 = np.zeros(d) if center == "zero" else self.theta0.copy()
        self.pen = np.ones(d)
        if intercept == "free":
            self.pen[-1] = 0.0
        self.settings = dict(C=C, center=center, intercept=intercept)
        self.fallback_rounds: list[int] = []

    def initial_scores(self) -> np.ndarray:
        return self.X @ self.theta0

    def update(self, rows, labels, round_idx) -> np.ndarray:
        from scipy.optimize import minimize
        y = np.asarray(labels, dtype=float)
        if len(np.unique(y)) < 2:
            self.fallback_rounds.append(round_idx)
            return self.initial_scores()
        Xr, mu0, pen, C = self.X[rows], self.mu0, self.pen, self.C
        yy = 2.0 * y - 1.0

        def f(theta):
            z = Xr @ theta
            dev = theta - mu0
            loss = np.logaddexp(0.0, -yy * z).sum() + (pen * dev * dev).sum() / (2.0 * C)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xr.T @ (p - y) + pen * dev / C
            return loss, grad

        res = minimize(f, mu0.copy(), jac=True, method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-8})
        self.last_theta = res.x
        return 1.0 / (1.0 + np.exp(-(self.X @ res.x)))


# ---------------------------------------------------------------- DeepSAD

class DeepSAD(FeedbackMethod):
    """Deep SAD (Ruff et al., 2020) on the AE-emb encoder of the same cell.

    Network phi = embedding layers plus the AE-emb encoder, initialised from
    the trained AE-emb state dict. Centre c = mean latent code of all rows at
    initialisation (entries with |c_j| < 0.1 are pushed to +-0.1 as in the
    reference implementation). Loss per row: ||phi(x) - c||^2 for unlabeled
    rows, eta * (||phi(x) - c||^2 + eps)^(y~) for labeled rows with y~ = +1
    for reviewed normals and y~ = -1 for reviewed anomalies; batch mean.
    Adam lr 1e-4, weight decay 1e-6, 50 epochs, batch 128, retrained from the
    AE initialisation every round on all rows with all feedback so far.
    Score = squared distance to c. Round 1 ranking = AE-emb ranking."""
    name = "DeepSAD"

    def __init__(self, enc, ae_state: dict, ae_scores: np.ndarray, seed: int, eta: float = 1.0,
                 lr: float = 1e-4, weight_decay: float = 1e-6, epochs: int = 50, batch_size: int = 128,
                 eps: float = 1e-6):
        import torch
        from .detectors import build_ae_emb, seed_everything

        self.seed = int(seed)
        self.eta, self.lr, self.wd, self.epochs, self.bs, self.eps = eta, lr, weight_decay, epochs, batch_size, eps
        self.ae_scores = np.asarray(ae_scores, dtype=np.float64)
        self.cat = torch.as_tensor(enc.cat_codes, dtype=torch.long)
        self.num = torch.as_tensor(enc.num, dtype=torch.float32)
        self.ae_state = ae_state
        self.vocab_sizes = list(enc.vocab_sizes)
        self.n_num = int(self.num.shape[1])
        seed_everything(self.seed)
        model = self._fresh_model()
        model.eval()
        with torch.no_grad():
            z = torch.cat([model.encode(self.cat[s: s + 8192], self.num[s: s + 8192])
                           for s in range(0, len(self.cat), 8192)])
        c = z.mean(dim=0)
        small = c.abs() < 0.1
        c[small] = torch.where(c[small] >= 0, torch.full_like(c[small], 0.1), torch.full_like(c[small], -0.1))
        self.c = c
        self.round_runtimes: list[float] = []

    def _fresh_model(self):
        from .detectors import build_ae_emb
        model = build_ae_emb(self.vocab_sizes, self.n_num)
        model.load_state_dict(self.ae_state)
        return model

    def initial_scores(self) -> np.ndarray:
        return self.ae_scores

    def update(self, rows, labels, round_idx) -> np.ndarray:
        import torch
        from .detectors import seed_everything

        t0 = time.perf_counter()
        seed_everything(self.seed * 1000 + round_idx)
        torch.set_num_threads(1)
        N = len(self.cat)
        semi = torch.zeros(N, dtype=torch.float32)
        semi[torch.as_tensor(rows)] = torch.where(torch.as_tensor(labels) == 1, -1.0, 1.0)
        model = self._fresh_model()
        model.train()
        params = list(model.embs.parameters()) + list(model.encoder.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)
        g = torch.Generator().manual_seed(self.seed * 1000 + round_idx)
        for _ in range(self.epochs):
            perm = torch.randperm(N, generator=g)
            for s in range(0, N, self.bs):
                b = perm[s: s + self.bs]
                opt.zero_grad()
                dist = ((model.encode(self.cat[b], self.num[b]) - self.c) ** 2).sum(dim=1)
                st = semi[b]
                loss = torch.where(st == 0, dist, self.eta * ((dist + self.eps) ** st)).mean()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            scores = torch.cat([((model.encode(self.cat[s: s + 8192], self.num[s: s + 8192]) - self.c) ** 2).sum(dim=1)
                                for s in range(0, N, 8192)]).numpy().astype(np.float64)
        self.round_runtimes.append(time.perf_counter() - t0)
        return scores
