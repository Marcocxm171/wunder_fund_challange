"""Data access, the fixed validation split and the official metric.

The starter pack (folder with utils.py and datasets/) is found from, in order: --root, $WUNDER_ROOT,
the current folder, and the folders above this repository. utils.py is the competition's own file and
is imported as-is: the official GlobalAccumulator is used for every reported score.

Validation split (fixed for the whole search, by row-group index of valid.parquet):
  A = even row groups  -> development: monitoring, tuning, checkpoint choice, blend weights
  B = odd row groups   -> untouched confirmation; only scored when a run finishes
Monitoring subsets (fixed): MON_A = first 300 of A; MON_TRAIN = 100 train sequences (seed 12345).
"""
from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
_utils = None


def find_root(root=None) -> Path:
    cands = [root, os.environ.get("WUNDER_ROOT"), Path.cwd(), REPO.parent, REPO.parent.parent]
    for c in cands:
        if c and (Path(c) / "utils.py").exists() and (Path(c) / "datasets").is_dir():
            return Path(c).resolve()
    raise SystemExit("Starter pack not found. Pass --root <folder with utils.py and datasets/> or set WUNDER_ROOT.")


def official(root=None):
    """The competition's utils.py module."""
    global _utils
    if _utils is None:
        r = str(find_root(root))
        if r not in sys.path:
            sys.path.insert(0, r)
        import utils as u                              # noqa: E402  (the starter pack's file)
        _utils = u
    return _utils


def dataset_paths(root=None):
    r = find_root(root)
    d = r / "datasets"
    return {"train": d / "train.parquet", "valid": d / "valid.parquet", "valid_mask": d / "valid_mask.parquet"}


def n_groups(path) -> int:
    return pq.ParquetFile(path).num_row_groups


def split_valid(n_valid):
    g = np.arange(n_valid)
    return g[g % 2 == 0], g[g % 2 == 1]


def monitor_groups(n_train, n_valid):
    A, _ = split_valid(n_valid)
    rng = np.random.default_rng(12345)
    return A[:300], np.sort(rng.choice(n_train, min(100, n_train), replace=False))


def read_sequence(pf, group, scored=False):
    """X [20000, 112] float32 in utils.FEATURE_COLUMNS order (exactly the DataPoint.state values),
    Y [20000, 2] float32 (t0, t1), S [20000] bool is_scored (if requested)."""
    u = official()
    cols = list(u.FEATURE_COLUMNS) + list(u.TARGET_COLUMNS) + (["is_scored"] if scored else [])
    t = pf.read_row_group(int(group), columns=cols)
    X = np.column_stack([t[c].to_numpy() for c in u.FEATURE_COLUMNS]).astype(np.float32)
    Y = np.column_stack([t[c].to_numpy() for c in u.TARGET_COLUMNS]).astype(np.float32)
    S = t["is_scored"].to_numpy().astype(bool) if scored else None
    return X, Y, S


def batches(path, groups, batch_size, scored=False, prefetch=2):
    """Yield (X [B,T,112], Y [B,T,2], S [B,T] or None, groups) with reading done in a background thread."""
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    groups = list(groups)

    def worker():
        try:
            pf = pq.ParquetFile(path)
            for i in range(0, len(groups), batch_size):
                gs = groups[i:i + batch_size]
                seqs = [read_sequence(pf, g, scored) for g in gs]
                X = np.stack([s[0] for s in seqs]); Y = np.stack([s[1] for s in seqs])
                S = np.stack([s[2] for s in seqs]) if scored else None
                q.put((X, Y, S, gs))
        except BaseException as e:                     # surface reader errors in the main thread
            q.put(e)
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        yield item


def chunk_spans(T, chunk, ctx):
    """Training chunks: (a, s, e) = read rows a..e-1, predict/learn rows s..e-1, with a = max(0, s - ctx)."""
    return [(max(0, s - ctx), s, min(s + chunk, T)) for s in range(0, T, chunk)]


# ----------------------------------------------------------------------------- metric
def seq_moments(y, p, mask):
    """Per-sequence sufficient statistics of the clipped, |y|-weighted Pearson, per target.
    Returns [2, 6]: W, Sum w*y, Sum w*p, Sum w*y^2, Sum w*p^2, Sum w*y*p (float64).
    Summing these over any set of sequences gives that set's global WP exactly (used for bootstraps)."""
    out = np.zeros((2, 6))
    if not mask.any():
        return out
    yy = np.clip(y[mask].astype(np.float64), -2, 2); pp = np.clip(p[mask].astype(np.float64), -2, 2)
    for k in range(2):
        w, a, b = np.abs(yy[:, k]), yy[:, k], pp[:, k]
        out[k] = [w.sum(), (w * a).sum(), (w * b).sum(), (w * a * a).sum(), (w * b * b).sum(), (w * a * b).sum()]
    return out


def wp_from_moments(m):
    """m [..., 2, 6] summed moments -> (wp, t0, t1). Same conventions as METRIC.md."""
    W, Sy, Sp, Syy, Spp, Syp = (m[..., i] for i in range(6))
    with np.errstate(invalid="ignore", divide="ignore"):
        my, mp = Sy / W, Sp / W
        cov = Syp / W - my * mp
        vy = np.maximum(Syy / W - my * my, 0); vp = np.maximum(Spp / W - mp * mp, 0)
        r = cov / np.sqrt(vy * vp)
    ok = (W >= 1e-8) & (np.sqrt(vy) > 1e-8) & (np.sqrt(vp) > 1e-8)
    r = np.where(ok, np.clip(r, -1, 1), 0.0)
    return r.mean(-1), r[..., 0], r[..., 1]


class Scores:
    """Official GlobalAccumulator on (a) scored rows = is_scored & need_prediction and (b) all required rows;
    also keeps per-sequence moments for bootstraps."""

    def __init__(self):
        u = official()
        self.acc_s, self.acc_a = u.GlobalAccumulator(), u.GlobalAccumulator()
        self.mom_s, self.mom_a, self.groups = [], [], []
        self.need = np.arange(u.SEQUENCE_LENGTH) >= u.WARMUP

    def add(self, y, p, s, group=None):
        m = s & self.need
        self.acc_s.add(y, p, m); self.acc_a.add(y, p, self.need)
        self.mom_s.append(seq_moments(y, p, m)); self.mom_a.append(seq_moments(y, p, self.need))
        self.groups.append(-1 if group is None else int(group))

    def result(self):
        rs, ra = self.acc_s.result(), self.acc_a.result()
        return {"scored": {"wp": rs["weighted_pearson"], "t0": rs["t0"], "t1": rs["t1"], "rows": rs["selected_rows"]},
                "all": {"wp": ra["weighted_pearson"], "t0": ra["t0"], "t1": ra["t1"], "rows": ra["selected_rows"]},
                "sequences": rs["blocks"]}

    def moments(self):
        return {"groups": np.array(self.groups), "scored": np.array(self.mom_s), "all": np.array(self.mom_a)}


def reference_global_wp(Y, P, mask):
    """Independent re-implementation of the METRIC.md reference (all rows in memory). Used only by checks."""
    Y = np.asarray(Y, np.float32); P = np.asarray(P, np.float32); mask = np.asarray(mask, bool)
    out = []
    for k in range(2):
        y = np.clip(Y[mask, k], -2, 2).astype(np.float64); p = np.clip(P[mask, k], -2, 2).astype(np.float64)
        w = np.abs(y); W = w.sum()
        if W < 1e-8:
            out.append(0.0); continue
        yc = y - (w * y).sum() / W; pc = p - (w * p).sum() / W
        sy = np.sqrt((w * yc * yc).sum() / W); sp = np.sqrt((w * pc * pc).sum() / W)
        out.append(0.0 if sy <= 1e-8 or sp <= 1e-8 else float(np.clip((w * yc * pc).sum() / W / (sy * sp), -1, 1)))
    return float(np.mean(out)), out[0], out[1]
