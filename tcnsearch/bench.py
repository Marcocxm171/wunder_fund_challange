"""CPU inference speed, row by row, the way the competition runs it (1 vCPU, ONNX Runtime, one thread).

Run on an IDLE machine (not while GPUs are training: CPU load distorts the timing).
    python -m tcnsearch.bench                          # finalists + GRU from runs/_search/finalists.json
    python -m tcnsearch.bench --runs-dirs runs/<id> runs/<id2>
    python -m tcnsearch.bench --with-gru 1 2           # also time each model combined with 1 / 2 GRU-64s in ONE graph

Each model is exported from its selected.pt and driven through a scorer-like loop (a DataPoint per row,
predict(), state reset per sequence) over --rows rows of real validation data.
Platform minutes are an ESTIMATE: (us per row / us per row of the 3 x GRU-64 reference) x --ref-minutes,
where the reference (one ONNX graph with three GRU-64s) took 36 min on the platform for your earlier submission.
Writes bench.json into each run folder and runs/_search/bench.json.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from . import data as D
from .models import GRUModel, build_model, inference_copy, is_recurrent, n_params
from .streaming import ConvStreamer, StepModule, _export, export_onnx, ort_session, ort_streamer


class MultiGRUStep(nn.Module):
    """K GRU-64s in one graph (the reference submission layout). states [K, 2, 1, H]."""

    def __init__(self, K, H=64):
        super().__init__()
        torch.manual_seed(0)
        self.ms = nn.ModuleList([GRUModel(H).eval() for _ in range(K)])

    def forward(self, x, states):
        preds, outs = [], []
        for i, m in enumerate(self.ms):
            p, (h1, h2) = m(x, (states[i, 0], states[i, 1]))
            preds.append(p); outs.append(torch.stack([h1, h2]))
        return torch.stack(preds).mean(0), torch.stack(outs)


class ConvPlusGRU(nn.Module):
    def __init__(self, model, K, H=64):
        super().__init__()
        self.c = StepModule(model); self.g = MultiGRUStep(K, H); self.K = K

    def forward(self, x, past, states):
        p1, now = self.c(x, past)
        p2, st = self.g(x, states)
        return (p1 + self.K * p2) / (1 + self.K), now, st


def drive(step_fn, X, reset):
    """Scorer-like loop: DataPoint creation + predict + reset at new sequences. Returns microseconds per row."""
    u = D.official()
    n = len(X)
    t = time.perf_counter()
    seq = -1
    for i in range(n):
        dp = u.DataPoint(i // 20000, i % 20000, (i % 20000) >= u.WARMUP, X[i])
        if dp.seq_ix != seq:
            reset(); seq = dp.seq_ix
        p = step_fn(dp.state)
        _ = np.asarray(p, np.float32) if dp.need_prediction else None
    return (time.perf_counter() - t) / n * 1e6


def bench_model(model, X, td, with_gru=()):
    out = {}
    path = Path(td) / "m.onnx"
    info = export_onnx(model, path)
    s = ort_streamer(path, info)
    out["alone_us_per_row"] = drive(s.step, X, s.reset)
    for K in with_gru:
        if is_recurrent(model):
            continue
        mod = ConvPlusGRU(model, K).eval()
        p = Path(td) / f"m_g{K}.onnx"
        _export(mod, (torch.zeros(1, 1, 112), torch.zeros(1, mod.c.plan.P, 1), torch.zeros(K, 2, 1, 1, 64)), p,
                ["features", "past", "states"], ["prediction", "now", "states_out"])
        sess = ort_session(p)
        streamer = ConvStreamer(None, mod.c.plan)
        st = {"s": np.zeros((K, 2, 1, 1, 64), np.float32)}

        def run(x, past):
            pr, now, st["s"] = sess.run(None, {"features": x, "past": past, "states": st["s"]})
            return pr, now
        streamer.run = run

        def reset():
            streamer.reset(); st["s"][:] = 0
        out[f"with_{K}_gru_us_per_row"] = drive(streamer.step, X, reset)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--runs", default=str(D.REPO / "runs"))
    ap.add_argument("--runs-dirs", nargs="*", default=None)
    ap.add_argument("--rows", type=int, default=40000)
    ap.add_argument("--ref-minutes", type=float, default=36.0, help="platform minutes of the 3 x GRU-64 reference")
    ap.add_argument("--with-gru", type=int, nargs="*", default=[1, 2])
    a = ap.parse_args()
    torch.set_num_threads(1)
    D.official(a.root)
    paths = D.dataset_paths(a.root)
    runs = Path(a.runs)
    if a.runs_dirs:
        dirs = [Path(d) for d in a.runs_dirs]
    else:
        fl = json.loads((runs / "_search" / "finalists.json").read_text())
        dirs = [runs / g[0] for g in fl["tcn"]] + [runs / fl["gru"][0]]      # seed-0 run folder of each finalist
    pf = pq.ParquetFile(paths["valid"])
    X = np.concatenate([D.read_sequence(pf, g)[0] for g in range(min(pf.num_row_groups, -(-a.rows // 20000)))])[:a.rows]
    with tempfile.TemporaryDirectory() as td:
        ref = MultiGRUStep(3).eval()
        p = Path(td) / "ref.onnx"
        _export(ref, (torch.zeros(1, 1, 112), torch.zeros(3, 2, 1, 1, 64)), p, ["features", "states"], ["prediction", "states_out"])
        sess = ort_session(p); st = {"s": np.zeros((3, 2, 1, 1, 64), np.float32)}

        def ref_step(x):
            pr, st["s"] = sess.run(None, {"features": x.reshape(1, 1, 112), "states": st["s"]})
            return pr
        ref_us = drive(ref_step, X, lambda: st["s"].fill(0))
        print(f"reference 3 x GRU-64: {ref_us:.1f} us/row = {a.ref_minutes:.0f} platform minutes (by definition)")
        allres = {"reference_us_per_row": ref_us, "ref_minutes": a.ref_minutes, "rows": len(X), "runs": {}}
        for d in dirs:
            fin = json.loads((d / "final.json").read_text())
            m = build_model(fin["model"]); m.load_state_dict(torch.load(d / "selected.pt", map_location="cpu"))
            r = bench_model(inference_copy(m), X, td, a.with_gru)
            r = {**r, **{k.replace("us_per_row", "platform_min_ESTIMATE"): v / ref_us * a.ref_minutes for k, v in r.items()},
                 "params": n_params(m), "name": fin["name"]}
            (d / "bench.json").write_text(json.dumps(r, indent=2))
            allres["runs"][d.name] = r
            print(f"{fin['name']:40s} " + " | ".join(f"{k}: {v:.1f}" for k, v in r.items() if isinstance(v, float)))
    (runs / "_search").mkdir(exist_ok=True)
    (runs / "_search" / "bench.json").write_text(json.dumps(allres, indent=2))


if __name__ == "__main__":
    main()
