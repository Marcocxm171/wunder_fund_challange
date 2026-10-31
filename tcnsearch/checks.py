"""Automatic checks. The search runner calls these before any training and refuses to train if one fails.

    python -m tcnsearch.checks                 # data checks (T1-T5, T9, T11) + model checks of the baseline configs
    python -m tcnsearch.checks --full          # T1 also reads every row of train and valid (slow, ~35 GB)
    python -m tcnsearch.checks --model-config runs/<id>/config.json

Data checks (once per data files + code version):
  T1 data integrity      row groups, 20,000 rows each, schema/column order, sequence ids, step order, warm-up
                         contract, train/valid id overlap, finite values, missing-marker shares (a5-a7)
  T2 mask                valid_mask.parquet == valid.parquet is_scored, row by row
  T3 preprocessing       loader X/Y == the official scorer's construction (bitwise)
  T4 target alignment    oracle = 1, constant = 0, one-row shift clearly < 1, random ~ 0
  T5 metric              official GlobalAccumulator == independent METRIC.md reference == moment formula
  T9 streaming           ONNX row by row through the official ScorerStepByStep == batched predictions
  T11 training           same seed twice -> same weights; a small model can fit 2 sequences
Model checks (every new model config, CPU, random weights):
  T6 causality           future rows never change the past (eval and train mode); d pred_t / d x_(t+j) == 0;
                         receptive field measured from gradients == formula
  T7 sequence boundaries batch order does not matter; streaming reset == fresh start (zero padding)
  T8 chunks              training chunks with context == full sequence, incl. a chunk size that does not divide T;
                         every row >= 99 is learned exactly once per pass
  T9m streaming          torch and ONNX row-by-row == full sequence
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from . import data as D
from .models import build_model, is_recurrent, n_params
from .streaming import PredictionModel, export_onnx, ort_streamer, torch_streamer

TOL_STREAM = 2e-5


def _res(name, ok, **info):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info, default=str)[:300]}", flush=True)
    return {"test": name, "pass": bool(ok), **info}


def cfg_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


# ============================================================================= model checks
def _full(m, x):
    with torch.no_grad():
        return m(x)[0] if is_recurrent(m) else m(x)


def check_model(model_cfg, seed=0, verbose=True):
    torch.manual_seed(seed)
    m = build_model(model_cfg).eval()
    rec = is_recurrent(m)
    RF = None if rec else m.receptive_field
    T = 600 if rec else int(min(max(3 * RF, 600), 7000))
    out = []
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, T, 112, generator=g)
    full = _full(m, x)

    # T6 causality: perturb rows after t0
    t0 = T // 2
    x2 = x.clone(); x2[:, t0 + 1:] = torch.randn(2, T - t0 - 1, 112, generator=g)
    d_eval = float((_full(m, x2)[:, :t0 + 1] - full[:, :t0 + 1]).abs().max())
    m.train()
    torch.manual_seed(1); a = m(x)[0] if rec else m(x)
    torch.manual_seed(1); b = m(x2)[0] if rec else m(x2)
    m.eval()
    d_train = float((a[:, :t0 + 1] - b[:, :t0 + 1]).abs().max())
    out.append(_res("T6a causality (future rows changed)", d_eval <= 1e-6 and d_train <= 1e-6, eval_diff=d_eval, train_diff=d_train))

    def input_grad(model):
        xg = x[:1].clone().requires_grad_(True)
        y = (model(xg)[0] if rec else model(xg))[0, t0].sum()
        y.backward()
        return xg.grad[0].abs().sum(1)                        # [T]

    def reach(gn):
        nz = torch.nonzero(gn[:t0 + 1] > 0).flatten()
        return int(t0 - nz.min()) + 1 if len(nz) else 0

    gn = input_grad(m)
    fut = float(gn[t0 + 1:].max())
    info = {"future_grad_max": fut}
    ok = fut == 0.0
    if not rec:
        lag_far = t0 - (RF - 1)
        measured = reach(gn)
        info |= {"rf_formula": RF, "rf_measured": measured, "grad_beyond_rf": float(gn[:max(lag_far, 0)].max()) if lag_far > 0 else 0.0}
        if measured != RF:
            # Dead ReLU paths at initialisation can zero the gradient at the furthest lags. Re-measure the
            # structural reach with activations replaced by identity (padding/dilations are what is tested).
            lin = copy.deepcopy(m)
            for mod in lin.modules():
                if hasattr(mod, "act") and callable(mod.act) and getattr(mod.act, "__name__", "") != "gated":
                    mod.act = lambda v: v
            gl = input_grad(lin)
            info |= {"rf_structural": reach(gl), "structural_future_grad_max": float(gl[t0 + 1:].max())}
            ok = ok and info["rf_structural"] == RF and measured <= RF and info["structural_future_grad_max"] == 0.0
        ok = ok and info["grad_beyond_rf"] == 0.0
    out.append(_res("T6b gradient causality + receptive field", ok, **info))

    # T7 batch independence and streaming reset
    perm = _full(m, x[[1, 0]])
    d_perm = float((perm[[1, 0]] - full).abs().max())
    out.append(_res("T7a batch order", d_perm <= 1e-6, diff=d_perm))
    n_stream = T if rec else min(T, RF + 300)
    s = torch_streamer(m)
    for t in range(50):                                       # run some rows of sequence 0, then reset
        s.step(x[0, t].numpy())
    s.reset()
    st = np.stack([s.step(x[1, t].numpy()) for t in range(n_stream)])
    d_reset = float(np.abs(st - full[1, :n_stream].numpy()).max())
    out.append(_res("T7b stream reset == fresh sequence (zero padding)", d_reset <= TOL_STREAM, diff=d_reset, rows=n_stream))

    # T8 chunks
    ctx = 0 if rec else RF - 1
    for chunk in (1000, 777):
        parts, state = [], None
        with torch.no_grad():
            for a_, s_, e_ in D.chunk_spans(T, chunk, ctx):
                if rec:
                    p, state = m(x[:, s_:e_], state); parts.append(p)
                else:
                    parts.append(m(x[:, a_:e_])[:, s_ - a_:])
        d_chunk = float((torch.cat(parts, 1) - full).abs().max())
        out.append(_res(f"T8a chunks of {chunk} with context == full", d_chunk <= 1e-5, diff=d_chunk))
    cover = np.zeros(20000, int)
    for a_, s_, e_ in D.chunk_spans(20000, 1000, ctx):
        cover[s_:e_] += 1
    cover[:99] = 0
    out.append(_res("T8b every row >= 99 learned exactly once", bool((cover[99:] == 1).all())))

    # T9m streaming torch + ONNX
    with tempfile.TemporaryDirectory() as td:
        info_onnx = export_onnx(m, Path(td) / "m.onnx")
        so = ort_streamer(Path(td) / "m.onnx", info_onnx)
        ot = np.stack([so.step(x[0, t].numpy()) for t in range(n_stream)])
    s.reset()
    tt = np.stack([s.step(x[0, t].numpy()) for t in range(n_stream)])
    f0 = full[0, :n_stream].numpy()
    d_t, d_o = float(np.abs(tt - f0).max()), float(np.abs(ot - f0).max())
    out.append(_res("T9m row by row == full (torch, ONNX)", d_t <= TOL_STREAM and d_o <= TOL_STREAM, torch=d_t, onnx=d_o))
    return {"model": model_cfg, "params": n_params(m), "receptive_field": RF, "tests": out,
            "pass": all(r["pass"] for r in out)}


def model_check_cached(model_cfg, cache_dir):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{cfg_hash(model_cfg)}.json"
    if f.exists():
        r = json.loads(f.read_text())
        if r.get("pass"):
            return r
    r = check_model(model_cfg)
    f.write_text(json.dumps(r, indent=2, default=str))
    return r


# ============================================================================= data checks
def t1_integrity(paths, full=False, sample=200):
    u = D.official()
    res = []
    feat_t = list(u.FEATURE_COLUMNS) + list(u.TARGET_COLUMNS)
    ids = {}
    for name in ("train", "valid"):
        pf = pq.ParquetFile(paths[name])
        names = pf.schema_arrow.names
        rows_ok = all(pf.metadata.row_group(i).num_rows == u.SEQUENCE_LENGTH for i in range(pf.num_row_groups))
        order_ok = [c for c in names if c in feat_t] == feat_t
        expected_valid = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", *feat_t]
        schema_ok = names == expected_valid if name == "valid" else order_ok and {"seq_ix", "step_in_seq", "need_prediction"} <= set(names)
        seqs, step_ok, warm_ok = [], True, True
        steps = np.arange(u.SEQUENCE_LENGTH)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["seq_ix", "step_in_seq", "need_prediction"])
            sid = t["seq_ix"].to_numpy()
            seqs.append(int(sid[0]))
            step_ok &= bool((sid == sid[0]).all() and np.array_equal(t["step_in_seq"].to_numpy(), steps))
            warm_ok &= bool(np.array_equal(t["need_prediction"].to_numpy().astype(bool), steps >= u.WARMUP))
        ids[name] = seqs
        res.append(_res(f"T1 {name}: layout", rows_ok and schema_ok and step_ok and warm_ok and len(set(seqs)) == len(seqs),
                        row_groups=pf.num_row_groups, rows_20000=rows_ok, schema=schema_ok, step_order=step_ok,
                        warmup_contract=warm_ok, unique_ids=len(set(seqs)) == len(seqs)))
        groups = range(pf.num_row_groups) if full else np.random.default_rng(0).choice(pf.num_row_groups, min(sample, pf.num_row_groups), replace=False)
        finite, miss, n = True, np.zeros(3), 0
        for g in groups:
            X, Y, _ = D.read_sequence(pf, g)
            finite &= bool(np.isfinite(X).all() and np.isfinite(Y).all())
            miss += np.isclose(X[:, 109:112], -5.199, atol=1e-3).mean(0); n += 1
        res.append(_res(f"T1 {name}: finite values ({'all' if full else len(groups)} sequences)", finite,
                        missing_marker_share_a5_a6_a7=[round(float(v), 4) for v in miss / n]))
    overlap = len(set(ids["train"]) & set(ids["valid"]))
    res.append(_res("T1 no sequence in both train and valid", overlap == 0, overlap=overlap,
                    train=len(ids["train"]), valid=len(ids["valid"])))
    return res


def t2_mask(paths):
    pv, pm = pq.ParquetFile(paths["valid"]), pq.ParquetFile(paths["valid_mask"])
    ok = pv.num_row_groups == pm.num_row_groups
    bad, scored = 0, 0
    if ok:
        for g in range(pv.num_row_groups):
            a = pv.read_row_group(g, columns=["seq_ix", "step_in_seq", "is_scored"])
            b = pm.read_row_group(g, columns=["seq_ix", "step_in_seq", "is_scored"])
            for c in ("seq_ix", "step_in_seq", "is_scored"):
                if not np.array_equal(a[c].to_numpy(), b[c].to_numpy()):
                    bad += 1; break
            scored += int(a["is_scored"].to_numpy().sum())
    return [_res("T2 valid_mask.parquet == valid is_scored", ok and bad == 0, sequences_differing=bad, scored_rows=scored)]


def t3_t4_t5(paths, n=40):
    u = D.official()
    pf = pq.ParquetFile(paths["valid"])
    groups = np.arange(min(n, pf.num_row_groups))
    res = []
    same = True
    for g in groups[:5]:
        t = pf.read_row_group(int(g))
        Xo = np.column_stack([t[c].to_numpy() for c in u.FEATURE_COLUMNS]).astype(np.float32)   # as ScorerStepByStep
        Yo = np.column_stack([t[c].to_numpy() for c in u.TARGET_COLUMNS]).astype(np.float32)
        X, Y, S = D.read_sequence(pf, g, scored=True)
        same &= np.array_equal(X, Xo) and np.array_equal(Y, Yo) and np.array_equal(S, t["is_scored"].to_numpy().astype(bool))
    res.append(_res("T3 loader == official scorer inputs/targets (bitwise)", same))

    Ys, Ss = [], []
    for g in groups:
        _, Y, S = D.read_sequence(pf, g, scored=True); Ys.append(Y); Ss.append(S)
    rng = np.random.default_rng(0)
    preds = {"oracle": [y.copy() for y in Ys], "constant": [np.full_like(y, 0.3) for y in Ys],
             "shift1": [np.vstack([y[:1], y[:-1]]) for y in Ys], "random": [rng.normal(size=y.shape).astype(np.float32) for y in Ys],
             "noisy": [y + rng.normal(scale=1.0, size=y.shape).astype(np.float32) for y in Ys]}
    vals = {}
    worst = 0.0
    need = np.arange(u.SEQUENCE_LENGTH) >= u.WARMUP
    for k, P in preds.items():
        sc = D.Scores()
        for y, p, s in zip(Ys, P, Ss):
            sc.add(y, p, s)
        off = sc.result()["scored"]["wp"]
        ref = D.reference_global_wp(np.concatenate(Ys), np.concatenate(P), np.concatenate([s & need for s in Ss]))[0]
        mom = float(D.wp_from_moments(sc.moments()["scored"].sum(0))[0])
        vals[k] = round(off, 6)
        if k in ("random", "noisy", "shift1"):
            worst = max(worst, abs(off - ref), abs(off - mom))
    res.append(_res("T4 target alignment", vals["oracle"] > 0.999999 and vals["constant"] == 0.0 and vals["shift1"] < 0.999
                    and abs(vals["random"]) < 0.02, **vals))
    res.append(_res("T5 official metric == METRIC.md reference == moments", worst < 1e-8, max_abs_diff=worst))
    return res


def t9_official_scorer(paths, model_cfgs, n_seq=2):
    """Write n_seq validation sequences to a small parquet and run the official ScorerStepByStep with ONNX models."""
    u = D.official()
    res = []
    pf = pq.ParquetFile(paths["valid"])
    with tempfile.TemporaryDirectory() as td:
        sub = Path(td) / "sub.parquet"
        with pq.ParquetWriter(sub, pf.schema_arrow) as w:
            for g in range(n_seq):
                w.write_table(pf.read_row_group(g))
        for cfg in model_cfgs:
            torch.manual_seed(0)
            m = build_model(cfg).eval()
            info = export_onnx(m, Path(td) / "m.onnx")
            outs = []

            class Rec(PredictionModel):
                def predict(self, dp):
                    p = super().predict(dp)
                    if p is not None:
                        outs.append(p)
                    return p
            t = time.time()
            r_off = u.ScorerStepByStep(sub).score(Rec(ort_streamer(Path(td) / "m.onnx", info)))
            secs = time.time() - t
            sc = D.Scores(); diffs = []
            P_all = []
            for g in range(n_seq):
                X, Y, S = D.read_sequence(pq.ParquetFile(sub), g, scored=True)
                with torch.no_grad():
                    P = (m(torch.from_numpy(X)[None])[0] if is_recurrent(m) else m(torch.from_numpy(X)[None]))[0].numpy()
                sc.add(Y, P, S); P_all.append(P[u.WARMUP:])
            streamed = np.array(outs)
            d = float(np.abs(streamed - np.concatenate(P_all)).max())
            dwp = abs(r_off["weighted_pearson"] - sc.result()["scored"]["wp"])
            res.append(_res(f"T9 official scorer, row by row == batched ({cfg['family']})", d <= 1e-4 and dwp <= 1e-6,
                            max_pred_diff=d, score_diff=dwp, us_per_row=round(secs / (n_seq * 20000) * 1e6, 1)))
    return res


def t11_training(paths, device=None):
    from .train import set_seed, wmse
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pf = pq.ParquetFile(paths["train"])
    seqs = [D.read_sequence(pf, g) for g in range(2)]
    X = torch.from_numpy(np.stack([s[0] for s in seqs])).to(device)
    Y = torch.from_numpy(np.stack([s[1] for s in seqs])).to(device).clamp(-2, 2)
    cfg = {"family": "tcn", "channels": 32, "kernel": 3, "dilations": [1, 2, 4, 8, 16, 32], "act": "relu", "norm": "weight"}

    def fit(steps, lr, seed=0):
        set_seed(seed)
        m = build_model(cfg).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        ctx = m.receptive_field - 1
        spans = [sp for sp in D.chunk_spans(20000, 1000, ctx) if sp[2] > 99]
        losses = []
        for i in range(steps):
            a, s, e = spans[i % len(spans)]
            p = m(X[:, a:e])[:, s - a:]
            lo = max(0, 99 - s)
            loss = wmse(p[:, lo:].reshape(-1, 2), Y[:, s + lo:e].reshape(-1, 2))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        return m, losses

    def score(m):
        with torch.no_grad():
            P = m(X).cpu().numpy()
        sc = D.Scores()
        for i in range(2):
            sc.add(seqs[i][1], P[i], np.ones(20000, bool))
        return sc.result()["all"]["wp"]

    m1, _ = fit(30, 1e-3); m2, _ = fit(30, 1e-3)
    dw = max(float((a - b).abs().max()) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    res = [_res("T11a same seed twice -> same weights", dw <= 1e-6, max_weight_diff=dw, device=device)]
    set_seed(0); m0 = build_model(cfg).to(device)
    w0 = score(m0)
    m3, losses = fit(400, 2e-3)
    w3 = score(m3)
    drop = 1 - np.mean(losses[-40:]) / np.mean(losses[:40])
    res.append(_res("T11b small TCN fits 2 sequences", w3 - w0 >= 0.2 and drop >= 0.2, wp_before=round(w0, 4),
                    wp_after=round(w3, 4), loss_drop=round(float(drop), 3)))
    return res


BASELINE_MODEL = {"family": "tcn", "channels": 64, "kernel": 3, "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                  "act": "relu", "norm": "weight", "dropout": 0.0}
GRU_MODEL = {"family": "gru", "hidden": 64}


def data_fingerprint(paths):
    parts = []
    for k in sorted(paths):
        p = Path(paths[k])
        parts.append(f"{k}:{p.stat().st_size if p.exists() else 0}:{D.n_groups(p) if p.exists() else 0}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


def run_data_checks(root=None, full=False, out_dir=None, skip_training=False):
    D.official(root)
    paths = D.dataset_paths(root)
    t = time.time()
    print("== data checks ==", flush=True)
    res = t1_integrity(paths, full=full) + t2_mask(paths) + t3_t4_t5(paths)
    res += t9_official_scorer(paths, [BASELINE_MODEL, GRU_MODEL])
    if not skip_training:
        res += t11_training(paths)
    print("== model checks: baseline TCN and GRU ==", flush=True)
    mres = [check_model(BASELINE_MODEL), check_model(GRU_MODEL)]
    report = {"pass": all(r["pass"] for r in res) and all(m["pass"] for m in mres), "data_tests": res, "model_tests": mres,
              "full_scan": full, "minutes": (time.time() - t) / 60, "fingerprint": data_fingerprint(paths),
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "data_checks.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"== data checks {'PASSED' if report['pass'] else 'FAILED'} ({report['minutes']:.1f} min) ==", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--full", action="store_true", help="T1 reads every row (slow)")
    ap.add_argument("--model-config", default=None, help="check only this run config's model")
    ap.add_argument("--out", default=str(D.REPO / "runs" / "_checks"))
    a = ap.parse_args()
    if a.model_config:
        r = check_model(json.loads(Path(a.model_config).read_text())["model"])
        raise SystemExit(0 if r["pass"] else 1)
    r = run_data_checks(a.root, a.full, a.out)
    raise SystemExit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
