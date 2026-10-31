"""Final evaluation of the finalists against the GRU baseline (run after the search).

    python -m tcnsearch.final [--root ...] [--runs runs] [--lb-gap 0.027]

For every finalist (3 seeds) and the GRU (3 seeds), all with the official metric:
  * half A (used for selection), half B (untouched confirmation) and the FULL validation set,
    each on scored rows (is_scored = valid_mask) and all rows, t0 / t1 / combined
  * mean and standard deviation over seeds, parameters, receptive field, training time, convergence
  * TCN minus GRU on half B: 3-seed mean difference with a bootstrap over sequences (2,000 resamples, seed 0),
    plus a combined uncertainty that also includes seed-to-seed spread
  * 3-seed ensembles (mean prediction) per family on B and full, and a TCN+GRU blend whose weight is chosen on A
A leaderboard estimate is added ONLY if --lb-gap is given, and is labelled as an uncertain estimate.
Writes runs/_search/final_results.json and final_table.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from . import data as D
from .models import build_model
from .train import predict_groups


def load(runs, name_or_dir):
    hits = [p for p in runs.glob("*/final.json") if json.loads(p.read_text())["name"] == name_or_dir or p.parent.name == name_or_dir]
    if not hits:
        raise SystemExit(f"run {name_or_dir} not finished")
    return hits[0].parent, json.loads(hits[0].read_text())


def ensure_preds(run_dir, fin, paths, device):
    A, B = D.split_valid(D.n_groups(paths["valid"]))
    lim = fin["train"].get("limit_valid")
    if lim:
        A, B = A[:lim], B[:lim]
    for name, groups in (("A", A), ("B", B)):
        f = run_dir / f"preds_{name}.npy"
        if not f.exists():
            m = build_model(fin["model"]).to(device)
            m.load_state_dict(torch.load(run_dir / "selected.pt", map_location=device))
            predict_groups(m, paths["valid"], groups, device, fin["train"].get("eval_batch", 8), f)
    return A, B


def targets(paths, groups):
    pf = pq.ParquetFile(paths["valid"])
    Y, S = [], []
    for g in groups:
        _, y, s = D.read_sequence(pf, g, scored=True); Y.append(y); S.append(s)
    return Y, S


def score_preds(P, Y, S):
    sc = D.Scores()
    for p, y, s in zip(P, Y, S):
        sc.add(y, p.astype(np.float32), s)
    return sc.result()


def summarise(group):
    keys = [("A", "scored"), ("A", "all"), ("B", "scored"), ("B", "all"), ("full", "scored"), ("full", "all")]
    out = {}
    for split, rows in keys:
        for m in ("wp", "t0", "t1"):
            v = np.array([f[split][rows][m] for _, f in group])
            out[f"{split}_{rows}_{m}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                          "per_seed": v.round(5).tolist()}
    f0 = group[0][1]
    out |= {"name": f0["name"], "model": f0["model"], "lr": f0["train"]["lr"], "epochs": f0["train"]["epochs"],
            "params": f0["params"], "receptive_field": f0["receptive_field"], "seeds": [f["train"]["seed"] for _, f in group],
            "train_minutes_mean": float(np.mean([f["train_minutes"] for _, f in group])),
            "minutes_per_epoch": float(np.mean([f["minutes_per_epoch"] for _, f in group])),
            "still_rising_any_seed": any(f["convergence"]["still_rising"] for _, f in group),
            "skipped_updates": int(sum(f["convergence"]["skipped_updates"] for _, f in group)),
            "run_dirs": [d.name for d, _ in group]}
    return out


def bootstrap_diff(tcn_group, gru_group, rows="scored", n=2000, seed=0):
    """Half-B difference of 3-seed mean scores, resampling sequences (same resample for every run)."""
    mT = [np.load(d / "moments_B.npz")[rows] for d, _ in tcn_group]       # [n_seq, 2, 6]
    mG = [np.load(d / "moments_B.npz")[rows] for d, _ in gru_group]
    n_seq = mT[0].shape[0]
    point = np.mean([D.wp_from_moments(m.sum(0))[0] for m in mT]) - np.mean([D.wp_from_moments(m.sum(0))[0] for m in mG])
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_seq, n_seq)
        diffs[i] = np.mean([D.wp_from_moments(m[idx].sum(0))[0] for m in mT]) - np.mean([D.wp_from_moments(m[idx].sum(0))[0] for m in mG])
    sT = np.std([D.wp_from_moments(m.sum(0))[0] for m in mT], ddof=1) if len(mT) > 1 else 0.0
    sG = np.std([D.wp_from_moments(m.sum(0))[0] for m in mG], ddof=1) if len(mG) > 1 else 0.0
    se = float(np.sqrt(diffs.var() + sT ** 2 / len(mT) + sG ** 2 / len(mG)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "boot_ci95": [float(lo), float(hi)], "combined_se": se,
            "combined_ci95": [float(point - 1.96 * se), float(point + 1.96 * se)]}


def verdict(b):
    if b["diff"] >= 0.002 and b["combined_ci95"][0] > 0:
        return "TCN better (>= +0.002, interval above 0)"
    if b["diff"] <= -0.002 and b["combined_ci95"][1] < 0:
        return "TCN worse (<= -0.002, interval below 0)"
    return "no reliable difference (|diff| < 0.002 or interval includes 0)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--runs", default=str(D.REPO / "runs"))
    ap.add_argument("--lb-gap", type=float, default=None, help="optional: local-minus-leaderboard gap for an UNCERTAIN estimate")
    a = ap.parse_args()
    D.official(a.root)
    paths = D.dataset_paths(a.root)
    runs = Path(a.runs)
    fl = json.loads((runs / "_search" / "finalists.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gru = [load(runs, n) for n in fl["gru"]]
    tcns = [[load(runs, n) for n in g] for g in fl["tcn"]]
    res = {"gru": summarise(gru), "finalists": []}

    # ensembles need predictions
    A, B = ensure_preds(*gru[0], paths, device)
    for d, f in gru + [x for g in tcns for x in g]:
        ensure_preds(d, f, paths, device)
    YA, SA = targets(paths, A); YB, SB = targets(paths, B)

    def ens(group, split):
        return np.mean([np.load(d / f"preds_{split}.npy").astype(np.float32) for d, _ in group], 0)
    gA, gB = ens(gru, "A"), ens(gru, "B")
    res["gru"]["ensemble"] = {"B": score_preds(gB, YB, SB), "full": score_preds(np.concatenate([gA, gB]), YA + YB, SA + SB)}
    for g in tcns:
        s = summarise(g)
        s["vs_gru_B_scored"] = bootstrap_diff(g, gru, "scored"); s["vs_gru_B_all"] = bootstrap_diff(g, gru, "all")
        s["verdict_B_scored"] = verdict(s["vs_gru_B_scored"])
        tA, tB = ens(g, "A"), ens(g, "B")
        s["ensemble"] = {"B": score_preds(tB, YB, SB), "full": score_preds(np.concatenate([tA, tB]), YA + YB, SA + SB)}
        best_w, best = 0.0, -1
        for w in np.round(np.arange(0, 1.01, 0.1), 2):            # blend weight chosen on half A only
            v = score_preds((1 - w) * gA + w * tA, YA, SA)["scored"]["wp"]
            if v > best:
                best, best_w = v, float(w)
        s["blend_with_gru"] = {"tcn_weight_chosen_on_A": best_w,
                               "B": score_preds((1 - best_w) * gB + best_w * tB, YB, SB),
                               "full": score_preds(np.concatenate([(1 - best_w) * gA + best_w * tA, (1 - best_w) * gB + best_w * tB]),
                                                   YA + YB, SA + SB)}
        res["finalists"].append(s)
    if a.lb_gap is not None:
        res["leaderboard_estimate_UNCERTAIN"] = {
            "note": "full-validation scored score minus a fixed gap measured on earlier GRU submissions; not an official score",
            "gap": a.lb_gap, "gru": res["gru"]["full_scored_wp"]["mean"] - a.lb_gap,
            "finalists": {s["name"]: s["full_scored_wp"]["mean"] - a.lb_gap for s in res["finalists"]}}
    out = runs / "_search"
    (out / "final_results.json").write_text(json.dumps(res, indent=2, default=float))
    rows = []
    for s in [res["gru"]] + res["finalists"]:
        r = {"name": s["name"], "family": s["model"]["family"], "params": s["params"], "rf": s["receptive_field"],
             "lr": s["lr"], "epochs": s["epochs"], "seeds": len(s["seeds"])}
        for k in ("full_scored_wp", "full_all_wp", "full_scored_t0", "full_scored_t1", "B_scored_wp", "B_all_wp", "A_scored_wp"):
            r[k] = f"{s[k]['mean']:.4f} ± {s[k]['sd']:.4f}"
        r["ensemble_full_scored"] = round(s["ensemble"]["full"]["scored"]["wp"], 4)
        r["vs_gru_B"] = f"{s['vs_gru_B_scored']['diff']:+.4f} {s['vs_gru_B_scored']['combined_ci95']}" if "vs_gru_B_scored" in s else ""
        r["train_min"] = round(s["train_minutes_mean"], 1)
        rows.append(r)
    with open(out / "final_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)
    print(f"wrote {out / 'final_results.json'}")


if __name__ == "__main__":
    main()
