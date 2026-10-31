"""Build the experiment report from the run folders (no training, no data needed).

    python -m tcnsearch.report [--runs runs] [--out report]

Writes report/REPORT.md, report/all_runs.csv and report/figures/*.png:
  * every run: configuration, parameters, receptive field, half-A scores (scored/all, t0/t1), epochs, time,
    convergence (still improving?), skipped updates
  * training curves (half-A monitor and training sequences) per phase
  * phase 2 heatmap (kernel x receptive field), phase 3 width curves, phase 4 heatmap (activation x normalisation)
  * ranking of the strongest configurations (half A)
  * decisions taken by the search and the rule behind each one, missing / failed runs
  * if final.py has run: finalists vs the GRU on half B and the full validation set, ensembles, blend
  * if bench.py has run: CPU speed
Half-B / full-validation numbers appear only in the finalists section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data as D  # noqa: E402


def load_runs(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs.glob("*/final.json")):
        r = json.loads(f.read_text()); m = r["model"]
        rows.append({"run_id": r["run_id"], "phase": r["phase"], "name": r["name"], "family": m["family"],
                     "kernel": m.get("kernel"), "channels": m.get("channels", m.get("width", m.get("hidden"))),
                     "act": m.get("act"), "norm": m.get("norm"), "params": r["params"], "rf": r["receptive_field"],
                     "lr": r["train"]["lr"], "epochs": r["train"]["epochs"], "seed": r["train"]["seed"],
                     "A_scored": r["A"]["scored"]["wp"], "A_t0": r["A"]["scored"]["t0"], "A_t1": r["A"]["scored"]["t1"],
                     "A_all": r["A"]["all"]["wp"], "selected": r["selected"], "min_per_epoch": r["minutes_per_epoch"],
                     "train_min": r["train_minutes"], "still_rising": r["convergence"]["still_rising"],
                     "slope": r["convergence"]["monA_last_slope_per_epoch"],
                     "skipped": r["convergence"]["skipped_updates"], "model_json": json.dumps(m)})
    return pd.DataFrame(rows)


def curves(runs: Path, df: pd.DataFrame, figs: Path):
    out = []
    for ph, g in df.groupby("phase"):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for _, r in g.sort_values("A_scored", ascending=False).iterrows():
            log = [json.loads(l) for l in open(runs / r.run_id / "log.jsonl")]
            ep = [x["epoch"] for x in log]
            ax[0].plot(ep, [x["monA_scored"]["wp"] for x in log], marker=".", lw=1, label=r["name"][:28])
            ax[1].plot(ep, [x["train_all"]["wp"] for x in log], marker=".", lw=1)
        ax[0].set_title(f"phase {ph}: half-A monitor (scored rows)"); ax[1].set_title("training sequences (all rows)")
        for a in ax:
            a.set_xlabel("epoch"); a.grid(alpha=.3)
        ax[0].set_ylabel("weighted Pearson")
        if len(g) <= 16:
            ax[0].legend(fontsize=6, loc="lower right")
        fig.tight_layout(); p = figs / f"curves_phase{ph}.png"; fig.savefig(p, dpi=110); plt.close(fig)
        out.append(p.name)
    return out


def heatmap(df, rows, cols, row_vals, col_vals, title, path):
    M = np.full((len(row_vals), len(col_vals)), np.nan)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            v = df[(df[rows] == rv) & (df[cols] == cv)]["A_scored"]
            if len(v):
                M[i, j] = v.max()
    fig, ax = plt.subplots(figsize=(1.6 * len(col_vals) + 2, 0.7 * len(row_vals) + 1.8))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(col_vals)), [str(c) for c in col_vals]); ax.set_yticks(range(len(row_vals)), [str(r) for r in row_vals])
    ax.set_xlabel(cols); ax.set_ylabel(rows); ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.8); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def rf_target(rf):
    return min([250, 500, 1000, 2000], key=lambda t: abs(t - rf)) if rf else None


def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c](v) if c in fmt and pd.notna(v) else ("" if (isinstance(v, float) and np.isnan(v)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build(runs: Path, out: Path):
    figs = out / "figures"; figs.mkdir(parents=True, exist_ok=True)
    df = load_runs(runs)
    if df.empty:
        raise SystemExit("no finished runs")
    df.drop(columns=["model_json"]).to_csv(out / "all_runs.csv", index=False)
    f4 = {c: (lambda v: f"{v:.4f}") for c in ("A_scored", "A_t0", "A_t1", "A_all")}
    f4 |= {"lr": lambda v: f"{v:g}", "min_per_epoch": lambda v: f"{v:.1f}", "params": lambda v: f"{int(v):,}"}
    md = ["# TCN search report", "",
          "All scores in the run tables are on validation **half A** (development set; used for every decision), "
          "official metric, selected checkpoint. Half B and full validation appear only in the finalists section.", ""]

    curve_files = curves(runs, df, figs)
    tc = df[df.family == "tcn"].copy(); tc["rf_target"] = tc.rf.map(rf_target)
    p2 = tc[(tc.phase == "2")]
    if len(p2):
        heatmap(p2, "kernel", "rf_target", [2, 3, 5, 7], [250, 500, 1000, 2000], "phase 2: half-A score", figs / "phase2_kernel_rf.png")
    dec_f = runs / "_search" / "decisions.json"
    dec = json.loads(dec_f.read_text()) if dec_f.exists() else {}
    if "best_cell_width" in dec and (tc.phase == "4").any():
        (k, rft), C = dec["best_cell_width"]["value"]
        screen_e = tc[tc.phase == "4"].epochs.min()
        sel = tc[(tc.kernel == k) & (tc.rf_target == rft) & (tc.channels == C) & (tc.epochs == screen_e)
                 & tc.phase.isin(["2", "3", "4"])]
        if len(sel):
            heatmap(sel, "act", "norm", ["relu", "silu", "gelu", "gated"], ["weight", "layer", "none"],
                    "phase 4: activation x normalisation", figs / "phase4_act_norm.png")
    p3 = tc[tc.phase.isin(["2", "3"])]
    if (tc.phase == "3").any():
        fig, ax = plt.subplots(figsize=(6, 4))
        for (k, rft), g in p3.groupby(["kernel", "rf_target"]):
            if len(g) > 1:
                g = g.sort_values("channels"); ax.plot(g.channels, g.A_scored, marker="o", label=f"k{k} rf{rft}")
        ax.set_xlabel("channels"); ax.set_ylabel("half-A score"); ax.set_title("phase 3: width"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(figs / "phase3_width.png", dpi=110); plt.close(fig)

    md += ["## Decisions taken by the search", ""]
    for k, v in dec.items():
        md.append(f"- **{k}** = `{json.dumps(v['value'])}` — {v['why']}")
    miss_f = runs / "_search" / "missing_runs.json"
    miss = json.loads(miss_f.read_text()) if miss_f.exists() else []
    md += ["", "## Missing or failed runs", "", "None." if not miss else "\n".join(f"- {m}" for m in miss), ""]

    md += ["## Ranking (half A, best 20 runs, all families)", ""]
    top = df.sort_values("A_scored", ascending=False).head(20)
    md.append(md_table(top, ["name", "family", "params", "rf", "lr", "epochs", "seed", "A_scored", "A_t0", "A_t1", "A_all", "still_rising"], f4))
    gru = df[df.family == "gru"]
    if len(gru):
        md += ["", f"GRU runs (half A): best {gru.A_scored.max():.4f}.", ""]

    md += ["## Figures", ""]
    for f in ["phase2_kernel_rf.png", "phase3_width.png", "phase4_act_norm.png"] + curve_files:
        if (figs / f).exists():
            md.append(f"![{f}](figures/{f})")
    md.append("")

    fr = runs / "_search" / "final_results.json"
    if fr.exists():
        R = json.loads(fr.read_text())
        md += ["## Finalists vs GRU baseline (3 seeds each, official metric)", "",
               "| model | params | RF | full scored | full all rows | full t0 | full t1 | B scored | B all rows | 3-seed ensemble (full scored) | TCN − GRU on B (scored) | verdict |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in [R["gru"]] + R["finalists"]:
            ms = lambda k: f"{s[k]['mean']:.4f} ± {s[k]['sd']:.4f}"  # noqa: E731
            vs = s.get("vs_gru_B_scored")
            md.append(f"| {s['name']} | {s['params']:,} | {s['receptive_field']} | {ms('full_scored_wp')} | {ms('full_all_wp')} | "
                      f"{ms('full_scored_t0')} | {ms('full_scored_t1')} | {ms('B_scored_wp')} | {ms('B_all_wp')} | "
                      f"{s['ensemble']['full']['scored']['wp']:.4f} | "
                      f"{'' if not vs else f'{vs['diff']:+.4f} [{vs['combined_ci95'][0]:+.4f}, {vs['combined_ci95'][1]:+.4f}]'} | "
                      f"{s.get('verdict_B_scored', 'baseline')} |")
        md.append("")
        for s in R["finalists"]:
            b = s["blend_with_gru"]
            md.append(f"- {s['name']}: blend with GRU ensemble, TCN weight {b['tcn_weight_chosen_on_A']} (chosen on A): "
                      f"B scored {b['B']['scored']['wp']:.4f}, full scored {b['full']['scored']['wp']:.4f} "
                      f"(GRU ensemble alone: B {R['gru']['ensemble']['B']['scored']['wp']:.4f}, full {R['gru']['ensemble']['full']['scored']['wp']:.4f})")
        if "leaderboard_estimate_UNCERTAIN" in R:
            L = R["leaderboard_estimate_UNCERTAIN"]
            md += ["", f"**Leaderboard estimate — UNCERTAIN, not an official score** ({L['note']}; gap {L['gap']}): "
                   f"GRU {L['gru']:.4f}; " + "; ".join(f"{k} {v:.4f}" for k, v in L["finalists"].items())]
        md.append("")
    bj = runs / "_search" / "bench.json"
    if bj.exists():
        Bn = json.loads(bj.read_text())
        md += ["## CPU inference (1 thread, row by row; platform minutes are an estimate)", "",
               f"Reference 3 × GRU-64 = {Bn['reference_us_per_row']:.1f} µs/row ≙ {Bn['ref_minutes']:.0f} min.", "",
               "| model | params | alone µs/row | alone min (est.) | + 1 GRU min (est.) | + 2 GRU min (est.) |", "|---|---|---|---|---|---|"]
        for r in Bn["runs"].values():
            md.append(f"| {r['name']} | {r['params']:,} | {r['alone_us_per_row']:.1f} | {r['alone_platform_min_ESTIMATE']:.1f} | "
                      f"{r.get('with_1_gru_platform_min_ESTIMATE', float('nan')):.1f} | {r.get('with_2_gru_platform_min_ESTIMATE', float('nan')):.1f} |")
        md.append("")

    md += ["## All runs (half A)", ""]
    md.append(md_table(df.sort_values(["phase", "A_scored"], ascending=[True, False]),
                       ["phase", "name", "family", "kernel", "channels", "act", "norm", "params", "rf", "lr", "epochs", "seed",
                        "A_scored", "A_t0", "A_t1", "A_all", "selected", "min_per_epoch", "still_rising", "skipped"], f4))
    (out / "REPORT.md").write_text("\n".join(md))
    print(f"wrote {out / 'REPORT.md'} ({len(df)} runs)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(D.REPO / "runs"))
    ap.add_argument("--out", default=str(D.REPO / "report"))
    a = ap.parse_args()
    build(Path(a.runs), Path(a.out))


if __name__ == "__main__":
    main()
