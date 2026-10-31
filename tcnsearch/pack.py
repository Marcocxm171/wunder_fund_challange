"""Bundle the results to send back (small: no per-epoch weights, no resume state).

    python -m tcnsearch.pack                       # -> tcn_results_<date>.zip
    python -m tcnsearch.pack --with-preds          # also the finalists' prediction files (bigger, ~150 MB each)

Included: runs/_search (summary, decisions, finalists, final results, bench), runs/_checks (all check reports),
report/ (if built), and from every run folder: config.json, env.json, log.jsonl, final.json, train.log,
failed.txt, bench.json, moments_A/B.npz and selected.pt (the chosen weights, needed to re-score or export).
"""
import argparse
import time
import zipfile
from pathlib import Path

from . import data as D

RUN_FILES = ["config.json", "env.json", "log.jsonl", "final.json", "train.log", "failed.txt", "bench.json",
             "moments_A.npz", "moments_B.npz", "selected.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(D.REPO / "runs"))
    ap.add_argument("--report", default=str(D.REPO / "report"))
    ap.add_argument("--out", default=f"tcn_results_{time.strftime('%Y%m%d_%H%M')}.zip")
    ap.add_argument("--with-preds", action="store_true")
    a = ap.parse_args()
    runs, report = Path(a.runs), Path(a.report)
    n = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("_search", "_checks"):
            for f in (runs / sub).rglob("*") if (runs / sub).exists() else []:
                if f.is_file():
                    z.write(f, Path("runs") / f.relative_to(runs)); n += 1
        for d in sorted(p for p in runs.iterdir() if p.is_dir() and not p.name.startswith("_")):
            names = RUN_FILES + (["preds_A.npy", "preds_B.npy"] if a.with_preds else [])
            for name in names:
                if (d / name).exists():
                    z.write(d / name, Path("runs") / d.name / name); n += 1
        if report.exists():
            for f in report.rglob("*"):
                if f.is_file():
                    z.write(f, Path("report") / f.relative_to(report)); n += 1
    size = Path(a.out).stat().st_size / 1e6
    print(f"wrote {a.out}: {n} files, {size:.0f} MB")


if __name__ == "__main__":
    main()
