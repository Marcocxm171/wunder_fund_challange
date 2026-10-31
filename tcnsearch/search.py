"""The full TCN search: phases, promotion rules and a multi-GPU scheduler.

    python -m tcnsearch.search --gpus 0,1,2,3                 # run everything (resumable: just run it again)
    python -m tcnsearch.search --gpus 0,1 --per-gpu 2         # two runs per GPU
    python -m tcnsearch.search --gpus 0 --until 2             # stop after phase 2 (review, then rerun with --until 3 ...)
    python -m tcnsearch.search --plan                         # print the phases and the runs known so far
    python -m tcnsearch.search --root devroot --smoke --gpus cpu   # tiny end-to-end test of the whole pipeline

Before any training: the data checks (checks.py) must pass, and every new model configuration is checked
(causality, receptive field, chunks, row-by-row == full) before its first run.

Every decision uses ONLY the half-A score of each run's selected checkpoint (final.json "A" -> "scored" -> "wp").
Half B and the full validation set are never read here; final.py reports them at the end.

A run is identified by a hash of its model + training settings, so an identical configuration requested by two
phases is trained once. Runs claim their folder atomically (several schedulers or machines can share one runs
folder on a shared disk). A crashed run is retried (resumed) up to --retries times; failures are recorded,
the search continues, and the summary lists every missing run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import data as D
from .checks import BASELINE_MODEL, GRU_MODEL, data_fingerprint, model_check_cached, run_data_checks
from .models import build_model, n_params

# ----------------------------------------------------------------------------- search constants
LR_GRID = [5e-4, 1e-3, 2e-3]
SCREEN_E = 8                 # epochs for screening runs
TIE = 0.002                  # within this of the best = tied (single-seed noise)
RISE = 0.001                 # per-epoch gain over the last 2 epochs that counts as "still improving"
WIDTHS = [32, 48, 64, 96, 128]
ACTS = ["relu", "silu", "gelu", "gated"]
NORMS = ["weight", "layer", "none"]
RF_TARGETS = [250, 500, 1000, 2000]


def p2(L):
    return [2 ** i for i in range(L)]


# (kernel, receptive-field target) -> dilations. Two convs per block: RF = 1 + 2 (k-1) sum(d).
CELLS = {
    (2, 250): p2(7), (2, 500): p2(8), (2, 1000): p2(9), (2, 2000): p2(10),
    (3, 250): p2(6), (3, 500): p2(7), (3, 1000): p2(8), (3, 2000): p2(9),
    (5, 250): p2(5), (5, 500): p2(6), (5, 1000): p2(7), (5, 2000): p2(8),
    (7, 250): [1, 2, 4, 8, 6], (7, 500): [1, 2, 4, 8, 16, 11],
    (7, 1000): [1, 2, 4, 8, 16, 32, 20], (7, 2000): [1, 2, 4, 8, 16, 32, 64, 40],
}
BASE_CELL = (3, 1000)


def tcn(cell, C=64, act="relu", norm="weight"):
    k, _ = cell
    return {"family": "tcn", "channels": C, "kernel": k, "dilations": CELLS[cell], "act": act, "norm": norm, "dropout": 0.0}


def multiscale(variant, C, act, norm):
    short = {"channels": C, "kernel": 3, "dilations": [1, 2, 4, 8]}                 # 61 rows
    mid = {"channels": C, "kernel": 3, "dilations": p2(7)}                         # 509 rows
    long = {"channels": C, "kernel": 3, "dilations": [1, 4, 16, 64, 256]}           # 1,365 rows
    br = [short, long] if variant == 2 else [short, mid, long]
    return {"family": "multiscale", "branches": br, "act": act, "norm": norm, "dropout": 0.0}


def modern(width=64, kernel=51, ratio=2.0, act="gelu", norm="layer"):
    dil = [1, 4, 15] if kernel == 51 else [1, 4, 16, 12]                            # 1,001 / 991 rows
    return {"family": "modern", "width": width, "kernel": kernel, "dilations": dil, "ffn_ratio": ratio,
            "act": act, "norm": norm, "small_kernel": 5, "dropout": 0.0}


def modern_vi(act, norm):
    return {"family": "modern_vi", "var_dim": 8, "kernel": 51, "dilations": [1, 4, 15], "ffn_ratio": 1.0,
            "act": act, "norm": norm, "small_kernel": 5, "dropout": 0.0}


def neighbours(C):
    i = WIDTHS.index(C)
    return [WIDTHS[j] for j in (i - 1, i + 1) if 0 <= j < len(WIDTHS)]


# ----------------------------------------------------------------------------- run specs and registry
HASH_KEYS = ("lr", "epochs", "seed", "batch_size", "chunk", "clip", "loss", "limit_train", "limit_valid")


class Search:
    def __init__(self, runs_dir: Path, smoke=False):
        self.dir = Path(runs_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.smoke = smoke
        self.screen_e = 2 if smoke else SCREEN_E
        self.budgets = [2, 4, 6] if smoke else [8, 16, 24]
        self.decisions: dict = {}
        self.missing: list = []

    # -- specs
    def scale(self, model):
        """--smoke: shrink widths so the whole pipeline runs on a CPU in minutes."""
        if not self.smoke:
            return model
        m = json.loads(json.dumps(model))
        for key in ("channels", "width"):
            if key in m:
                m[key] = max(4, m[key] // 8)
        if "branches" in m:
            for b in m["branches"]:
                b["channels"] = max(4, b["channels"] // 8)
        if m["family"] == "modern_vi":
            m["var_dim"] = 1
        if m["family"] == "gru":
            m["hidden"] = 8
        return m

    def spec(self, name, phase, model, lr, epochs, seed=0, save_preds=False):
        train = {"lr": lr, "epochs": epochs, "seed": seed, "batch_size": 32, "chunk": 1000, "clip": 1.0, "loss": "wmse",
                 "limit_train": None, "limit_valid": None, "save_preds": save_preds,
                 "eval_batch": 2 if model["family"] == "modern_vi" else 8}
        if self.smoke:
            train.update(batch_size=2, limit_train=4, limit_valid=2, eval_batch=2)
        model = self.scale(model)
        core = {"model": model, "train": {k: train[k] for k in HASH_KEYS}}
        h = hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:10]
        return {"name": name, "phase": phase, "model": model, "train": train, "hash": h}

    def run_dir(self, spec):
        hits = sorted(self.dir.glob(f"*_{spec['hash']}"))
        if hits:
            return hits[0]
        return self.dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', spec['name'])}_{spec['hash']}"

    def result(self, spec):
        f = self.run_dir(spec) / "final.json"
        return json.loads(f.read_text()) if f.exists() else None

    def score(self, spec):
        r = self.result(spec)
        return None if r is None else r["A"]["scored"]["wp"]

    def best(self, specs, key=None):
        scored = [(self.score(s), i, s) for i, s in enumerate(specs) if self.score(s) is not None]
        if not scored:
            raise SystemExit(f"no finished runs among: {[s['name'] for s in specs]}")
        return max(scored, key=lambda t: (t[0], -t[1]))[2]

    def ranked(self, specs):
        return sorted([s for s in specs if self.score(s) is not None], key=lambda s: -self.score(s))

    def decide(self, key, value, why):
        self.decisions[key] = {"value": value, "why": why}
        print(f"  decision {key} = {value}  ({why})", flush=True)

    # -- phases: each returns (specs to run, callback after they finished)
    def phases(self):
        """Yields (phase, step label, list of specs). Steps run in order; each may depend on earlier results."""
        E8 = self.screen_e
        # ---------- Phase 1: training calibration
        p1t = [self.spec(f"p1_tcn_lr{lr:g}", "1", BASELINE_MODEL, lr, E8) for lr in LR_GRID]
        p1g = [self.spec(f"p1_gru_lr{lr:g}", "1", GRU_MODEL, lr, E8) for lr in LR_GRID]
        yield "1", "1a learning rate (TCN and GRU, 8 epochs)", p1t + p1g
        lrT = self.best(p1t)["train"]["lr"]; lrG = self.best(p1g)["train"]["lr"]
        self.decide("lr_tcn", lrT, "best half-A score of p1_tcn_lr*")
        self.decide("lr_gru", lrG, "best half-A score of p1_gru_lr*")
        b8, b16, b24 = self.budgets
        t16 = self.spec(f"p1_tcn_e{b16}", "1", BASELINE_MODEL, lrT, b16)
        g16 = self.spec(f"p1_gru_e{b16}", "1", GRU_MODEL, lrG, b16)
        yield "1", f"1b epoch budget {b16}", [t16, g16]
        extra, budget = [], {}
        for fam, s8, s16, model, lr in (("tcn", self.best(p1t), t16, BASELINE_MODEL, lrT), ("gru", self.best(p1g), g16, GRU_MODEL, lrG)):
            gain = (self.score(s16) or -1) - self.score(s8)
            budget[fam] = (b8, gain)
            if gain >= RISE:
                extra.append((fam, self.spec(f"p1_{fam}_e{b24}", "1", model, lr, b24), s16))
        if extra:
            yield "1", f"1c epoch budget {b24} (only where {b16} still helped)", [e[1] for e in extra]
        for fam in ("tcn", "gru"):
            E, gain = budget[fam]
            why = f"{b16} vs {b8} epochs: {gain:+.4f}"
            if gain >= RISE:
                E = b16
                for f2, s24, s16 in extra:
                    if f2 == fam:
                        g24 = (self.score(s24) or -1) - self.score(s16)
                        why += f"; {b24} vs {b16}: {g24:+.4f}"
                        if g24 >= RISE:
                            E = b24
            self.decide(f"epochs_{fam}", E, why + f" (needs >= {RISE} to add epochs)")
        ET, EG = self.decisions["epochs_tcn"]["value"], self.decisions["epochs_gru"]["value"]

        # ---------- Phase 2: receptive field x kernel
        p2s = {cell: self.spec(f"p2_k{cell[0]}_rf{cell[1]}", "2", tcn(cell), lrT, E8) for cell in CELLS}
        yield "2", "2 receptive field x kernel (16 cells)", list(p2s.values())
        r2 = self.ranked(list(p2s.values()))
        cell_of = {s["hash"]: c for c, s in p2s.items()}
        top2 = [cell_of[s["hash"]] for s in r2[:2]]
        self.decide("top2_cells", [list(c) for c in top2], "two best half-A scores in phase 2")

        # ---------- Phase 3: width x top-2 cells
        p3s = {(c, C): (p2s[c] if C == 64 else self.spec(f"p3_k{c[0]}_rf{c[1]}_c{C}", "3", tcn(c, C), lrT, E8))
               for c in top2 for C in WIDTHS}
        yield "3", "3 width x two best cells", [s for (c, C), s in p3s.items() if C != 64]
        (cell, C) = max(((k, s) for k, s in p3s.items() if self.score(s) is not None), key=lambda t: self.score(t[1]))[0]
        self.decide("best_cell_width", [list(cell), C], "best half-A score in phase 3 (phase-2 C=64 runs included)")

        # ---------- Phase 4: activation x normalisation
        p4s = {(a, n): (p3s[(cell, C)] if (a, n) == ("relu", "weight") else
                        self.spec(f"p4_{a}_{n}", "4", tcn(cell, C, a, n), lrT, E8)) for a in ACTS for n in NORMS}
        yield "4", "4 activation x normalisation (12)", [s for k, s in p4s.items() if k != ("relu", "weight")]
        act, norm = max(((k, s) for k, s in p4s.items() if self.score(s) is not None), key=lambda t: self.score(t[1]))[0]
        self.decide("best_act_norm", [act, norm], "best half-A score in phase 4")
        best_tcn = tcn(cell, C, act, norm)

        # ---------- Phase 5: families
        ms = {"ms2_c32": multiscale(2, 32, act, norm), "ms2_c48": multiscale(2, 48, act, norm),
              "ms3_c32": multiscale(3, 32, act, norm), "ms3_c48": multiscale(3, 48, act, norm)}
        mo = {"mod_k51_r2": modern(64, 51, 2.0, act, norm), "mod_k51_r4": modern(64, 51, 4.0, act, norm),
              "mod_k31_r2": modern(64, 31, 2.0, act, norm), "mod_k51_r2_w96": modern(96, 51, 2.0, act, norm),
              "mod_vi": modern_vi(act, norm)}
        fams = {"multiscale": {k: self.spec(f"p5_{k}", "5", m, lrT, E8) for k, m in ms.items()},
                "modern": {k: self.spec(f"p5_{k}", "5", m, lrT, E8) for k, m in mo.items()}}
        yield "5", "5a multi-scale and ModernTCN variants", [s for f in fams.values() for s in f.values()]
        fam_best = {f: self.best(list(v.values())) for f, v in fams.items()}
        lr_runs = {f: [self.spec(f"{s['name']}_lr{lr:g}", "5", s["model"], lr, E8) for lr in LR_GRID if lr != lrT]
                   for f, s in fam_best.items()}
        yield "5", "5b learning rate for each family's best variant", [s for v in lr_runs.values() for s in v]
        full5 = []
        for f, s in fam_best.items():
            b = self.best([s] + lr_runs[f])
            self.decide(f"best_{f}", {"run": b["name"], "lr": b["train"]["lr"]}, "best half-A score in 5a/5b")
            full5.append(self.spec(f"p5_{f}_best_e{ET}", "5", b["model"], b["train"]["lr"], ET, save_preds=True))
        yield "5", f"5c each family's best at {ET} epochs", full5

        # ---------- Phase 6: interaction checks
        p6 = []
        other = [c for c in top2 if c != cell]
        if other:
            p6.append(self.spec(f"p6_{act}_{norm}_k{other[0][0]}_rf{other[0][1]}_c{C}", "6", tcn(other[0], C, act, norm), lrT, E8))
        for C2 in neighbours(C):
            p6.append(self.spec(f"p6_{act}_{norm}_k{cell[0]}_rf{cell[1]}_c{C2}", "6", tcn(cell, C2, act, norm), lrT, E8))
        if act == "gated" or norm == "none":
            for f in (0.5, 2.0):
                p6.append(self.spec(f"p6_best_lr{lrT * f:g}", "6", best_tcn, lrT * f, E8))
        if act == "gated":                                     # parameter-matched gated model
            target = n_params(build_model(tcn(cell, C, "relu", norm)))
            C3 = min(range(16, 129, 8), key=lambda c: abs(n_params(build_model(tcn(cell, c, act, norm))) - target))
            if C3 != C:
                p6.append(self.spec(f"p6_gated_parammatched_c{C3}", "6", tcn(cell, C3, act, norm), lrT, E8))
        yield "6", "6 interaction checks", p6

        # ---------- Phase 7: promotion and finals
        screens = [s for s in self.all_specs if s["train"]["epochs"] == E8 and s["model"]["family"] != "gru"
                   and self.score(s) is not None]
        by_model: dict = {}
        for s in screens:                                      # best learning rate per model configuration
            k = json.dumps(s["model"], sort_keys=True)
            if k not in by_model or self.score(s) > self.score(by_model[k]):
                by_model[k] = s
        pool = sorted(by_model.values(), key=lambda s: -self.score(s))
        top6 = pool[:6]
        rising = []
        for ph in "123456":
            rs = [s for s in pool[6:] if s["phase"] == ph and (self.result(s) or {}).get("convergence", {}).get("still_rising")]
            rising += rs[:2]
        self.decide("promoted", [s["name"] for s in top6] + [s["name"] + " (still rising)" for s in rising],
                    "top 6 screens by half-A score + up to 2 still-improving runs per phase")
        p7a = [self.spec(f"{s['name']}_e{ET}", "7", s["model"], s["train"]["lr"], ET, save_preds=True) for s in top6 + rising]
        yield "7", f"7a promoted runs at {ET} epochs", p7a
        fulls = [s for s in self.all_specs if s["train"]["epochs"] == ET and s["model"]["family"] != "gru"
                 and s["train"]["seed"] == 0 and self.score(s) is not None]
        top2f = self.ranked(fulls)[:2]
        p7b = [self.spec(f"{s['name']}_lr{s['train']['lr'] * f:g}", "7", s["model"], s["train"]["lr"] * f, ET, save_preds=True)
               for s in top2f for f in (0.5, 2.0)]
        yield "7", "7b learning rate x0.5 / x2 for the top 2", p7b
        fulls = [s for s in self.all_specs if s["train"]["epochs"] == ET and s["model"]["family"] != "gru"
                 and s["train"]["seed"] == 0 and self.score(s) is not None]
        by_model = {}
        for s in self.ranked(fulls):
            by_model.setdefault(json.dumps(s["model"], sort_keys=True), s)
        finalists = list(by_model.values())[:3]
        gru_final = self.best([s for s in self.all_specs if s["model"]["family"] == "gru" and s["train"]["epochs"] == EG
                               and s["train"]["lr"] == lrG] or p1g)
        self.decide("finalists", [{"name": s["name"], "lr": s["train"]["lr"]} for s in finalists],
                    "top 3 distinct models at full epochs (seed 0), half-A")
        self.decide("gru_final", {"name": gru_final["name"], "lr": gru_final["train"]["lr"], "epochs": gru_final["train"]["epochs"]},
                    "GRU at its best learning rate and epoch budget")
        p7c = [self.spec(f"final_{s['name']}_s{sd}", "7", s["model"], s["train"]["lr"], ET, seed=sd, save_preds=True)
               for s in finalists for sd in (1, 2)]
        p7d = [self.spec(f"final_gru_s{sd}", "7", GRU_MODEL, gru_final["train"]["lr"], gru_final["train"]["epochs"], seed=sd, save_preds=True)
               for sd in (1, 2)]
        yield "7", "7c/7d three seeds for the finalists and the GRU", p7c + p7d
        self.finalists = {"tcn": [[s] + [x for x in p7c if x["model"] == s["model"]] for s in finalists],
                          "gru": [gru_final] + p7d}

    # -- bookkeeping
    def write_summary(self):
        rows = []
        for f in sorted(self.dir.glob("*/final.json")):
            r = json.loads(f.read_text())
            rows.append({"run_id": r["run_id"], "phase": r["phase"], "name": r["name"], "family": r["model"]["family"],
                         "params": r["params"], "receptive_field": r["receptive_field"], "lr": r["train"]["lr"],
                         "epochs": r["train"]["epochs"], "seed": r["train"]["seed"],
                         "A_scored_wp": round(r["A"]["scored"]["wp"], 5), "A_scored_t0": round(r["A"]["scored"]["t0"], 5),
                         "A_scored_t1": round(r["A"]["scored"]["t1"], 5), "A_all_wp": round(r["A"]["all"]["wp"], 5),
                         "selected": r["selected"], "minutes_per_epoch": round(r["minutes_per_epoch"], 2),
                         "train_minutes": round(r["train_minutes"], 1),
                         "still_rising": r["convergence"]["still_rising"], "skipped_updates": r["convergence"]["skipped_updates"]})
        out = self.dir / "_search"; out.mkdir(exist_ok=True)
        if rows:
            with open(out / "summary_A.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (out / "decisions.json").write_text(json.dumps(self.decisions, indent=2))
        (out / "missing_runs.json").write_text(json.dumps(self.missing, indent=2))


# ----------------------------------------------------------------------------- scheduler
class Scheduler:
    def __init__(self, search: Search, gpus, per_gpu, root, retries, stale_minutes=60):
        self.s, self.root, self.retries, self.stale = search, root, retries, stale_minutes * 60
        self.slots = [g for g in gpus for _ in range(per_gpu)]

    def _claim(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(d / "claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.uname().nodename if hasattr(os, 'uname') else ''} {os.getpid()} {time.ctime()}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            hb = d / "heartbeat"
            age = time.time() - (hb.stat().st_mtime if hb.exists() else (d / "claim").stat().st_mtime)
            if age > self.stale:                               # owner died: take over
                (d / "claim").write_text(f"taken over {time.ctime()}")
                return True
            return False

    def run(self, specs):
        todo = []
        for sp in specs:
            d = self.s.run_dir(sp)
            if (d / "final.json").exists():
                continue
            todo.append(sp)
        if not todo:
            return
        for sp in todo:                                        # model checks before the first run of each config
            r = model_check_cached(sp["model"], self.s.dir / "_checks" / "models")
            if not r["pass"]:
                raise SystemExit(f"model checks FAILED for {sp['name']}: see runs/_checks/models/. Not training.")
        running, attempts, waiting = {}, {}, []
        free = list(self.slots)
        queue = list(todo)
        while queue or running or waiting:
            while queue and free:
                sp = queue.pop(0)
                d = self.s.run_dir(sp)
                if (d / "final.json").exists():
                    continue
                if not self._claim(d):
                    waiting.append(sp); continue
                cfg = {"name": sp["name"], "phase": sp["phase"], "model": sp["model"], "train": sp["train"]}
                (d / "config.json").write_text(json.dumps(cfg, indent=2))
                slot = free.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if slot == "cpu" else str(slot), PYTHONUNBUFFERED="1")
                cmd = [sys.executable, "-m", "tcnsearch.train", "--config", str(d / "config.json")]
                if self.root:
                    cmd += ["--root", str(self.root)]
                if (d / "state.pt").exists():
                    cmd.append("--resume")
                logf = open(d / "train.log", "a")
                p = subprocess.Popen(cmd, cwd=D.REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
                attempts[sp["hash"]] = attempts.get(sp["hash"], 0) + 1
                running[p] = (sp, slot, logf, time.time())
                print(f"  start {d.name} on {'cpu' if slot == 'cpu' else 'GPU ' + str(slot)} (attempt {attempts[sp['hash']]})", flush=True)
            time.sleep(5)
            for p in list(running):
                if p.poll() is None:
                    continue
                sp, slot, logf, t0 = running.pop(p); logf.close(); free.append(slot)
                d = self.s.run_dir(sp)
                (d / "claim").unlink(missing_ok=True)
                if (d / "final.json").exists():
                    print(f"  done  {d.name}: half-A {self.s.score(sp):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                elif attempts[sp["hash"]] <= self.retries:
                    print(f"  FAILED {d.name} (exit {p.returncode}), retrying; see {d / 'train.log'}", flush=True)
                    queue.append(sp)
                else:
                    print(f"  FAILED {d.name} (exit {p.returncode}), giving up; see {d / 'train.log'}", flush=True)
                    self.s.missing.append({"name": sp["name"], "phase": sp["phase"], "run_dir": d.name, "reason": f"exit {p.returncode}"})
            for sp in list(waiting):                           # runs owned by another scheduler
                d = self.s.run_dir(sp)
                if (d / "final.json").exists():
                    waiting.remove(sp)
                elif self._claim(d):
                    (d / "claim").unlink(missing_ok=True); waiting.remove(sp); queue.append(sp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="starter pack folder (utils.py + datasets/)")
    ap.add_argument("--runs", default=str(D.REPO / "runs"))
    ap.add_argument("--gpus", default="0", help="comma list of GPU ids, or 'cpu'")
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--until", default="7", help="stop after this phase (1-7)")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="tiny models and data: tests the whole pipeline")
    ap.add_argument("--full-checks", action="store_true", help="data checks read every row")
    ap.add_argument("--plan", action="store_true", help="print the phases and finished/pending runs, train nothing")
    a = ap.parse_args()
    D.official(a.root)
    paths = D.dataset_paths(a.root)
    search = Search(Path(a.runs), smoke=a.smoke)
    gpus = ["cpu"] if a.gpus == "cpu" else [int(g) for g in a.gpus.split(",")]

    if not a.plan:
        ck = search.dir / "_checks" / "data_checks.json"
        fp = data_fingerprint(paths)
        prev = json.loads(ck.read_text()) if ck.exists() else None
        if not (prev and prev["pass"] and prev["fingerprint"] == fp and (prev["full_scan"] or not a.full_checks)):
            rep = run_data_checks(a.root, full=a.full_checks, out_dir=ck.parent)
            if not rep["pass"]:
                raise SystemExit("data checks FAILED: see runs/_checks/data_checks.json. Not training.")
        else:
            print(f"data checks already passed ({prev['time']}), skipping", flush=True)

    sched = Scheduler(search, gpus, a.per_gpu, a.root, a.retries)
    search.all_specs = []
    t0 = time.time()
    gen = search.phases()
    try:
        for phase, label, specs in gen:
            if int(phase) > int(a.until):
                print(f"\nstopping before phase {phase} (--until {a.until}). Rerun with a higher --until to continue.")
                break
            search.all_specs += specs
            done = sum((search.run_dir(s) / "final.json").exists() for s in specs)
            print(f"\n===== phase {label}: {len(specs)} runs ({done} already done) =====", flush=True)
            if a.plan:
                for s in specs:
                    st = "done" if (search.run_dir(s) / "final.json").exists() else "pending"
                    print(f"  {st:7s} {s['name']:34s} lr {s['train']['lr']:<7g} epochs {s['train']['epochs']:<3} seed {s['train']['seed']}  {s['model']}")
                if done < len(specs):
                    print("  (later phases depend on these results)")
                    break
                continue
            sched.run(specs)
            search.write_summary()
    except SystemExit as e:
        search.write_summary()
        if a.plan and "no finished runs" in str(e):
            print("  (later phases depend on these results)")
            return
        raise
    search.write_summary()
    if hasattr(search, "finalists"):
        rd = lambda s: search.run_dir(s).name  # noqa: E731  (folder names: a reused run keeps its first name)
        (search.dir / "_search" / "finalists.json").write_text(json.dumps(
            {"tcn": [[rd(x) for x in g] for g in search.finalists["tcn"]], "gru": [rd(x) for x in search.finalists["gru"]]}, indent=2))
        print("\nSearch finished. Next: python -m tcnsearch.final   (then: python -m tcnsearch.bench on an idle machine)")
    print(f"total wall time {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
