"""Train one run from its config file and write everything the report needs into the run folder.

    python -m tcnsearch.train --config runs/<run_id>/config.json [--root <starter pack>] [--resume]

Same recipe for every family (the GRU baseline included):
  loss       wmse (squared error weighted by |target|) on rows >= 99, targets clipped to +-2
  optimiser  Adam, no weight decay, gradient clipping 1.0, cosine learning rate over the epochs (stepped per epoch)
  batches    32 sequences x chunks of 1,000 rows -> about 6,600 updates per epoch for every family.
             Convolutional models also get receptive_field - 1 earlier rows as context (no loss on them), so each
             prediction sees exactly the past it sees on the full sequence. The GRU carries its state across chunks.
  data order numpy seed = run seed, identical for every family at the same seed.
Monitoring per epoch: 300 fixed half-A sequences (scored and all rows, t0/t1) and 100 fixed training sequences.
Checkpoints: every epoch; candidates = last epoch and the average of the second half of the epochs ("swa").
The candidate with the higher half-A scored score is selected; only then is it scored on half B and on the
full validation set. Half B is never used for any choice.

Outputs in the run folder: config.json, env.json, log.jsonl, epoch_XX.pt, state.pt (resume), selected.pt,
final.json, moments_A.npz / moments_B.npz (per-sequence statistics for bootstraps), preds_A/B.npy (if save_preds).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import data as D
from .models import build_model, is_recurrent, n_params

DEFAULT_TRAIN = dict(lr=1e-3, epochs=8, seed=0, batch_size=32, chunk=1000, clip=1.0, loss="wmse",
                     save_preds=False, limit_train=None, limit_valid=None, eval_batch=8)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def env_info(paths):
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=D.REPO, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    files = {}
    for k, p in paths.items():
        if Path(p).exists():
            st = Path(p).stat()
            files[k] = {"bytes": st.st_size, "row_groups": D.n_groups(p)}
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "platform": platform.platform(), "host": socket.gethostname(),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "data": files, "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def group_key(name):
    parts = name.split(".")
    if "blocks" in parts:
        i = parts.index("blocks")
        return ".".join(parts[:i + 2])
    return parts[0]


def wmse(p, y):
    w = y.abs()
    return (w * (p - y) ** 2).sum() / w.sum().clamp_min(1e-8)


# ----------------------------------------------------------------------------- forward helpers
def predict_full(model, X):
    """Whole sequences at once: X [B,T,112] -> [B,T,2]."""
    return model(X)[0] if is_recurrent(model) else model(X)


@torch.no_grad()
def evaluate(models, path, groups, device, batch, scored=True):
    """Score several models on the same sequences in one pass. Returns ({name: Scores}, {name: preds or None})."""
    for m in models.values():
        m.eval()
    sc = {k: D.Scores() for k in models}
    for X, Y, S, gs in D.batches(path, groups, batch, scored=scored):
        Xt = torch.from_numpy(X).to(device)
        if S is None:
            S = np.ones(Y.shape[:2], bool)
        for k, m in models.items():
            P = predict_full(m, Xt).float().cpu().numpy()
            for i in range(len(X)):
                sc[k].add(Y[i], P[i], S[i], gs[i])
    return sc


@torch.no_grad()
def predict_groups(model, path, groups, device, batch, out_file):
    model.eval()
    arr = np.lib.format.open_memmap(out_file, mode="w+", dtype=np.float16, shape=(len(groups), 20000, 2))
    i = 0
    for X, Y, S, gs in D.batches(path, groups, batch, scored=False):
        P = predict_full(model, torch.from_numpy(X).to(device)).float().cpu().numpy()
        arr[i:i + len(X)] = P.astype(np.float16); i += len(X)
    arr.flush()


# ----------------------------------------------------------------------------- one epoch
def train_epoch(model, opt, path, groups, tc, device, ctx, hb):
    u = D.official()
    model.train()
    need = torch.zeros(u.SEQUENCE_LENGTH, dtype=torch.bool, device=device); need[u.WARMUP:] = True
    losses, norms, skipped, clipped, n_upd = [], [], 0, 0, 0
    block_norms: dict = {}
    rec = is_recurrent(model)
    t0 = time.time()
    splits = tc.get("micro_splits", 1)

    def forward_backward(X, Y, a, s, e, m, state, n_split):
        """Loss of one chunk; with n_split > 1 the batch is processed in slices whose losses are normalised by the
        WHOLE chunk's weight, so the summed gradient equals the full-batch gradient (only memory changes)."""
        Ym = Y[:, s:e][:, m]
        denom = Ym.abs().sum().clamp_min(1e-8) if tc["loss"] == "wmse" else torch.tensor(float(Ym.numel()), device=device)
        B = X.shape[0]
        edges = np.linspace(0, B, n_split + 1).astype(int)
        total, new_state = 0.0, []
        for i0, i1 in zip(edges[:-1], edges[1:]):
            if i1 <= i0:
                continue
            if rec:
                st = None if state is None else tuple(h[:, i0:i1] for h in state)
                pred, st = model(X[i0:i1, s:e], st); new_state.append(st)
            else:
                pred = model(X[i0:i1, a:e])[:, s - a:]
            p, y = pred[:, m].reshape(-1, 2), Y[i0:i1, s:e][:, m].reshape(-1, 2)
            err = (p - y) ** 2
            part = ((y.abs() * err).sum() if tc["loss"] == "wmse" else err.sum()) / denom
            if torch.isfinite(part):
                part.backward()
            total = total + part.detach()
        if rec:
            new_state = tuple(torch.cat([st[j] for st in new_state], 1) for j in range(2))
        return total, (new_state if rec else None)

    for b, (X, Y, _, _) in enumerate(D.batches(path, groups, tc["batch_size"])):
        X = torch.from_numpy(X).to(device); Y = torch.from_numpy(Y).to(device).clamp(-2, 2)
        state = None
        for a, s, e in D.chunk_spans(u.SEQUENCE_LENGTH, tc["chunk"], ctx):
            m = need[s:e]
            if not m.any():                                      # warm-up-only chunk: GRU still needs the state
                if rec:
                    with torch.no_grad():
                        _, state = model(X[:, s:e], state)
                continue
            while True:
                opt.zero_grad(set_to_none=True)
                try:
                    loss, new_state = forward_backward(X, Y, a, s, e, m, state, splits)
                    break
                except torch.cuda.OutOfMemoryError:
                    if splits >= X.shape[0]:
                        raise
                    splits *= 2
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    print(f"  out of GPU memory: now {splits} micro-batches per update (same gradient)", flush=True)
            if rec:                                              # carry the values, cut the gradient
                state = tuple(h.detach() for h in new_state)
            if torch.isfinite(loss):
                if n_upd % 50 == 0:                              # per-block gradient norms (sampled)
                    acc: dict = {}
                    for name, prm in model.named_parameters():
                        if prm.grad is not None:
                            acc[group_key(name)] = acc.get(group_key(name), 0.0) + float(prm.grad.detach().pow(2).sum())
                    for k, v in acc.items():
                        block_norms.setdefault(k, []).append(math.sqrt(v))
                norm = float(nn.utils.clip_grad_norm_(model.parameters(), tc["clip"]))
                if math.isfinite(norm):
                    opt.step(); losses.append(float(loss)); norms.append(norm); clipped += norm > tc["clip"]
                else:
                    skipped += 1
            else:
                skipped += 1
            n_upd += 1
        if b % 20 == 0:
            done = min((b + 1) * tc["batch_size"], len(groups))
            print(f"  batch {b:4d} | seqs {done}/{len(groups)} | loss {np.mean(losses[-100:]) if losses else float('nan'):.4f} | "
                  f"skipped {skipped} | {time.time() - t0:.0f}s", flush=True)
            hb()
    tc["micro_splits"] = splits                                  # keep a split found necessary for later epochs
    return {"train_loss": float(np.mean(losses)) if losses else float("nan"), "updates": n_upd, "skipped_updates": skipped,
            "micro_splits": splits,
            "grad_norm_mean": float(np.mean(norms)) if norms else float("nan"),
            "grad_norm_max": float(np.max(norms)) if norms else float("nan"),
            "clipped_frac": clipped / max(1, len(norms)),
            "block_grad_norms": {k: float(np.mean(v)) for k, v in block_norms.items()}}


# ----------------------------------------------------------------------------- main
def run(cfg_path, root=None, resume=False):
    cfg_path = Path(cfg_path)
    out = cfg_path.parent
    cfg = json.loads(cfg_path.read_text())
    tc = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    D.official(root)
    paths = D.dataset_paths(root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_tr, n_va = D.n_groups(paths["train"]), D.n_groups(paths["valid"])
    A, B = D.split_valid(n_va)
    monA, monT = D.monitor_groups(n_tr, n_va)
    if tc["limit_valid"]:                                     # smoke tests only
        A, B, monA = A[:tc["limit_valid"]], B[:tc["limit_valid"]], monA[:tc["limit_valid"]]
        monT = monT[:tc["limit_valid"]]
    (out / "env.json").write_text(json.dumps(env_info(paths), indent=2))
    hb_file = out / "heartbeat"
    hb = lambda: hb_file.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))  # noqa: E731
    hb()

    set_seed(tc["seed"])
    model = build_model(cfg["model"]).to(device)
    ctx = 0 if is_recurrent(model) else model.receptive_field - 1
    rf = None if is_recurrent(model) else model.receptive_field
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    rng = np.random.default_rng(tc["seed"])
    train_groups = rng.permutation(n_tr)
    if tc["limit_train"]:
        train_groups = train_groups[:tc["limit_train"]]
    swa_from = tc["epochs"] // 2 + 1
    swa, n_swa, start, train_min = None, 0, 1, 0.0
    log_path = out / "log.jsonl"
    if resume and (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
        swa, n_swa, start, train_min = st["swa"], st["n_swa"], st["epoch"] + 1, st["train_min"]
        train_groups = st["train_groups"]
        print(f"resumed after epoch {st['epoch']}")
    elif log_path.exists():
        log_path.unlink()
    print(f"run {out.name} | device {device} | {cfg['model']} | params {n_params(model):,} | receptive field {rf} | "
          f"lr {tc['lr']} epochs {tc['epochs']} seed {tc['seed']}", flush=True)

    for epoch in range(start, tc["epochs"] + 1):
        rng.shuffle(train_groups)
        t = time.time()
        stats = train_epoch(model, opt, paths["train"], train_groups, tc, device, ctx, hb)
        t_train = (time.time() - t) / 60
        lr_used = opt.param_groups[0]["lr"]
        sched.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise SystemExit("weights became non-finite")
        t = time.time()
        ra = evaluate({"m": model}, paths["valid"], monA, device, tc["eval_batch"])["m"].result()
        rt = evaluate({"m": model}, paths["train"], monT, device, tc["eval_batch"], scored=False)["m"].result()
        t_eval = (time.time() - t) / 60
        train_min += t_train
        rec = {"epoch": epoch, "lr": lr_used, **{k: v for k, v in stats.items()},
               "monA_scored": ra["scored"], "monA_all": ra["all"], "train_all": rt["all"],
               "train_minutes": t_train, "eval_minutes": t_eval,
               "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0}
        print(json.dumps({k: rec[k] for k in ("epoch", "train_loss", "skipped_updates", "grad_norm_mean")} |
                         {"monA_wp": round(ra["scored"]["wp"], 4), "train_wp": round(rt["all"]["wp"], 4),
                          "min": round(t_train, 1)}), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(sd, out / f"epoch_{epoch:02d}.pt")
        if epoch >= swa_from:
            if swa is None:
                swa, n_swa = {k: v.float().clone() for k, v in sd.items()}, 1
            else:
                n_swa += 1
                for k in swa:
                    swa[k] += (sd[k].float() - swa[k]) / n_swa
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "swa": swa, "n_swa": n_swa,
                    "epoch": epoch, "train_min": train_min, "train_groups": train_groups}, out / "state.pt")
        hb()

    # ---- checkpoint choice on half A, then B and full validation
    cands = {"last": model}
    if swa is not None and n_swa > 1:
        m2 = build_model(cfg["model"]).to(device); m2.load_state_dict(swa); cands["swa"] = m2
    t = time.time()
    scA = evaluate(cands, paths["valid"], A, device, tc["eval_batch"])
    cand_scores = {k: v.result() for k, v in scA.items()}
    chosen = max(cand_scores, key=lambda k: cand_scores[k]["scored"]["wp"])
    best = cands[chosen]
    torch.save({k: v.detach().cpu() for k, v in best.state_dict().items()}, out / "selected.pt")
    scB = evaluate({"m": best}, paths["valid"], B, device, tc["eval_batch"])["m"]
    scF = evaluate({"m": best}, paths["valid"], np.sort(np.concatenate([A, B])), device, tc["eval_batch"])["m"]
    for name, sc in (("A", scA[chosen]), ("B", scB)):
        mo = sc.moments(); np.savez(out / f"moments_{name}.npz", **mo)
    if tc["save_preds"]:
        predict_groups(best, paths["valid"], A, device, tc["eval_batch"], out / "preds_A.npy")
        predict_groups(best, paths["valid"], B, device, tc["eval_batch"], out / "preds_B.npy")
    eval_min = (time.time() - t) / 60

    log = [json.loads(l) for l in open(log_path)]
    w = [r["monA_scored"]["wp"] for r in log]
    slope = float(np.mean(np.diff(w[-3:]))) if len(w) >= 3 else float("nan")
    final = {"run_id": out.name, "phase": cfg.get("phase"), "name": cfg.get("name"), "model": cfg["model"], "train": tc,
             "params": n_params(model), "receptive_field": rf, "epochs_done": len(log),
             "train_minutes": train_min, "minutes_per_epoch": train_min / max(1, len(log)), "final_eval_minutes": eval_min,
             "candidates_A": {k: v["scored"] | {"all_wp": v["all"]["wp"]} for k, v in cand_scores.items()},
             "selected": chosen, "swa_epochs": n_swa,
             "A": cand_scores[chosen], "B": scB.result(), "full": scF.result(),
             "convergence": {"monA_last_slope_per_epoch": slope, "still_rising": bool(slope >= 0.001),
                             "monA_curve": w, "train_curve": [r["train_all"]["wp"] for r in log],
                             "skipped_updates": int(sum(r["skipped_updates"] for r in log))}}
    (out / "final.json").write_text(json.dumps(final, indent=2))
    print(f"DONE {out.name}: selected {chosen} | A {final['A']['scored']['wp']:.4f} | B {final['B']['scored']['wp']:.4f} | "
          f"full {final['full']['scored']['wp']:.4f} (all rows {final['full']['all']['wp']:.4f})", flush=True)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = Path(a.config).parent
    try:
        run(a.config, a.root, a.resume)
        (out / "failed.txt").unlink(missing_ok=True)
    except BaseException:
        (out / "failed.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
