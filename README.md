# Wunder Connectome: can a causal TCN beat the GRU baseline?

A self-contained search over causal Temporal Convolutional Networks (residual TCN, multi-scale TCN, causal
ModernTCN) against the verified GRU baseline. Everything is trained and judged with the same data,
preprocessing, loss and **official competition metric**. It runs on any number of GPUs and resumes after
interruptions.

## 1. Setup

```bash
git clone https://github.com/Marcocxm171/wunder_fund_challange.git
cd wunder_fund_challange
pip install -r requirements.txt        # install the CUDA build of torch for your driver first
```

You also need the competition **starter pack**, a folder containing the official `utils.py` and
`datasets/train.parquet`, `datasets/valid.parquet` and `datasets/valid_mask.parquet` (about 35 GB). The code
finds it in this order:
1. `--root <folder>`
2. the `WUNDER_ROOT` environment variable
3. the current folder
4. the folder above this repo

The official `utils.py` (metric, `DataPoint`, `ScorerStepByStep`) is used unchanged for every score.

```bash
export WUNDER_ROOT=/path/to/wnn_connectome_starterpack
python -m pytest -q tests              # 20 unit tests, CPU, about 4 min
```

## 2. Run the search

```bash
python -m tcnsearch.search --gpus 0,1,2,3          # one run per GPU; add --per-gpu 2 for two per GPU
```

* **Checks come first.** The data checks (T1–T5, T9, T11) run first, and the search refuses to train if any
  of them fails. Each new model configuration is then checked before its first run (T6–T8, T9m).
* **Resumable.** Run the same command again after a crash or reboot. Finished runs are skipped and
  interrupted runs resume from their last epoch.
* **Review between phases.** `--until 2` stops after phase 2. Look at the results, then rerun with `--until 3`, and so on.
* **Plan view.** `--plan` lists the phases and the runs done or pending, and trains nothing.
* **Several machines** can share one `runs/` folder on a shared disk. Runs claim their folder atomically.
* **Pipeline test.** Build a 10-sequence copy of the data, then run the whole search on CPU with tiny models
  (about an hour):
  ```bash
  python -m tcnsearch.devdata --out devroot --n-train 6 --n-valid 4
  python -m tcnsearch.search --root devroot --runs devroot/runs --smoke --gpus cpu
  ```

When the search finishes:
```bash
python -m tcnsearch.final                          # finalists vs GRU: full validation, half B, 3 seeds, ensembles
python -m tcnsearch.bench                          # CPU speed, on an IDLE machine (not while training)
python -m tcnsearch.report                         # report/REPORT.md + figures; or open notebooks/tcn_report.ipynb
```

Send back the whole `runs/` folder. Logs and checkpoints are small. The prediction files
(`preds_*.npy`) are needed for ensembles and add roughly 150 MB per finalist run.

## 3. Evaluation protocol

* **Validation split.** Validation is split by sequence: **half A** is the even row groups, **half B** the
  odd ones.
  * **A** is used for every decision: monitoring, learning rate, epochs, checkpoint choice, promotion and blend
    weights.
  * **B** is the untouched confirmation set. The search never reads it. `final.py` reports it together with the
    **full validation set**.
* **Scores are on the official metric**, global weighted Pearson:
  * **scored rows** use `is_scored` (identical to `valid_mask.parquet`, checked by T2);
  * **all rows** use every row from step 99;
  * t0, t1 and the combined score are reported for both.
* **Checkpoint choice:** the last epoch, or the average of the second half of the epochs, whichever scores
  higher on A.
* **Comparisons:** 3-seed means on B, with a bootstrap over sequences plus seed-to-seed spread. A model counts
  as better only if it wins by at least 0.002 and the interval is above 0.
* **Leaderboard estimates** appear only if you pass `final.py --lb-gap`, and are labelled as uncertain.

## 4. Training recipe (identical for every model, including the GRU)

| | |
|---|---|
| Inputs | the 112 raw features exactly as `DataPoint.state` (check T3) |
| Loss | `wmse` (squared error × \|target\|) on rows ≥ 99, targets clipped to ±2 |
| Optimiser | Adam, no weight decay, gradient clipping 1.0, cosine learning rate over the epochs |
| Batches | 32 sequences × 1,000-row chunks, the same number of updates for every model. TCNs also see `receptive_field − 1` rows of context per chunk (no loss on them); the GRU carries its state |
| Determinism | seeds fixed, cuDNN deterministic, the same data order for every model at the same seed |
| Recorded per epoch | loss, gradient norms (total and per block), clipped and skipped updates, learning rate, half-A monitor (300 sequences: scored and all rows, t0 and t1), 100 training sequences, minutes, GPU memory |

## 5. The search (`tcnsearch/search.py`)

Screening runs use 8 epochs. "Full" means the epoch count chosen in phase 1. Each phase builds on the
winners of the earlier ones (half-A score).

| Phase | Runs | What |
|---|---|---|
| 1 | 8–10 | Learning rate 0.0005, 0.001 and 0.002 for the TCN and the GRU. Then epochs 8 → 16 → 24, adding epochs only while they gain ≥ 0.001 |
| 2 | 15 | Receptive field ≈ 250 / 500 / 1,000 / 2,000 rows × kernel 2 / 3 / 5 / 7, all 16 combinations (64 channels, ReLU, WeightNorm) |
| 3 | 8 | Width 32 / 48 / 96 / 128 on the two best phase-2 combinations (64 is already done) |
| 4 | 11 | Activation ReLU / SiLU / GELU / gated × normalisation WeightNorm / LayerNorm / none, all 12 combinations |
| 5 | 15 | Multi-scale TCN (4 variants) and causal ModernTCN (5 variants, including the variable-independent design). Each family's best gets a learning-rate check, then a full-length run |
| 6 | 2–6 | Interaction checks: the best activation and normalisation on the second-best combination and on neighbouring widths. For gated activation or no normalisation, a learning-rate check; for gated, a version with the parameter count matched |
| 7 | ≈18 | The top 6 screens, plus runs still improving at their last epoch, retrained at full length. Learning rate ×0.5 / ×2 for the top 2. The top 3 get **3 seeds**, and so does the GRU |

Tie rule: anything within 0.002 of the best counts as a tie. No family is dropped without its own
learning-rate check and a full-length run. Every decision and its rule is written to
`runs/_search/decisions.json`. Failed runs are retried once, then listed in `missing_runs.json`.

## 6. Code layout

| File | Contents |
|---|---|
| `tcnsearch/models.py` | All families. Every temporal operation is one `CausalConv1d` that runs on whole sequences or row by row |
| `tcnsearch/streaming.py` | Row-by-row inference with one flat ring buffer for all convolutions; ONNX export; GRU state |
| `tcnsearch/data.py` | Parquet loading, the A/B split, the official metric wrapper, per-sequence moments for bootstraps |
| `tcnsearch/checks.py` | The automatic checks (T1–T11) |
| `tcnsearch/train.py` | One run from a config file |
| `tcnsearch/search.py` | Phases, promotion rules, multi-GPU scheduler |
| `tcnsearch/final.py` | Final comparison against the GRU |
| `tcnsearch/bench.py` | CPU speed |
| `tcnsearch/report.py`, `notebooks/tcn_report.ipynb` | The report |

## 7. What each run folder contains

`runs/<name>_<hash>/`:
* `config.json`, `env.json` (git commit, torch, CUDA, GPU, data file sizes), `log.jsonl` (per epoch)
* `epoch_XX.pt`, `state.pt` (for resuming), `selected.pt`
* `final.json`: scores on A, B and full validation; parameters; receptive field; times; convergence
* `moments_A/B.npz`, optional `preds_A/B.npy`, `train.log`

The folder name ends in a hash of the full configuration, so identical configurations are trained once and
results can always be traced back to their exact settings.
