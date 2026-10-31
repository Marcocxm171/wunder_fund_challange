# Running the TCN search: instructions

Thanks for running this! The job: train about 75 models on your GPUs, then send back the results zip.
It runs mostly unattended. If anything stops, run the same command again and it continues where it left off.

**Before you start:** run Step 4 (the quick test) first. After Step 5 (phase 1), send the results back and
wait for a go-ahead before running the rest.

---

## What you need

| | |
|---|---|
| GPU | Any NVIDIA GPU with CUDA. More GPUs = faster; it uses as many as you give it |
| GPU memory | 12 GB+ is comfortable. Less also works: it splits batches automatically if memory runs out |
| Disk | About 35 GB for the data, plus about 20 GB for results (a fast SSD helps) |
| RAM | 16 GB+ |
| OS | Linux recommended (Windows works too) |
| Python | 3.10 or newer |
| Data | The competition **starter pack** folder from Marco: `utils.py` and `datasets/train.parquet`, `datasets/valid.parquet`, `datasets/valid_mask.parquet` (about 35 GB) |
| Time | About 100 GPU hours in total: roughly 4–5 days on 1 GPU, 1.5 days on 4 GPUs, a little over a day on 8 |

---

## Step 1: Get the code and install

```bash
git clone https://github.com/Marcocxm171/wunder_fund_challange.git
cd wunder_fund_challange
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Install the CUDA build of PyTorch for your driver, using the command from https://pytorch.org/get-started/locally/.
Then install the rest:

```bash
pip install -r requirements.txt
```

Check that PyTorch sees your GPUs:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

## Step 2: Point it at the data

```bash
export WUNDER_ROOT=/path/to/wnn_connectome_starterpack      # Windows: set WUNDER_ROOT=C:\path\to\...
```

That folder must contain `utils.py` and the `datasets/` folder. Don't edit or replace `utils.py`: it is the
competition's official metric.

## Step 3: Unit tests (about 4 minutes, CPU)

```bash
python -m pytest -q tests
```

Expected: `20 passed`. If anything fails, stop and send Marco the output.

## Step 4: Quick end-to-end test (about 30–60 minutes)

This runs the whole pipeline with tiny models on 10 sequences, so any problem with your setup shows up now
rather than after a day of training:

```bash
python -m tcnsearch.devdata --out devroot --n-train 6 --n-valid 4
python -m tcnsearch.search --root devroot --runs devroot/runs --smoke --gpus 0
```

It should end with `Search finished`. The scores are meaningless at this size; only "does it run" matters.
Then delete the test folder: `rm -rf devroot`.

## Step 5: Phase 1 only, then send results (a few hours)

Use a session that survives logging out, such as `tmux` or `screen` on Linux. List your GPU ids after
`--gpus`:

```bash
tmux new -s tcn
python -m tcnsearch.search --gpus 0,1,2,3 --until 1 2>&1 | tee search_log.txt
```

* **Checks first.** It starts with the automatic data checks (a few minutes). If any check says `FAIL`,
  nothing is trained. Send Marco `runs/_checks/data_checks.json`.
* **Phase 1** trains about 10 models: learning rate and number of epochs, for the TCN and the GRU.
* **When it stops**, send Marco the results (see **What to send back**) and wait for his OK.

Phase 1 measures the real time per epoch on your GPU, so the timetable for the rest can be confirmed.

## Step 6: The full search

```bash
python -m tcnsearch.search --gpus 0,1,2,3 2>&1 | tee -a search_log.txt
```

* **Two runs per GPU:** add `--per-gpu 2` if your GPUs are large (24 GB+) and mostly idle with one run each
  (check `nvidia-smi`).
* **Pausing:** `--until N` stops after phase N. Rerun with a higher number to continue.
* **Progress:** `python -m tcnsearch.search --plan` (trains nothing). Each run's own log is
  `runs/<run>/train.log`.
* **If the machine reboots or the command dies:** run the same command again. Finished runs are skipped and
  interrupted runs resume from their last finished epoch.
* **A run that crashes** is retried once, then skipped and listed in `runs/_search/missing_runs.json`. The
  search carries on.

The search ends with `Search finished. Next: python -m tcnsearch.final ...`.

## Step 7: Final evaluation, speed test, report

```bash
python -m tcnsearch.final        # finalists vs GRU, 3 seeds each (GPU, ~30 min)
python -m tcnsearch.bench        # CPU speed. Run ONLY when nothing else is running on the machine
python -m tcnsearch.report       # writes report/REPORT.md and figures
python -m tcnsearch.pack         # writes tcn_results_<date>.zip
```

`bench` measures CPU timing, so it must run while nothing else is using the machine: no training and no other
heavy programs.

---

## What to send back

### After phase 1 (Step 5)

1. **The zip:** run `python -m tcnsearch.pack`. It makes `tcn_results_<date>.zip`, usually under 100 MB at
   this point.
2. **`search_log.txt`**
3. **A short note** with:
   * GPU model and how many (e.g. "4 × RTX 4090")
   * OS, and whether you used `--per-gpu 2`
   * anything odd: errors, warnings that repeated, runs that restarted

### At the end (Step 7)

1. **The final zip:** `python -m tcnsearch.pack --with-preds`. The `--with-preds` part matters: it includes
   the finalists' predictions, which the ensembles are built from. Expect about 2–5 GB. Send it through
   Google Drive, WeTransfer or similar.
2. **`search_log.txt`**, the complete log
3. **The same short note**, plus:
   * total wall-clock time
   * whether `bench` ran on an idle machine
   * any runs listed in `runs/_search/missing_runs.json`, and why, if you know

The zip contains:
* every run's settings, environment (GPU, versions, code commit), per-epoch curves, final scores, training log
  and the chosen weights
* all check reports
* the search's decisions and summary
* the final comparison, the CPU timings and the report

Not included: the per-epoch checkpoints and resume files (large, not needed). Please **keep your `runs/`
folder until Marco confirms** he has everything, in case a run needs re-checking.

---

## Please don't

* **Edit the code** during the search. Every run records the code version, and changes make results
  incomparable. If something seems wrong, stop and tell Marco.
* **Move, rename or delete** anything inside `runs/` while the search is running.
* **Run `bench` while training** is going on.
* **Start two searches at once** pointing at different `runs/` folders. One command uses all the GPUs you
  list.

## If something goes wrong

| Problem | What to do |
|---|---|
| `Starter pack not found` | `WUNDER_ROOT` isn't set, or the folder lacks `utils.py` / `datasets/` |
| `data checks FAILED` | Send `runs/_checks/data_checks.json` to Marco; don't work around it |
| `model checks FAILED for ...` | Send `runs/_checks/models/` and `search_log.txt` |
| `out of GPU memory: now N micro-batches` | Normal. It adapts automatically; the results are unaffected |
| A run keeps failing | Look at `runs/<run>/train.log` / `failed.txt`, then send them |
| Training seems very slow | Check `nvidia-smi`. If GPUs sit idle, data reading is the limit: use fewer parallel runs, or put the data on an SSD |
