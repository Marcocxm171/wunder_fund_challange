"""Build a tiny starter-pack copy for smoke tests (the real data stays where it is).

    python -m tcnsearch.devdata --root <starter pack> --out devroot --n-train 6 --n-valid 4

Takes the first sequences of valid.parquet (or --source, e.g. datasets/valid_subset10.parquet):
n-train of them become devroot/datasets/train.parquet, the next n-valid become valid.parquet,
with a matching valid_mask.parquet. utils.py is copied unchanged. Then e.g.
    python -m tcnsearch.search --root devroot --smoke --gpus cpu
"""
import argparse
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from . import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--out", default="devroot")
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-valid", type=int, default=4)
    a = ap.parse_args()
    root = D.find_root(a.root)
    src = Path(a.source) if a.source else root / "datasets" / "valid.parquet"
    out = Path(a.out); (out / "datasets").mkdir(parents=True, exist_ok=True)
    shutil.copy(root / "utils.py", out / "utils.py")
    pf = pq.ParquetFile(src)
    assert pf.num_row_groups >= a.n_train + a.n_valid, "source has too few sequences"
    with pq.ParquetWriter(out / "datasets" / "train.parquet", pf.schema_arrow) as w:
        for g in range(a.n_train):
            w.write_table(pf.read_row_group(g))
    with pq.ParquetWriter(out / "datasets" / "valid.parquet", pf.schema_arrow) as w, \
            pq.ParquetWriter(out / "datasets" / "valid_mask.parquet",
                             pf.schema_arrow.__class__.from_pandas(pf.read_row_group(0, columns=["seq_ix", "step_in_seq", "is_scored"]).to_pandas(), preserve_index=False)) as wm:
        for g in range(a.n_train, a.n_train + a.n_valid):
            w.write_table(pf.read_row_group(g))
            wm.write_table(pf.read_row_group(g, columns=["seq_ix", "step_in_seq", "is_scored"]))
    print(f"wrote {out}: {a.n_train} train + {a.n_valid} valid sequences")


if __name__ == "__main__":
    main()
