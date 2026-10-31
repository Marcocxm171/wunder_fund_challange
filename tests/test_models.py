"""Fast unit tests (no data needed):  python -m pytest -q tests

Every family x activation x normalisation must be causal, have the receptive field its formula says,
give identical results in training chunks and on the full sequence, and stream row by row (torch and ONNX)
exactly like full mode. The search runs the same checks automatically on every real configuration.
"""
import itertools

import numpy as np
import pytest
import torch

from tcnsearch import data as D
from tcnsearch.checks import check_model
from tcnsearch.search import CELLS
from tcnsearch.models import rf_tcn

SMALL = []
for act, norm in itertools.product(["relu", "silu", "gelu", "gated"], ["weight", "layer", "none"]):
    SMALL.append({"family": "tcn", "channels": 8, "kernel": 3, "dilations": [1, 2, 4], "act": act, "norm": norm})
SMALL += [
    {"family": "tcn", "channels": 8, "kernel": 2, "dilations": [1, 2, 4, 8], "act": "relu", "norm": "weight"},
    {"family": "tcn", "channels": 8, "kernel": 7, "dilations": [1, 2, 4, 3], "act": "silu", "norm": "layer"},
    {"family": "multiscale", "branches": [{"channels": 4, "kernel": 3, "dilations": [1, 2]},
                                          {"channels": 4, "kernel": 3, "dilations": [1, 4, 16]}], "act": "gelu", "norm": "weight"},
    {"family": "modern", "width": 8, "kernel": 11, "dilations": [1, 3], "ffn_ratio": 2.0, "act": "gated", "norm": "layer"},
    {"family": "modern_vi", "var_dim": 1, "kernel": 7, "dilations": [1, 2], "ffn_ratio": 1.0, "act": "relu", "norm": "none"},
    {"family": "gru", "hidden": 8},
]


@pytest.mark.parametrize("cfg", SMALL, ids=lambda c: f"{c['family']}-{c.get('act', '')}-{c.get('norm', '')}")
def test_model_checks(cfg):
    r = check_model(cfg)
    assert r["pass"], [t for t in r["tests"] if not t["pass"]]


def test_cells_hit_targets():
    for (k, target), dil in CELLS.items():
        rf = rf_tcn(k, dil)
        assert abs(rf - target) / target < 0.03, (k, target, rf)


def test_moments_equal_reference():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(3, 20000, 2)).astype(np.float32); P = (Y + rng.normal(size=Y.shape)).astype(np.float32)
    M = rng.random((3, 20000)) < 0.3; M[:, :99] = False
    mom = sum(D.seq_moments(Y[i], P[i], M[i]) for i in range(3))
    ref = D.reference_global_wp(Y.reshape(-1, 2), P.reshape(-1, 2), M.reshape(-1))
    assert abs(D.wp_from_moments(mom)[0] - ref[0]) < 1e-9
