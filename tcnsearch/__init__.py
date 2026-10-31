"""Wunder Connectome TCN search. See README.md."""
import os

# Must be set before CUDA initialises, for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
