"""Row-by-row inference for every model family (what the competition scorer calls).

Convolutional models: every CausalConv1d keeps a ring buffer of its own past inputs. At row t the
buffer supplies the k-1 taps it needs (t-(k-1)d ... t-d); after the row, the conv's current input is
written into slot t mod S. All buffers live in ONE flat numpy array, and the taps for all convs are
gathered with one fancy-index operation per row, so the per-row Python cost does not grow with depth.

  StepModule(model)        torch module: (features [1,1,112], past [P]) -> (prediction [1,1,2], now [N])
  export_onnx(model, path) writes that module as ONNX (weight norm folded first)
  ConvStreamer(run, plan)  the ring buffer; `run(x, past)` is ORT or torch
  GRU models: export_onnx / GRUStreamer with explicit hidden states.

Before row 0 every buffer is zero, which is exactly the zero left-padding of full mode.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .models import N_IN, causal_convs, inference_copy, is_recurrent, to_fast_stream


class TapPlan:
    """Index bookkeeping for the flat ring buffer of all convs."""

    def __init__(self, convs):
        self.specs = [(c.cin, c.k, c.d) for c in convs]
        starts, total = [], 0
        g_lag, g_S, g_start, g_ch = [], [], [], []
        w_S, w_start, w_ch = [], [], []
        self.past_sizes, self.now_sizes = [], []
        for cin, k, d in self.specs:
            S = max(1, (k - 1) * d)                       # slots: furthest look-back
            starts.append(total)
            # gather: taps ordered oldest (j = k-1) ... newest (j = 1), each a full channel vector
            for j in range(k - 1, 0, -1):
                g_lag += [j * d] * cin; g_S += [S] * cin; g_start += [total] * cin; g_ch += list(range(cin))
            w_S += [S] * cin; w_start += [total] * cin; w_ch += list(range(cin))
            self.past_sizes.append((k - 1) * cin)
            self.now_sizes.append(cin)
            total += S * cin
        self.total = total
        self.g_lag, self.g_S = np.array(g_lag, np.int64), np.array(g_S, np.int64)
        self.g_start, self.g_ch = np.array(g_start, np.int64), np.array(g_ch, np.int64)
        self.g_cin = np.array([cin for cin, k, d in self.specs for _ in range((k - 1) * cin)], np.int64)
        self.w_S, self.w_start, self.w_ch = np.array(w_S, np.int64), np.array(w_start, np.int64), np.array(w_ch, np.int64)
        self.w_cin = np.array([cin for cin, k, d in self.specs for _ in range(cin)], np.int64)
        self.P, self.N = int(sum(self.past_sizes)), int(sum(self.now_sizes))

    def gather_index(self, t):
        return self.g_start + ((t - self.g_lag) % self.g_S) * self.g_cin + self.g_ch

    def write_index(self, t):
        return self.w_start + (t % self.w_S) * self.w_cin + self.w_ch


class StepModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = to_fast_stream(inference_copy(model))
        self.convs = causal_convs(self.m)
        for c in self.convs:
            c.stream = True
        self.plan = TapPlan(self.convs)

    def forward(self, features, past):
        off = 0
        for c, n in zip(self.convs, self.plan.past_sizes):
            if c.k > 1:
                if c.fast is not None:                    # [1, (k-1)*cin, 1], tap-major: matches W2
                    c.taps_flat = past[:, off:off + n]
                else:                                     # [k-1, cin] -> [1, cin, k-1]
                    c.taps = past[0, off:off + n, 0].reshape(c.k - 1, c.cin).transpose(0, 1).unsqueeze(0)
            off += n
        pred = self.m(features)                           # [1, 1, 2]
        now = torch.cat([c.now for c in self.convs], 1)   # [1, N, 1]
        return pred, now


class ConvStreamer:
    def __init__(self, run, plan: TapPlan):
        self.run, self.plan = run, plan
        self.buf = np.zeros(plan.total, np.float32)
        self.t = 0

    def reset(self):
        self.buf.fill(0.0); self.t = 0

    def step(self, x):
        past = self.buf[self.plan.gather_index(self.t)]
        pred, now = self.run(np.asarray(x, np.float32).reshape(1, 1, N_IN), past.reshape(1, -1, 1))
        self.buf[self.plan.write_index(self.t)] = np.asarray(now).reshape(-1)
        self.t += 1
        return np.asarray(pred).reshape(2)


class GRUStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = inference_copy(model)

    def forward(self, features, h1, h2):
        pred, (h1, h2) = self.m(features, (h1, h2))
        return pred, h1, h2


class GRUStreamer:
    def __init__(self, run, hidden):
        self.run, self.H = run, hidden
        self.reset()

    def reset(self):
        self.h1 = np.zeros((1, 1, self.H), np.float32); self.h2 = np.zeros((1, 1, self.H), np.float32)

    def step(self, x):
        pred, self.h1, self.h2 = self.run(np.asarray(x, np.float32).reshape(1, 1, N_IN), self.h1, self.h2)
        return np.asarray(pred).reshape(2)


# ----------------------------------------------------------------------------- torch / ONNX runners
def torch_streamer(model):
    if is_recurrent(model):
        step = GRUStep(model).eval()

        def run(x, h1, h2):
            with torch.no_grad():
                p, a, b = step(torch.from_numpy(x), torch.from_numpy(h1), torch.from_numpy(h2))
            return p.numpy(), a.numpy(), b.numpy()
        return GRUStreamer(run, model.hidden)
    step = StepModule(model).eval()

    def run(x, past):
        with torch.no_grad():
            p, n = step(torch.from_numpy(x), torch.from_numpy(past))
        return p.numpy(), n.numpy()
    return ConvStreamer(run, step.plan)


def _export(module, args, path, inputs, outputs):
    kw = dict(input_names=inputs, output_names=outputs, opset_version=17)
    try:
        torch.onnx.export(module, args, str(path), dynamo=False, **kw)
    except TypeError:                                     # older torch without the dynamo flag
        torch.onnx.export(module, args, str(path), **kw)


def export_onnx(model, path):
    """Write the one-row model to `path`. Returns a dict describing how to stream it."""
    if is_recurrent(model):
        step = GRUStep(model).eval()
        H = model.hidden
        _export(step, (torch.zeros(1, 1, N_IN), torch.zeros(1, 1, H), torch.zeros(1, 1, H)), path,
                ["features", "h1", "h2"], ["prediction", "h1_out", "h2_out"])
        return {"kind": "gru", "hidden": H}
    step = StepModule(model).eval()
    _export(step, (torch.zeros(1, 1, N_IN), torch.zeros(1, step.plan.P, 1)), path, ["features", "past"], ["prediction", "now"])
    return {"kind": "conv", "specs": step.plan.specs}


def ort_session(path, threads=1):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


class _Spec:
    def __init__(self, cin, k, d):
        self.cin, self.k, self.d = cin, k, d


def ort_streamer(path, info, threads=1):
    sess = ort_session(path, threads)
    if info["kind"] == "gru":
        def run(x, h1, h2):
            return sess.run(None, {"features": x, "h1": h1, "h2": h2})
        return GRUStreamer(run, info["hidden"])
    plan = TapPlan([_Spec(*s) for s in info["specs"]])

    def run(x, past):
        return sess.run(None, {"features": x, "past": past})
    return ConvStreamer(run, plan)


class PredictionModel:
    """Competition-style callback (same contract as solution.py): predict(DataPoint) -> None | [t0, t1]."""

    def __init__(self, streamer):
        self.s, self.seq = streamer, None

    def predict(self, dp):
        if dp.seq_ix != self.seq:
            self.s.reset(); self.seq = dp.seq_ix
        p = self.s.step(dp.state)
        return p.astype(np.float32) if dp.need_prediction else None
