from typing import Optional, Sequence

import onnxruntime as ort
import numpy as np
import numpy.typing as npt

from pkg.utils import DataPoint, grab_session

type f64_NARRAY = npt.NDArray[np.float64]
type f32_NARRAY = npt.NDArray[np.float32]

type g_f = type[np.floating]
type gf_NARRAY = npt.NDArray[np.floating] # np array of generic type

type T_DataPoint = tuple[int, int, bool, gf_NARRAY]

class PredictionModel:
    def __init__(self, depth: int = 2, precision: g_f = np.float64):
        self.session: ort.InferenceSession = grab_session(weights_name="weight_0.onnx")
        self.precision: g_f = precision

        self.layers: list[gf_NARRAY] = self._empty_layers(
            size=(1, 1, 128), precision=precision, depth=depth
            )

        self.current_sequence_idx: Optional[int] = None
        self.previous_step: Optional[int] = None

    @staticmethod
    def _empty_layers(size: tuple[int, ...], precision: g_f, depth: int = 2) -> list[gf_NARRAY]:
        return [np.zeros(size, dtype=precision) for _ in range(depth)]

    def _fill_layers(self, v: float | int) -> None:
        for l in self.layers:
            l.fill(v)

    def _assign_layers(self, new: Sequence[gf_NARRAY]) -> None:
        if len(new) != len(self.layers):
            raise IndexError(f"Incompatible layer updates, current layers: {len(self.layers)}, update length: {len(new)}.")
        for i, l in enumerate(new):
            self.layers[i] = l

    def update_state(self, seq_id: Optional[int], prev_step: Optional[int]) -> None:
        self.current_sequence_idx: Optional[int] = seq_id
        self.previous_step: Optional[int] = prev_step

    def extract_datapoint(self, data_point: DataPoint) -> T_DataPoint:
        return (int(data_point.seq_ix), int(data_point.step_in_seq), 
                bool(data_point.need_prediction), np.asarray(data_point.state, self.precision))

    def run(self, feed: dict[str, gf_NARRAY]) -> tuple[gf_NARRAY, Sequence[gf_NARRAY]]:
        packed: tuple[gf_NARRAY, ...] = self.session.run(None, feed,) # type: ignore
        return (packed[0], packed[1:])

    def predict(self, data_point: DataPoint) -> Optional[gf_NARRAY]:
        (idx, step, predict, features) = self.extract_datapoint(data_point=data_point)
        if self.current_sequence_idx != idx:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.update_state(idx, None)
            self._fill_layers(v=0.)
        self.validate_step(step)
        self.previous_step = step

        self.validate_features(features)

        inference, layers = self.run({
                "features": features.reshape((1, 1, 112)),
                **{f"hidden_{i}": l for i, l in enumerate(self.layers)}
            })
        self._assign_layers(layers)
        if not predict:
            return None
        
        return np.asarray(inference[0, 0], dtype=self.precision)

    def validate_step(self, step: Optional[int]) -> None:
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
    def validate_features(self, f: gf_NARRAY) -> None:
        if f.shape != (112,) or not np.isfinite(f).all():
            raise ValueError(f"Expected 112 finite float32 features, got {f.shape}.")