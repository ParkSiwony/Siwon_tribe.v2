"""Run TRIBE v2 (or an offline fake) on each trial and store the predicted
fMRI time series as ``<trial_id>.npz`` with arrays ``times`` and ``preds``.

TRIBE v2 predicts the response of the *average subject* (unseen-subject mode)
on the fsaverage5 cortical mesh (20,484 vertices) at 1 Hz.
"""

from __future__ import annotations

import hashlib
import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from .stimuli import Trial

LOGGER = logging.getLogger(__name__)


class Predictor(tp.Protocol):
    def predict(self, wav_path: Path, trial: Trial, timings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(times[n_tr], preds[n_tr, n_vertices])``."""


class TribePredictor:
    """Thin wrapper around ``tribev2.TribeModel``."""

    def __init__(self, checkpoint: str = "facebook/tribev2", cache_folder: str | Path = "./cache",
                 device: str = "auto"):
        from tribev2 import TribeModel

        self.model = TribeModel.from_pretrained(checkpoint, cache_folder=cache_folder, device=device)
        # Keep the silent gaps between probes: they carry the hemodynamic tail.
        self.model.remove_empty_segments = False

    def predict(self, wav_path, trial, timings):
        # Audio -> whisperX word timings -> Llama-3.2 text features (with the
        # preceding 1,024 words as context) + Wav2Vec-BERT audio features.
        events = self.model.get_events_dataframe(audio_path=str(wav_path))
        preds, segments = self.model.predict(events=events, verbose=False)
        times = np.array([float(s.start) for s in segments])
        order = np.argsort(times)
        return times[order], preds[order]


def _seed(*parts) -> int:
    """Deterministic seed (Python's ``hash`` is salted per process)."""
    return int(hashlib.sha1(repr(parts).encode()).hexdigest()[:8], 16)


def hrf(tr: float = 1.0, length: float = 25.0) -> np.ndarray:
    """SPM-like double-gamma HRF sampled every ``tr`` seconds."""
    from scipy.stats import gamma

    t = np.arange(0, length, tr)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    return h / h.max()


class FakePredictor:
    """Synthetic predictor with known, planted effects (for dry runs and tests).

    Each segment category drives a fixed random spatial pattern, convolved
    with an HRF. On top of that:
      * every MS prime additionally activates the ``salience`` ROI, and
      * WV_NORM probes that follow an MS prime additionally activate ``vmPFC``.
    A correct analysis pipeline must recover exactly these two effects.
    """

    def __init__(self, n_vertices: int, roi_indices: dict[str, np.ndarray], effect: float = 1.0,
                 noise: float = 0.3, seed: int = 0):
        self.n_vertices, self.rois, self.effect, self.noise = n_vertices, roi_indices, effect, noise
        self.seed = seed
        self._patterns: dict[str, np.ndarray] = {}

    def _pattern(self, key: str) -> np.ndarray:
        if key not in self._patterns:
            rng = np.random.default_rng(_seed(self.seed, key))
            self._patterns[key] = rng.normal(size=self.n_vertices)
        return self._patterns[key]

    def predict(self, wav_path, trial, timings):
        n_tr = int(np.ceil(timings.onset.iloc[-1] + timings.duration.iloc[-1] + trial.tail))
        drive = np.zeros((n_tr, self.n_vertices))
        for row in timings.itertuples():
            box = np.zeros(n_tr)
            box[int(row.onset) : int(np.ceil(row.onset + row.duration))] = 1.0
            pattern = self._pattern(row.category).copy()
            if row.kind == "prime" and trial.prime_cond == "MS":
                pattern[self.rois["salience"]] += self.effect
            if row.kind == "probe" and row.category == "WV_NORM" and trial.prime_cond == "MS":
                pattern[self.rois["vmPFC"]] += self.effect
            drive += box[:, None] * pattern[None, :]
        h = hrf()
        bold = np.apply_along_axis(lambda x: np.convolve(x, h)[:n_tr], 0, drive)
        rng = np.random.default_rng(_seed(self.seed, trial.trial_id))
        bold += self.noise * rng.normal(size=bold.shape)
        return np.arange(n_tr, dtype=float), bold.astype(np.float32)


def save_prediction(path: Path, times: np.ndarray, preds: np.ndarray, dtype: str = "float16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, times=times, preds=preds.astype(dtype))


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as f:
        return f["times"], f["preds"].astype(np.float32)
