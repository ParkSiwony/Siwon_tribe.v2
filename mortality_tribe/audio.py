"""Text-to-speech and assembly of one audio file per trial.

Every segment is synthesised separately and cached by its text, so the probe
audio is *byte-identical* across prime conditions. The exact onset of every
segment is written to a sidecar table, which the analysis uses to extract
the predicted response to each probe.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from .stimuli import Trial

SAMPLE_RATE = 16_000


class TTSBackend(tp.Protocol):
    name: str

    def synth(self, text: str) -> np.ndarray:
        """Return mono float32 audio at ``SAMPLE_RATE``."""


class GTTSBackend:
    """Google TTS (what ``TribeModel.get_events_dataframe(text_path=...)`` uses)."""

    name = "gtts"

    def __init__(self, lang: str = "en", tld: str = "com"):
        self.lang, self.tld = lang, tld

    def synth(self, text: str) -> np.ndarray:
        from gtts import gTTS

        with tempfile.TemporaryDirectory() as tmp:
            mp3 = Path(tmp) / "x.mp3"
            gTTS(text, lang=self.lang, tld=self.tld).save(str(mp3))
            return _decode_with_ffmpeg(mp3)


class SilentBackend:
    """Offline stand-in: silence with a speech-like duration (~2.7 words/s).

    Only for dry runs and tests; TRIBE needs real speech.
    """

    name = "silent"

    def __init__(self, seconds_per_word: float = 0.37):
        self.spw = seconds_per_word

    def synth(self, text: str) -> np.ndarray:
        n = int(len(text.split()) * self.spw * SAMPLE_RATE)
        return np.zeros(max(n, SAMPLE_RATE // 2), dtype=np.float32)


def get_backend(name: str) -> TTSBackend:
    if name == "gtts":
        return GTTSBackend()
    if name == "silent":
        return SilentBackend()
    raise ValueError(f"Unknown TTS backend {name!r}")


def _decode_with_ffmpeg(path: Path) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode TTS output")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _trim_silence(wav: np.ndarray, thresh: float = 1e-3, pad: float = 0.05) -> np.ndarray:
    """Strip leading/trailing silence so that segment onsets are speech onsets."""
    loud = np.flatnonzero(np.abs(wav) > thresh)
    if len(loud) == 0:
        return wav
    p = int(pad * SAMPLE_RATE)
    return wav[max(loud[0] - p, 0) : loud[-1] + p]


class SegmentCache:
    def __init__(self, folder: str | Path, backend: TTSBackend):
        self.folder = Path(folder) / backend.name
        self.folder.mkdir(parents=True, exist_ok=True)
        self.backend = backend

    def get(self, text: str) -> np.ndarray:
        key = hashlib.sha1(text.encode()).hexdigest()[:16]
        path = self.folder / f"{key}.npy"
        if path.exists():
            return np.load(path)
        wav = _trim_silence(self.backend.synth(text)).astype(np.float32)
        np.save(path, wav)
        return wav


def render_trial(trial: Trial, cache: SegmentCache, out_dir: str | Path) -> tuple[Path, pd.DataFrame]:
    """Write ``<trial_id>.wav`` and return it with a table of segment timings."""
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = [np.zeros(int(trial.lead_in * SAMPLE_RATE), dtype=np.float32)]
    t = trial.lead_in
    rows = []
    for pos, seg in enumerate(trial.segments):
        wav = cache.get(seg.text)
        dur = len(wav) / SAMPLE_RATE
        rows.append(
            dict(
                trial_id=trial.trial_id, position=pos, kind=seg.kind, item_id=seg.item_id,
                category=seg.category, text=seg.text, onset=t, duration=dur,
            )
        )
        chunks += [wav, np.zeros(int(seg.gap_after * SAMPLE_RATE), dtype=np.float32)]
        t += dur + seg.gap_after
    chunks.append(np.zeros(int(trial.tail * SAMPLE_RATE), dtype=np.float32))
    audio = np.concatenate(chunks)
    wav_path = out_dir / f"{trial.trial_id}.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)
    return wav_path, pd.DataFrame(rows)
