"""Gate step A2: TRIBE's predicted brain response to your contrast sentences.

Each pair becomes two text timelines, ``context`` then ``opt_pos`` and
``context`` then ``opt_neg`` (the A/B question is identical within a pair
and left out). The response to a timeline is the mean prediction over the
option period plus a short tail; ``pos - neg`` per pair gives a text-evoked
brain contrast that can be compared with the image contrasts.

Text is presented without audio (TRIBE was trained with modality dropout, so
text-only input is supported). Words get synthetic timings, and context comes
from TRIBE's own neuralset transforms, as in ``pilot_bridge.text_events``.
"""

from __future__ import annotations

import logging
import typing as tp

import numpy as np
import pandas as pd

from .pilot_bridge import finish_text_events

LOGGER = logging.getLogger(__name__)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def pair_text_events(pairs: pd.DataFrame, words_per_second: float = 2.5, word_duration: float = 0.3,
                     gap: float = 0.8, lead: float = 1.0, language: str = "english",
                     use_neuralset: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Word events (one timeline per pair x role) and an index with the option window.

    Returns ``(events, index)``; ``index`` has one row per timeline with
    columns timeline, pair_id, set, family, gen, role, opt_on, opt_off.
    """
    rows, index = [], []
    step = 1.0 / words_per_second
    for r in pairs.itertuples():
        for role, option in (("pos", r.pos), ("neg", r.neg)):
            timeline = f"{r.set}|{r.id}|{role}"
            t = lead
            context = _sentence(r.context).split() if r.context else []
            for w in context:
                rows.append(dict(type="Word", text=w, start=t, duration=word_duration, timeline=timeline,
                                 subject="default", language=language))
                t += step
            if context:
                t += gap
            opt_on = t
            for w in _sentence(option).split():
                rows.append(dict(type="Word", text=w, start=t, duration=word_duration, timeline=timeline,
                                 subject="default", language=language))
                t += step
            index.append(dict(timeline=timeline, pair_id=r.id, set=r.set, family=r.family, gen=r.gen, role=role,
                              opt_on=opt_on, opt_off=t - step + word_duration))
    return finish_text_events(pd.DataFrame(rows), use_neuralset), pd.DataFrame(index)


def text_responses(tribe, events: pd.DataFrame, index: pd.DataFrame, after: float = 3.0,
                   free_gpu_for_llama: bool = True) -> np.ndarray:
    """``[n_timelines, n_vertices]`` mean prediction over [opt_on, opt_off + after],
    in ``index`` order.

    Batches are streamed through the brain model so that hundreds of short
    timelines (each padded to TRIBE's 100-s window) never sit in memory at
    once. The text features are computed once by ``get_loaders`` (Llama-3.2-3B
    in fp32), with the brain model parked on the CPU meanwhile.
    """
    import torch

    model = tribe._model
    device = next(model.parameters()).device
    if free_gpu_for_llama and torch.cuda.is_available():
        model.to("cpu")
        torch.cuda.empty_cache()
    try:
        loader = tribe.data.get_loaders(events=events, split_to_build="all")["all"]
    finally:
        model.to(device)
    windows = {r.timeline: (np.floor(r.opt_on), r.opt_off + after) for r in index.itertuples()}
    tr = float(tribe.data.TR)
    found = {}
    with torch.no_grad():
        for batch in loader:
            out = model(batch.to(device)).float().cpu().numpy()  # B, V, T
            for i, seg in enumerate(batch.segments):
                name = str(seg.timeline)
                if name not in windows:
                    continue
                lo, hi = windows[name]
                t = float(seg.start) + np.arange(out.shape[-1]) * tr
                sel = (t >= lo) & (t <= hi)
                found[name] = out[i][:, sel].mean(1)
    missing = [t for t in index.timeline if t not in found]
    if missing:
        raise RuntimeError(f"{len(missing)} timelines without predictions, e.g. {missing[0]}")
    return np.stack([found[t] for t in index.timeline])


def text_pair_differences(index: pd.DataFrame, maps: np.ndarray) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """``{set: (pair table, pos - neg maps [n_pairs, V])}``."""
    rows = {(r.pair_id, r.role): i for i, r in enumerate(index.itertuples())}
    out = {}
    for name, g in index[index.role == "pos"].groupby("set", sort=False):
        meta = g[["pair_id", "set", "family", "gen"]].reset_index(drop=True)
        d = np.stack([maps[rows[(p, "pos")]] - maps[rows[(p, "neg")]] for p in meta.pair_id])
        out[name] = (meta, d)
    return out


def family_maps(meta: pd.DataFrame, d: np.ndarray) -> dict[str, np.ndarray]:
    """Mean pos - neg map per family (exploratory: which families look like which image contrast)."""
    return {f: d[(meta.family == f).to_numpy()].mean(0) for f in sorted(meta.family.unique())}
