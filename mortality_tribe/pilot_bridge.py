"""Pilot step 2: turn a predicted brain map into a Llama steering direction.

A cortical map (20,484 vertices) cannot be added to Llama: the two live in
different spaces, and Llama never sees the images. TRIBE v2 connects them.
Its text stream is Llama-3.2-3B hidden states, which TRIBE maps to cortex. So
we ask TRIBE the inverse question:

    Which direction u in Llama's residual stream, when added to the text
    features of ordinary sentences, changes the predicted cortex most like
    the MS-image contrast does?

``invert_brain_map`` answers it by gradient ascent through TRIBE (its weights
stay frozen; only u is optimised, at a fixed norm). The result is a Llama
direction defined by its *predicted brain effect*, which step 3 injects into
Llama while it answers questions.

Approximations (state them when reporting):
* TRIBE reads the *average* of Llama hidden states over layer groups
  (``tribe_text_layer_groups``). Adding u to every group corresponds to
  injecting u into the residual stream at the first layer TRIBE reads, if the
  later layers carry it forward unchanged. They don't exactly;
  ``closed_loop_check`` measures the real effect by steering TRIBE's own Llama.
* u is added once per 0.5-s feature bin that contains words (bins can hold
  one or two words).
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Which Llama layers does TRIBE read?
# --------------------------------------------------------------------------


def hidden_state_groups(layers: float | tp.Sequence[float], aggregation: str | None,
                        n_hidden_states: int) -> list[tuple[int, int]]:
    """Index ranges [a, b) of HF ``hidden_states`` that TRIBE averages.

    Mirrors neuralset's ``HuggingFaceMixin._aggregate_layers``: relative depths
    are mapped to ``int(depth * (n_hidden_states - 1))``; with "group_mean"
    consecutive indices delimit the groups (the last one inclusive).
    """
    layers = [layers] if isinstance(layers, (int, float)) else list(layers)
    idx = sorted(set(int(x * (n_hidden_states - 1)) for x in layers))
    if aggregation == "group_mean" and len(idx) > 1:
        idx[-1] += 1
        return list(zip(idx[:-1], idx[1:]))
    return [(i, i + 1) for i in idx]  # "mean"/"cat"/None: individual states


def tribe_text_layer_groups(tribe, n_hidden_states: int | None = None) -> list[tuple[int, int]]:
    """Layer groups of the loaded TRIBE checkpoint (Llama-3.2-3B: 29 hidden states)."""
    feat = tribe.data.text_feature
    if n_hidden_states is None:
        from transformers import AutoConfig

        n_hidden_states = AutoConfig.from_pretrained(feat.model_name).num_hidden_layers + 1
    return hidden_state_groups(feat.layers, feat.layer_aggregation, n_hidden_states)


def injection_layer(groups: list[tuple[int, int]]) -> int:
    """Decoder layer whose *output* is the first hidden state TRIBE reads.

    HF convention: ``hidden_states[k]`` is the output of ``layers[k - 1]``.
    """
    return max(groups[0][0] - 1, 0)


# --------------------------------------------------------------------------
# Neutral text for TRIBE, without TTS or speech recognition
# --------------------------------------------------------------------------


def text_events(passages: dict[str, str], words_per_second: float = 2.5, word_duration: float = 0.3,
                sentence_gap: float = 0.5, language: str = "english", use_neuralset: bool = True) -> pd.DataFrame:
    """Word events with synthetic timings, one timeline per passage, then the
    same text transforms TRIBE's demo applies after transcription
    (sentences, 1,024-word left context)."""
    rows = []
    for name, text in passages.items():
        t = 0.5
        for word in text.split():
            rows.append(dict(type="Word", text=word, start=t, duration=word_duration, timeline=name,
                             subject="default", language=language))
            t += 1.0 / words_per_second + (sentence_gap if word.endswith((".", "!", "?")) else 0.0)
    return finish_text_events(pd.DataFrame(rows), use_neuralset)


def finish_text_events(words: pd.DataFrame, use_neuralset: bool = True) -> pd.DataFrame:
    """Add sentences and left context to raw word rows.

    ``use_neuralset=True`` runs TRIBE's own transforms (spaCy sentences, up to
    1,024 words of context); ``False`` uses the preceding words of the
    timeline as context (dry runs, no spaCy needed).
    """
    if not use_neuralset:
        words = words.copy()
        words["context"] = words.groupby("timeline", sort=False).text.transform(
            lambda s: [" ".join(s.iloc[max(0, i - 1023): i + 1]) for i in range(len(s))])
        return words
    from neuralset.events.transforms import AddContextToWords, AddSentenceToWords, AddText, RemoveMissing
    from neuralset.events.utils import standardize_events

    events = standardize_events(words)
    for transform in (AddText(), AddSentenceToWords(max_unmatched_ratio=0.05),
                      AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
                      RemoveMissing()):
        events = transform(events)
    return standardize_events(events)


def text_batches(tribe, events: pd.DataFrame, device: str | None = None, free_gpu_for_llama: bool = True) -> list:
    """TRIBE input batches (with Llama text features) for ``events``.

    TRIBE's text extractor loads Llama-3.2-3B in fp32 (~13 GB). On a 16-GB GPU
    the brain model is parked on the CPU while the features are computed.
    """
    import torch

    model = tribe._model
    device = device or str(model.device)
    if free_gpu_for_llama and torch.cuda.is_available():
        model.to("cpu")
        torch.cuda.empty_cache()
    try:
        loader = tribe.data.get_loaders(events=events, split_to_build="all")["all"]
        batches = [b for b in loader]
    finally:
        model.to(device)
    if "text" not in batches[0].data:
        raise RuntimeError("No text features in the batch: check the Word events")
    return [b.to(device) for b in batches]


def with_text(batch, text: tp.Any):
    """Copy of a TRIBE batch with its text features replaced."""
    return dataclasses.replace(batch, data={**batch.data, "text": text})


# --------------------------------------------------------------------------
# Inversion
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Inversion:
    u: np.ndarray  # unit vector, Llama hidden size
    r: float  # correlation of the achieved cortical change with the target (on the mask)
    feature_norm: float  # |u| used during optimisation (in TRIBE text-feature units)
    history: list[float]
    delta: np.ndarray  # achieved cortical change for +u (all vertices)


def _word_mask(x):
    """(B, T) 1 where a feature bin contains words."""
    return (x.abs().sum(dim=(1, 2)) > 0).to(x.dtype)


def steered_delta(model, batches, base_outs, u):
    """Mean (over batches and time) predicted-cortex change when u is added to
    every word bin of every text-feature layer group."""
    acc = None
    for b, base in zip(batches, base_outs):
        x = b.data["text"].float()
        x2 = x + u.view(1, 1, -1, 1) * _word_mask(x)[:, None, None, :]
        d = (model(with_text(b, x2)) - base).mean(dim=(0, 2))
        acc = d if acc is None else acc + d
    return acc / len(batches)


def text_feature_norm(batches) -> float:
    """Median norm of a word bin's features (per layer group)."""
    norms = []
    for b in batches:
        x = b.data["text"].float()  # B, L, D, T
        m = _word_mask(x).bool()
        n = x.norm(dim=2)  # B, L, T
        norms.append(n.permute(0, 2, 1)[m].flatten())
    import torch

    return float(torch.cat(norms).median())


def invert_brain_map(model, batches, target: np.ndarray, mask: np.ndarray | None = None, rel_norm: float = 0.5,
                     n_steps: int = 300, lr: float = 0.05, seed: int = 0, verbose: bool = True) -> Inversion:
    """Find the unit direction u (Llama hidden size) whose addition to TRIBE's
    text features best reproduces ``target`` (Pearson r over ``mask``).

    ``rel_norm`` sets |u| relative to the median word-feature norm, keeping
    the perturbation moderate so TRIBE stays near its training distribution.
    """
    import torch

    device = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    sel = torch.as_tensor(mask if mask is not None else np.ones(len(target), bool), device=device)
    tgt = torch.as_tensor(target, dtype=torch.float32, device=device)[sel]
    tgt = (tgt - tgt.mean()) / (tgt.std() + 1e-12)
    with torch.no_grad():
        base_outs = [model(b) for b in batches]
    scale = rel_norm * text_feature_norm(batches)
    dim = batches[0].data["text"].shape[2]
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.randn(dim, generator=g).to(device).requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    history = []
    with torch.enable_grad():
        for step in range(n_steps):
            u = scale * p / p.norm()
            d = steered_delta(model, batches, base_outs, u)[sel]
            d = (d - d.mean()) / (d.std() + 1e-12)
            r = (d * tgt).mean()
            opt.zero_grad()
            (-r).backward()
            opt.step()
            history.append(float(r.detach()))
            if verbose and (step % 50 == 0 or step == n_steps - 1):
                LOGGER.info("inversion step %d: r = %.3f", step, history[-1])
    with torch.no_grad():
        u = scale * p / p.norm()
        delta = steered_delta(model, batches, base_outs, u).cpu().numpy()
    unit = (p / p.norm()).detach().cpu().numpy()
    m = mask if mask is not None else slice(None)
    return Inversion(u=unit, r=_corr(delta[m], target[m]), feature_norm=scale, history=history, delta=delta)


def delta_for(model, batches, u_unit: np.ndarray, feature_norm: float) -> np.ndarray:
    """Predicted-cortex change for a given unit direction (no optimisation)."""
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        base_outs = [model(b) for b in batches]
        u = torch.as_tensor(u_unit, dtype=torch.float32, device=device) * feature_norm
        return steered_delta(model, batches, base_outs, u).cpu().numpy()


def symmetry(model, batches, u_unit: np.ndarray, feature_norm: float, target: np.ndarray,
             mask: np.ndarray | None = None) -> dict[str, float]:
    """Directional vs symmetric parts of the effect of +u and -u (cf. the
    steering notebook): steering should flip with the sign of u."""
    m = mask if mask is not None else slice(None)
    plus = delta_for(model, batches, u_unit, feature_norm)
    minus = delta_for(model, batches, -u_unit, feature_norm)
    direc, sym = (plus - minus) / 2, (plus + minus) / 2
    return dict(r_plus=_corr(plus[m], target[m]), r_minus=_corr(minus[m], target[m]),
                r_directional=_corr(direc[m], target[m]),
                directional_over_symmetric=float(np.linalg.norm(direc[m]) / (np.linalg.norm(sym[m]) + 1e-12)))


def sign_flip_targets(d: np.ndarray, n: int, seed: int = 0) -> list[np.ndarray]:
    """Null target maps: the same pair differences averaged with random signs.

    They have the smoothness and scale of a real contrast but no consistent
    meaning, so inverting them shows how well *any* such map can be matched.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = rng.choice([-1.0, 1.0], size=len(d))
        if abs(s.sum()) <= 1:  # balanced, so the true effect cancels
            out.append((d * s[:, None]).mean(0))
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------------------------------
# Closed loop: steer TRIBE's own Llama and look at the predicted brain
# --------------------------------------------------------------------------


@contextlib.contextmanager
def residual_hook(layers, layer: int, vec, coef: float):
    """Add ``coef * vec`` to the output of decoder ``layers[layer]`` (all positions)."""
    import torch

    v = torch.as_tensor(np.asarray(vec, dtype=np.float32))

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + (coef * v).to(device=h.device, dtype=h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def decoder_layers(model):
    """The ModuleList of transformer blocks of a HF decoder (CausalLM or base)."""
    import torch.nn as nn

    for name in ("layers", "model.layers", "language_model.layers", "model.language_model.layers"):
        obj = model
        try:
            for part in name.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise AttributeError("Could not find the decoder layers")


def closed_loop_check(checkpoint: str, main_cache: str | Path, steer_cache_root: str | Path,
                      events: pd.DataFrame, vectors: dict[str, np.ndarray], layer: int, eps_rel: float,
                      targets: dict[str, np.ndarray], mask: np.ndarray | None = None,
                      config_update: dict | None = None,
                      factory: tp.Callable[[Path], tp.Any] | None = None) -> pd.DataFrame:
    """Steer TRIBE's own Llama with each vector and correlate the change in
    the predicted brain with every target map.

    Each condition gets a fresh TRIBE instance with its own feature cache: the
    cache key does not include the hook, so sharing a cache would silently
    reuse unsteered features. The brain model runs on CPU so that the fp32
    Llama fits on a 16-GB GPU; move any other TRIBE instance off the GPU first.
    ``factory(cache_folder)`` replaces ``TribeModel.from_pretrained`` (dry runs).
    """
    import gc

    import torch

    if factory is None:
        from tribev2 import TribeModel

        def factory(cache):
            return TribeModel.from_pretrained(checkpoint, cache_folder=cache, device="cpu",
                                              config_update=config_update)

    def predict(cache, hook_vec=None):
        tribe = factory(cache)
        ctx = contextlib.nullcontext()
        if hook_vec is not None:
            llm = tribe.data.text_feature.model  # loads Llama-3.2-3B (base)
            layers = decoder_layers(llm)
            norm = _mean_residual_norm(llm, tribe.data.text_feature, events, layer)
            ctx = residual_hook(layers, layer, hook_vec, eps_rel * norm)
        with ctx:
            preds, segs = tribe.predict(events=events, verbose=False)
        key = [(str(s.timeline), float(s.start)) for s in segs]
        del tribe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return dict(zip(key, preds))

    base = predict(main_cache)
    rows = []
    m = mask if mask is not None else slice(None)
    for name, vec in vectors.items():
        steered = predict(Path(steer_cache_root) / f"steer_{name}_eps{eps_rel:g}", vec)
        common = sorted(set(base) & set(steered))
        delta = np.mean([steered[k] - base[k] for k in common], 0)
        row = dict(vector=name, eps_rel=eps_rel)
        row.update({f"r_{t}": _corr(delta[m], tm[m]) for t, tm in targets.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_residual_norm(llm, feat, events: pd.DataFrame, layer: int, n_texts: int = 4) -> float:
    """Mean norm of the residual stream after ``layers[layer]`` on a few contexts."""
    import torch

    tok = feat.tokenizer  # the extractor's own tokenizer (left truncation)
    texts = [c for c in events.loc[events.type == "Word", "context"].dropna().tolist() if c][-n_texts:]
    norms = []
    dev = next(llm.parameters()).device
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids[:, -512:].to(dev)
            hs = llm(ids, output_hidden_states=True).hidden_states[layer + 1]
            norms.append(hs.float().norm(dim=-1).mean().item())
    return float(np.mean(norms))
