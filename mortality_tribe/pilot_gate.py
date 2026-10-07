"""Gate step A3: do text and images agree in the predicted brain, and which
modality should the steering vectors come from?

The comparison is a multitrait-multimethod design in brain space (as in the
steering notebook's MTMM, Campbell & Fiske 1959):

    trait  = finality (mortality)  vs  negative affect (control)
    method = text (your contrast pairs)  vs  image (matched image pairs)

* **Convergent validity**: finality-text and finality-image maps correlate.
* **Discriminant validity**: that correlation beats the heterotrait-
  heteromethod ones (finality-image vs negative-text, negative-image vs
  finality-text). Otherwise the "agreement" is generic negativity.

``recommend_modality`` turns this into a decision with a rule fixed before
the data are seen (see its docstring).
"""

from __future__ import annotations

import itertools
import typing as tp

import numpy as np
import pandas as pd

from .pilot_images import compare_contrasts, contrast_reliability_null


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mtmm(contrasts: dict[str, np.ndarray], mask: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlations between the mean maps of every contrast, and each
    contrast's consistency (sign-flip p) and split-half reliability.

    ``contrasts``: ``{name: pair differences [n_pairs, V]}``, e.g.
    ``{"finality|text": ..., "finality|image": ..., "negative|text": ...}``.
    """
    m = mask if mask is not None else slice(None)
    means = {k: d[:, m].mean(0) for k, d in contrasts.items()}
    names = list(means)
    corr = pd.DataFrame([[_corr(means[a], means[b]) for b in names] for a in names], index=names, columns=names)
    rows = []
    for k, d in contrasts.items():
        rel = contrast_reliability_null(d, mask)
        half = compare_contrasts(d[0::2], d[1::2], mask, n_perm=10)["r"] if len(d) >= 4 else np.nan
        rows.append(dict(contrast=k, n_pairs=len(d), consistency_p=rel["p_perm"],
                         split_half=2 * half / (1 + half) if half == half and half > 0 else np.nan))
    return corr, pd.DataFrame(rows).set_index("contrast")


def convergence(contrasts: dict[str, np.ndarray], pairs: tp.Sequence[tuple[str, str]],
                mask: np.ndarray | None = None, n_perm: int = 5000) -> pd.DataFrame:
    """Sign-flip tested correlation for each requested pair of contrasts."""
    rows = []
    for a, b in pairs:
        if a in contrasts and b in contrasts:
            res = compare_contrasts(contrasts[a], contrasts[b], mask, n_perm=n_perm)
            rows.append(dict(a=a, b=b, **res))
    return pd.DataFrame(rows)


def recommend_modality(info: pd.DataFrame, conv: pd.DataFrame, target_text: str = "finality|text",
                       target_image: str = "finality|image", control_text: str = "negative|text",
                       control_image: str = "negative|image", alpha_consistency: float = 0.01,
                       alpha_convergence: float = 0.05) -> tuple[str, list[str]]:
    """Decision rule, fixed in advance.

    1. A modality is *usable* if its finality contrast is consistent across
       pairs (sign-flip p < ``alpha_consistency``).
    2. Text and image *converge* if r(finality-text, finality-image) is
       positive with p < ``alpha_convergence`` and larger than both
       heterotrait-heteromethod correlations (when those were measured).
    3. Both usable and convergent -> "text": more pairs, the construct of the
       LLM study, and natively expressible in Llama; the images validate it.
       Both usable, not convergent -> "both": they capture different things
       (here: AI-self finality vs human death), so steer with each and report
       both. Only one usable -> that one. Neither -> "caa_only": steer with
       the direct contrast-pair vectors only.
    """
    reasons = []
    usable = {}
    for mod, key in (("text", target_text), ("image", target_image)):
        p = info.loc[key, "consistency_p"] if key in info.index else np.nan
        usable[mod] = bool(p == p and p < alpha_consistency)
        reasons.append(f"{key}: consistency p = {p:.4g} -> {'usable' if usable[mod] else 'not consistent'}")

    def r_of(a, b):
        hit = conv[((conv.a == a) & (conv.b == b)) | ((conv.a == b) & (conv.b == a))]
        return (float(hit.r.iloc[0]), float(hit.p_perm.iloc[0])) if len(hit) else (np.nan, np.nan)

    r_conv, p_conv = r_of(target_text, target_image)
    hetero = [r for r, _ in (r_of(target_image, control_text), r_of(control_image, target_text)) if r == r]
    convergent = bool(r_conv == r_conv and r_conv > 0 and p_conv < alpha_convergence
                      and all(r_conv > h for h in hetero))
    reasons.append(f"convergence r = {r_conv:.3f} (p = {p_conv:.4g}); heterotrait-heteromethod r = "
                   f"{', '.join(f'{h:.3f}' for h in hetero) or 'n/a'} -> {'convergent' if convergent else 'not convergent'}")
    if usable["text"] and usable["image"]:
        choice = "text" if convergent else "both"
    elif usable["text"] or usable["image"]:
        choice = "text" if usable["text"] else "image"
    else:
        choice = "caa_only"
    return choice, reasons


def mtmm_pairs(traits: tp.Sequence[str] = ("finality", "negative"),
               methods: tp.Sequence[str] = ("text", "image")) -> list[tuple[str, str]]:
    """Every pair of trait|method cells (for ``convergence``)."""
    cells = [f"{t}|{m}" for t, m in itertools.product(traits, methods)]
    return list(itertools.combinations(cells, 2))
