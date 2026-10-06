"""From predicted time series to answers.

Four analyses, each answering one part of "what does mortality / limited-time
salience induce?":

1. ``prime_contrasts``    What does the prime itself evoke?  (A - B during the prime)
2. ``probe_modulation``   How does the prime re-shape the response to the SAME
                          probe sentences, per probe category?  (+ category x prime
                          interaction against neutral probes)
3. ``carryover``          Does the probe-period shift resemble the prime-evoked
                          pattern (lingering state) or not (re-weighting)?
4. ``decode_prime``       Can the preceding prime be decoded from responses to
                          probes it never shares words with?

Statistics treat stimuli as the random factor (TRIBE predicts a deterministic,
subject-averaged response, so there is no subject variance to exploit): paired
sign-flip permutation tests over prime items or probe sentences, with
max-statistic family-wise correction across ROIs.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from .predict import hrf, load_prediction
from .rois import roi_means
from .stimuli import Trial

LOGGER = logging.getLogger(__name__)

DEFAULT_PAIRS = (
    ("MS", "PAIN"),  # existential vs merely aversive threat (TMT's canonical contrast)
    ("MS", "NONFINAL"),  # is finality needed, or is self/body focus enough?
    ("MS", "DISTRESS"),  # is it finality, or first-person negative affect?
    ("MS_STRUCT", "PAIN"),  # impersonal finality
    ("MS", "MS_STRUCT"),  # does the self-reference matter?
    ("PAIN", "NEU"),  # sanity check: aversive imagery vs neutral
    ("LT", "ET"),  # limited vs expansive future time (SST's canonical contrast)
)


@dataclasses.dataclass
class AnalysisConfig:
    # "glm": one HRF-convolved regressor per segment, fitted jointly within each
    #   trial, so overlapping hemodynamic tails (prime -> first probe) are
    #   separated. Same logic as the GLM the TRIBE paper fits for language tasks.
    # "window": mean of the predicted signal in [onset + lag, offset + lag + extra].
    method: tp.Literal["glm", "window"] = "glm"
    lag: float = 5.0  # window method only: hemodynamic delay (paper: peak ~5 s)
    extra: float = 1.0  # window method only: seconds added after the segment offset
    pairs: tp.Sequence[tuple[str, str]] = DEFAULT_PAIRS
    control_category: str = "NEUT"
    # Extra category-vs-category interactions, e.g. (("WV_NORM", "WV_EXCL"),):
    # does the prime shift one worldview axis more than the other?
    category_contrasts: tp.Sequence[tuple[str, str]] = ()
    # Pairs of contrasts whose probe-shift maps are correlated (exploratory),
    # e.g. (("MS-PAIN", "MS_STRUCT-PAIN"),): self vs impersonal finality.
    convergence: tp.Sequence[tuple[str, str]] = ()
    n_perm: int = 5000
    seed: int = 0
    # Confirmatory tests, fixed before looking at predictions (see configs/default.yaml).
    hypotheses: tp.Sequence[dict] = ()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _global_moments(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    total, total_sq, n = 0.0, 0.0, 0
    for p in paths:
        _, preds = load_prediction(p)
        total = total + preds.sum(0, dtype=np.float64)
        total_sq = total_sq + (preds.astype(np.float64) ** 2).sum(0)
        n += len(preds)
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean**2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def window_mean(times: np.ndarray, preds: np.ndarray, start: float, stop: float) -> np.ndarray:
    sel = (times >= start) & (times < stop)
    if not sel.any():  # window shorter than a TR: take the nearest sample
        sel = np.abs(times - (start + stop) / 2) == np.abs(times - (start + stop) / 2).min()
    return preds[sel].mean(0)


def glm_design(times: np.ndarray, onsets, durations, dt: float = 0.1, n_cosines: int | None = None,
               derivatives: bool = True) -> np.ndarray:
    """Design matrix: one HRF-convolved boxcar per segment, then (optionally)
    one temporal-derivative regressor per segment, then drift terms.

    The HRF is normalised to unit area, so a beta is the sustained response
    amplitude (in the units of the data) during the segment. Derivative
    regressors absorb small mismatches between the canonical HRF and the
    shape/latency of TRIBE's predicted response, which would otherwise leak
    part of a long prime's response into the betas of the probes after it.
    """
    t_hi = np.arange(0, times.max() + 32, dt)
    h = hrf(dt, 32.0)
    h = h / h.sum()
    cols, derivs = [], []
    for on, du in zip(onsets, durations):
        box = ((t_hi >= on) & (t_hi < on + du)).astype(float)
        conv = np.convolve(box, h)[: len(t_hi)]
        cols.append(np.interp(times, t_hi, conv))
        derivs.append(np.interp(times, t_hi, np.gradient(conv, dt)))
    if derivatives:
        cols += derivs
    span = times.max() - times.min() + 1
    n_cosines = n_cosines if n_cosines is not None else int(span // 128)  # 1/128 Hz high-pass
    x = (times - times.min()) / span
    drift = [np.ones_like(times)] + [np.cos(np.pi * k * (x + 0.5 / len(times))) for k in range(1, n_cosines + 1)]
    return np.column_stack(cols + drift)


def glm_betas(times: np.ndarray, preds: np.ndarray, onsets, durations) -> np.ndarray:
    """``[n_segments, n_vertices]`` least-squares betas."""
    X = glm_design(times, onsets, durations)
    beta, *_ = np.linalg.lstsq(X, preds, rcond=None)
    return beta[: len(onsets)]


def load_responses(
    trials: list[Trial], timings: pd.DataFrame, pred_dir: str | Path, cfg: AnalysisConfig
) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per (trial, segment): response of every vertex to that segment.

    Every vertex is first z-scored across all time points of all trials, so
    effects are in units of that vertex's standard deviation over the
    experiment. See ``AnalysisConfig.method`` for how a segment's response
    is estimated.
    """
    pred_dir = Path(pred_dir)
    paths = [pred_dir / f"{t.trial_id}.npz" for t in trials]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} prediction files missing, e.g. {missing[0]}")
    mean, std = _global_moments(paths)
    by_trial = {t.trial_id: t for t in trials}
    rows, resp = [], []
    for path in paths:
        trial = by_trial[path.stem]
        times, preds = load_prediction(path)
        preds = (preds - mean) / std
        segs = timings[timings.trial_id == trial.trial_id]
        if cfg.method == "glm":
            resp.extend(glm_betas(times, preds, segs.onset.to_numpy(), segs.duration.to_numpy()))
        for seg in segs.itertuples():
            if cfg.method == "window":
                start = seg.onset + cfg.lag
                resp.append(window_mean(times, preds, start, start + seg.duration + cfg.extra))
            rows.append(
                dict(
                    trial_id=trial.trial_id, prime_cond=trial.prime_cond, prime_id=trial.prime_id,
                    item_index=trial.item_index, delay=trial.delay, probe_set=trial.probe_set,
                    kind=seg.kind, category=seg.category, text=seg.text, position=seg.position,
                )
            )
    return pd.DataFrame(rows), np.stack(resp)


def evoked_timecourses(
    trials: list[Trial], timings: pd.DataFrame, pred_dir: str | Path, rois: dict[str, np.ndarray],
    kind: str = "probe", window: tuple[float, float] = (-2, 16),
) -> pd.DataFrame:
    """ROI time courses locked to segment onsets (raw units). Use it to check
    that ``AnalysisConfig.lag`` sits on the peak of the predicted response."""
    pred_dir = Path(pred_dir)
    rows = []
    lags = np.arange(window[0], window[1] + 1)
    for trial in trials:
        times, preds = load_prediction(pred_dir / f"{trial.trial_id}.npz")
        roi_ts = roi_means(preds, rois)
        segs = timings[(timings.trial_id == trial.trial_id) & (timings.kind == kind)]
        for seg in segs.itertuples():
            for lag in lags:
                i = np.argmin(np.abs(times - (seg.onset + lag)))
                for r, name in enumerate(rois):
                    rows.append(dict(prime_cond=trial.prime_cond, delay=trial.delay, category=seg.category,
                                     lag=lag, roi=name, value=roi_ts[i, r]))
    return pd.DataFrame(rows).groupby(["prime_cond", "delay", "category", "roi", "lag"], as_index=False).value.mean()


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def sign_flip_test(diffs: np.ndarray, n_perm: int = 5000, seed: int = 0) -> dict[str, np.ndarray]:
    """Paired permutation test of ``mean(diffs) != 0`` for each column.

    ``diffs``: ``[n_units, n_tests]``. Returns mean, t, uncorrected p, and
    family-wise p using the max-|t| distribution across columns. With
    n_units <= 12 all 2**n sign patterns are enumerated (exact test).
    """
    diffs = np.atleast_2d(np.asarray(diffs, dtype=np.float64))
    if diffs.shape[0] == 1:
        diffs = diffs.T
    n = diffs.shape[0]
    if n <= 12:
        signs = np.array(list(itertools.product([1, -1], repeat=n)), dtype=float)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([1.0, -1.0], size=(n_perm, n))
        signs[0] = 1.0

    def tstat(x):
        sd = x.std(-2, ddof=1)
        return x.mean(-2) / np.where(sd > 0, sd, np.inf) * np.sqrt(n)

    t_obs = tstat(diffs)
    t_null = np.stack([tstat(diffs * s[:, None]) for s in signs])  # [n_perm, n_tests]
    p_unc = (np.abs(t_null) >= np.abs(t_obs) - 1e-12).mean(0)
    max_null = np.abs(t_null).max(1)
    p_fwe = (max_null[:, None] >= np.abs(t_obs)[None, :] - 1e-12).mean(0)
    return dict(mean=diffs.mean(0), t=t_obs, p_unc=p_unc, p_fwe=p_fwe, n=np.full(t_obs.shape, n))


def label_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0) -> dict[str, np.ndarray]:
    """Unpaired test of ``mean(a) - mean(b)`` per column (max-stat FWE)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, na = np.concatenate([a, b]), len(a)
    rng = np.random.default_rng(seed)
    obs = a.mean(0) - b.mean(0)
    null = np.empty((n_perm, a.shape[1]))
    null[0] = obs
    for i in range(1, n_perm):
        perm = rng.permutation(len(pooled))
        null[i] = pooled[perm[:na]].mean(0) - pooled[perm[na:]].mean(0)
    p_unc = (np.abs(null) >= np.abs(obs) - 1e-12).mean(0)
    p_fwe = (np.abs(null).max(1)[:, None] >= np.abs(obs)[None, :] - 1e-12).mean(0)
    return dict(diff=obs, p_unc=p_unc, p_fwe=p_fwe)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(q, 1)
    return out


def _roi_table(stats: dict, roi_names: list[str], **meta) -> pd.DataFrame:
    df = pd.DataFrame({k: v for k, v in stats.items()})
    df.insert(0, "roi", roi_names)
    for k, v in reversed(list(meta.items())):
        df.insert(0, k, v)
    return df


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------


def _paired(meta: pd.DataFrame, resp: np.ndarray, sel: pd.Series, a: str, b: str, key: list[str]):
    """Average A and B responses within ``key`` and align them: ``[n_keys, V]`` each."""
    out = []
    for cond in (a, b):
        m = meta[sel & (meta.prime_cond == cond)]
        grouped = pd.DataFrame(resp[m.index.to_numpy()], index=pd.MultiIndex.from_frame(m[key]))
        out.append(grouped.groupby(level=list(range(len(key)))).mean())
    common = out[0].index.intersection(out[1].index)
    if len(common) == 0:
        raise ValueError(f"No matched units between {a} and {b} for key {key}")
    return out[0].loc[common].to_numpy(), out[1].loc[common].to_numpy(), common


def prime_contrasts(meta, resp, rois, cfg: AnalysisConfig):
    """A - B during the prime itself, paired by prime item index (pooled over delays)."""
    tables, maps = [], {}
    names = list(rois)
    for a, b in cfg.pairs:
        sel = meta.kind == "prime"
        if not ((meta.prime_cond == a) & sel).any() or not ((meta.prime_cond == b) & sel).any():
            continue
        ra, rb, _ = _paired(meta, resp, sel, a, b, ["item_index"])
        diff = ra - rb
        maps[f"prime_{a}-{b}"] = diff.mean(0)
        stats = sign_flip_test(roi_means(diff, rois), cfg.n_perm, cfg.seed)
        tables.append(_roi_table(stats, names, contrast=f"{a}-{b}"))
    tab = pd.concat(tables, ignore_index=True)
    tab["q_fdr"] = bh_fdr(tab.p_unc)
    return tab, maps


def probe_modulation(meta, resp, rois, cfg: AnalysisConfig):
    """Effect of the prime on identical probes, per category and delay.

    ``p_fwe`` uses probe sentences as units (each heard after both A and B):
    it generalises over probes but treats the prime items as fixed.
    ``p_fwe_items`` uses prime items as units. The category x prime
    interaction (is the A-B shift for category c larger than for neutral
    probes heard in the same trials?) cancels item-level generic shifts and
    is the recommended confirmatory test.
    """
    tables, inter, maps = [], [], {}
    names = list(rois)
    probes = meta.kind == "probe"
    for (a, b), delay in itertools.product(cfg.pairs, sorted(meta.delay.unique())):
        sel = probes & (meta.delay == delay)
        if not ((meta.prime_cond == a) & sel).any() or not ((meta.prime_cond == b) & sel).any():
            continue
        ra, rb, keys = _paired(meta, resp, sel, a, b, ["category", "text"])
        diff = ra - rb
        cats = np.array([k[0] for k in keys])
        diff_roi = roi_means(diff, rois)
        # Same contrast with prime items as units (probes averaged within item):
        # the test that generalises to new prime passages, but low-powered.
        ia, ib, ikeys = _paired(meta, resp, sel, a, b, ["category", "item_index"])
        item_cats = np.array([k[0] for k in ikeys])
        item_diff_roi = roi_means(ia - ib, rois)
        for cat in sorted(set(cats)):
            d = diff_roi[cats == cat]
            maps[f"probe_{a}-{b}_{delay}_{cat}"] = diff[cats == cat].mean(0)
            tab = _roi_table(sign_flip_test(d, cfg.n_perm, cfg.seed), names,
                             contrast=f"{a}-{b}", delay=delay, category=cat)
            tab["p_fwe_items"] = sign_flip_test(item_diff_roi[item_cats == cat], cfg.n_perm, cfg.seed)["p_fwe"]
            tables.append(tab)
        cat_pairs = [(c, cfg.control_category) for c in sorted(set(cats)) if c != cfg.control_category]
        cat_pairs += [tuple(cp) for cp in cfg.category_contrasts]
        for c1, c2 in cat_pairs:
            if c1 not in cats or c2 not in cats:
                continue
            st = label_permutation_test(diff_roi[cats == c1], diff_roi[cats == c2], cfg.n_perm, cfg.seed)
            inter.append(_roi_table(st, names, contrast=f"{a}-{b}", delay=delay, category=f"{c1}-vs-{c2}"))
    tab = pd.concat(tables, ignore_index=True)
    tab["q_fdr"] = bh_fdr(tab.p_unc)  # across everything tested, as a global guard
    inter_tab = pd.concat(inter, ignore_index=True) if inter else pd.DataFrame()
    if len(inter_tab):
        inter_tab["q_fdr"] = bh_fdr(inter_tab.p_unc)
    return tab, inter_tab, maps


def carryover(meta, resp, cfg: AnalysisConfig) -> pd.DataFrame:
    """Spatial correlation between the prime-evoked contrast map and the
    probe-period shift (all probes). High r: the prime leaves a lingering
    state that tints everything after it. Low r alongside category-specific
    probe effects: the prime re-weights how specific content is processed.
    Null: exact sign-flips over prime items."""
    rows = []
    for (a, b), delay in itertools.product(cfg.pairs, sorted(meta.delay.unique())):
        try:
            pa, pb, _ = _paired(meta, resp, (meta.kind == "prime") & (meta.delay == delay), a, b, ["item_index"])
            qa, qb, _ = _paired(meta, resp, (meta.kind == "probe") & (meta.delay == delay), a, b, ["item_index"])
        except (ValueError, KeyError):
            continue
        prime_map, diffs = (pa - pb).mean(0), qa - qb
        r_obs = np.corrcoef(prime_map, diffs.mean(0))[0, 1]
        if len(diffs) <= 12:
            signs = np.array(list(itertools.product([1, -1], repeat=len(diffs))), dtype=float)
        else:
            signs = np.random.default_rng(cfg.seed).choice([1.0, -1.0], size=(2000, len(diffs)))
            signs[0] = 1.0
        null = np.array([np.corrcoef(prime_map, (diffs * s_[:, None]).mean(0))[0, 1] for s_ in signs])
        rows.append(dict(contrast=f"{a}-{b}", delay=delay, r=r_obs, p=(np.abs(null) >= abs(r_obs) - 1e-12).mean()))
    return pd.DataFrame(rows)


def decode_prime(meta, resp, rois, cfg: AnalysisConfig) -> pd.DataFrame:
    """Decode the prime (A vs B) from ROI responses to the probes. Chance = 0.5.

    Cross-validation holds out whole prime items (with every probe heard after
    them): probes from the same trial share an item-specific offset, so
    holding out single probes would let the classifier recognise the trial
    rather than the condition. Requires scikit-learn."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        LOGGER.warning("scikit-learn not installed: skipping decoding")
        return pd.DataFrame()
    rows = []
    feats = roi_means(resp, rois)
    for (a, b), delay in itertools.product(cfg.pairs, sorted(meta.delay.unique())):
        m = meta[(meta.kind == "probe") & (meta.delay == delay) & meta.prime_cond.isin([a, b])]
        if m.prime_cond.nunique() < 2:
            continue
        X, y = feats[m.index], (m.prime_cond == a).to_numpy().astype(int)
        groups = m.item_index.to_numpy()
        n_splits = len(set(groups))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        cv = GroupKFold(n_splits=n_splits)
        acc = cross_val_score(clf, X, y, groups=groups, cv=cv).mean()
        rng = np.random.default_rng(cfg.seed)
        null = []
        trial_ids = m.trial_id.to_numpy()
        for _ in range(200):  # swap A/B labels of whole trials within each prime item
            y_perm = y.copy()
            for g in set(groups):
                tids = np.unique(trial_ids[groups == g])
                for tid, lab in zip(tids, rng.permutation([y[trial_ids == t][0] for t in tids])):
                    y_perm[trial_ids == tid] = lab
            null.append(cross_val_score(clf, X, y_perm, groups=groups, cv=cv).mean())
        rows.append(dict(contrast=f"{a}-{b}", delay=delay, accuracy=acc,
                         p=(np.sum(np.array(null) >= acc) + 1) / (len(null) + 1)))
    return pd.DataFrame(rows)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1)
    return out


def test_hypotheses(results: dict[str, pd.DataFrame], hypotheses: tp.Sequence[dict]) -> pd.DataFrame:
    """Look up each pre-registered (table, contrast, delay, category, roi) test,
    convert to a one-sided p in the predicted direction, and Holm-correct
    within each hypothesis ``family`` (default: one family for all)."""
    rows = []
    for h in hypotheses:
        df = results[h["table"]]
        sel = np.ones(len(df), bool)
        for key in ("contrast", "delay", "category", "roi"):
            if key in h:
                sel &= (df[key] == h[key]).to_numpy()
        if sel.sum() != 1:
            raise ValueError(f"Hypothesis {h['name']!r} matches {sel.sum()} rows, expected 1")
        r = df[sel].iloc[0]
        est = r["diff"] if "diff" in r else r["mean"]
        sign = 1 if h.get("direction", "+") == "+" else -1
        p_one = r.p_unc / 2 if np.sign(est) == sign else 1 - r.p_unc / 2
        rows.append(dict(name=h["name"], family=h.get("family", "all"), theory=h.get("theory", ""),
                         test=h["table"], contrast=r.contrast,
                         delay=r.get("delay", ""), category=r.get("category", ""), roi=r.roi,
                         predicted=h.get("direction", "+"), effect=est, p_one_sided=p_one))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = df.groupby("family").p_one_sided.transform(lambda p: holm(p.to_numpy()))
        df["supported"] = df.p_holm < 0.05
    return df


def convergence(meta, resp, cfg: AnalysisConfig) -> pd.DataFrame:
    """Do two contrasts re-shape the probes the same way? (exploratory)

    Brain-level analogue of the item-profile convergence test of the LLM
    steering study (self vs impersonal finality), with the same safeguards:

    * Cross-half estimation (removes shared noise). A and B are estimated
      from disjoint halves of the prime items; ``r_cross`` correlates A on
      one half with B on the other, averaged over both assignments.
    * Reliability. ``rel_a`` / ``rel_b`` are Spearman-Brown corrected
      split-half reliabilities of each map (NaN when the half-correlation is
      <= 0, i.e. the map is noise). Read r together with them.
    * Reference (removes shared-baseline structure; the analogue of the
      notebook's calibration curve). When A = X - Z and B = Y - Z share the
      baseline Z, both maps contain "minus Z's own effect" and converge for
      that reason alone. ``r_ref_max`` is the largest r_cross obtained by
      replacing X with any other condition W (W - Z vs Y - Z). Convergence
      beyond the shared baseline requires ``r_cross > r_ref_max``.
    """
    rows = []
    conds = sorted(meta.prime_cond.unique())
    for c1, c2 in cfg.convergence:
        (x, z1), (y, z2) = (c.split("-", 1) for c in (c1, c2))
        for delay in sorted(meta.delay.unique()):
            sel = (meta.kind == "probe") & (meta.delay == delay)
            try:
                A, B = _item_maps(meta, resp, sel, x, z1), _item_maps(meta, resp, sel, y, z2)
            except ValueError:
                continue
            refs = []
            if z1 == z2:
                for w in conds:
                    if w not in (x, y, z1):
                        try:
                            refs.append((w, _item_maps(meta, resp, sel, w, z1)))
                        except ValueError:
                            pass
            for cat in sorted(set(A) & set(B)):
                rc, rel_a, rel_b = _cross_half(A[cat], B[cat])
                ref = [_cross_half(R[cat], B[cat])[0] for _, R in refs if cat in R]
                row = dict(contrast_a=c1, contrast_b=c2, delay=delay, category=cat,
                           r_cross=rc, rel_a=rel_a, rel_b=rel_b)
                if ref:
                    row.update(r_ref_max=float(np.nanmax(ref)), r_ref_mean=float(np.nanmean(ref)),
                               beyond_baseline=bool(rc > np.nanmax(ref)))
                rows.append(row)
    return pd.DataFrame(rows)


def _cross_half(da: dict[int, np.ndarray], db: dict[int, np.ndarray]) -> tuple[float, float, float]:
    items = sorted(set(da) & set(db))
    if len(items) < 2:
        return np.nan, np.nan, np.nan
    h1, h2 = items[: len(items) // 2], items[len(items) // 2:]

    def m(d, h):
        return np.mean([d[i] for i in h], 0)

    def r(u, v):
        return float(np.corrcoef(u, v)[0, 1])

    def sb(x):
        return 2 * x / (1 + x) if x > 0 else np.nan

    r_cross = float(np.mean([r(m(da, h1), m(db, h2)), r(m(da, h2), m(db, h1))]))
    return r_cross, sb(r(m(da, h1), m(da, h2))), sb(r(m(db, h1), m(db, h2)))


def _item_maps(meta, resp, sel, a, b) -> dict[str, dict[int, np.ndarray]]:
    """{category: {item_index: A-B map averaged over that item's probes}}."""
    ra, rb, keys = _paired(meta, resp, sel, a, b, ["category", "item_index"])
    out: dict[str, dict[int, np.ndarray]] = {}
    for (cat, item), d in zip(keys, ra - rb):
        out.setdefault(cat, {})[item] = d
    return out


def run_all(meta, resp, rois, cfg: AnalysisConfig, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    prime_tab, prime_maps = prime_contrasts(meta, resp, rois, cfg)
    probe_tab, inter_tab, probe_maps = probe_modulation(meta, resp, rois, cfg)
    results = dict(
        prime_contrasts=prime_tab, probe_modulation=probe_tab, probe_interaction=inter_tab,
        carryover=carryover(meta, resp, cfg), decoding=decode_prime(meta, resp, rois, cfg),
    )
    results["convergence"] = convergence(meta, resp, cfg)
    results["hypotheses"] = test_hypotheses(results, cfg.hypotheses)
    for name, df in results.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    for name, vec in {**prime_maps, **probe_maps}.items():
        np.save(out_dir / "maps" / f"{name}.npy", vec.astype(np.float32))
    (out_dir / "summary.md").write_text(summarize(results))
    return results


def summarize(results: dict[str, pd.DataFrame], alpha: float = 0.05) -> str:
    lines = ["# Results summary", "",
             "Effects are in SD units of each vertex's predicted signal. Listed effects "
             f"pass both p_fwe < {alpha} (max-statistic permutation across ROIs) and "
             f"q_fdr < {alpha} (Benjamini-Hochberg across every test in the table).", ""]
    hyp = results.get("hypotheses", pd.DataFrame())
    if len(hyp):
        lines += ["## 0. Pre-registered hypotheses (one-sided, Holm-corrected within family)", ""]
        lines += _table(hyp[["name", "family", "theory", "contrast", "delay", "category", "roi", "predicted",
                             "effect", "p_one_sided", "p_holm", "supported"]])
        lines += ["", "Everything below is exploratory.", ""]
    pt = results["prime_contrasts"]
    lines += ["## 1. What the primes evoke (prime window)", ""]
    lines += _top(pt, ["contrast"], alpha) or ["Nothing survives. Note: prime contrasts are paired over "
                                                "prime items; with 4 items the smallest exact p is 0.125, "
                                                "so add items to stimuli/primes.yaml for power."]
    lines += ["", "## 2. How primes re-shape responses to identical probes", ""]
    lines += _top(results["probe_modulation"], ["contrast", "delay", "category"], alpha) or ["Nothing survives."]
    inter = results["probe_interaction"]
    lines += ["", "## 3. Category-specific modulation (vs neutral probes)", ""]
    lines += (_top(inter, ["contrast", "delay", "category"], alpha, est="diff") if len(inter) else []) or ["Nothing survives."]
    lines += ["", "## 4. Carry-over of the prime pattern into the probe period", ""]
    lines += _table(results["carryover"]) if len(results["carryover"]) else ["n/a"]
    lines += ["", "## 5. Decoding the prime from probe responses (chance = 0.5)", ""]
    lines += _table(results["decoding"]) if len(results["decoding"]) else ["n/a"]
    conv = results.get("convergence", pd.DataFrame())
    if len(conv):
        lines += ["", "## 6. Convergence between contrasts (cross-half spatial r of probe-shift maps; descriptive)", ""]
        lines += _table(conv)
    return "\n".join(lines) + "\n"


def _top(df: pd.DataFrame, by: list[str], alpha: float, est: str = "mean") -> list[str]:
    sig = df[(df.p_fwe < alpha) & (df.q_fdr < alpha)]
    out = []
    for keys, g in sig.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        effects = ", ".join(f"{r.roi} {getattr(r, est):+.2f} (p={r.p_fwe:.3g})"
                            for r in g.sort_values("p_fwe").itertuples())
        out.append(f"- **{' / '.join(map(str, keys))}**: {effects}")
    return out


def _table(df: pd.DataFrame) -> list[str]:
    return ["```", df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), "```"]
