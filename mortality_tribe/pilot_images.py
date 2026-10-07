"""Pilot step 1: show mortality-salience (MS) and limited-time (LT) images to
TRIBE v2 and compare the predicted brain responses.

Design: every target image has a *matched control* that keeps the scene and
changes the one detail that carries the meaning (coffin vs storage chest,
hourglass running out vs just turned). Contrasts are always target - control
within a pair, so what is compared between MS and LT is the meaning-specific
part of the predicted response, not "seeing a picture".

Each image becomes its own short video timeline: ``lead`` s of grey, the image
for ``show`` s, then ``tail`` s of grey. An identical all-grey timeline is the
baseline that is subtracted from every image response.

Time base: TRIBE's ``predict()`` row *t* is the predicted BOLD at *t* + 5 s
(its fMRI targets are read 5 s after the stimulus window), so responses appear
close to stimulus onset. The response window is therefore measured from the
data (``estimate_peak_lag``) instead of assumed.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

GREY = (128, 128, 128)

# (family, target prompt, matched-control prompt). Same seed for both members
# of a pair keeps composition similar in diffusion models.
PAIR_PROMPTS: dict[str, list[tuple[str, str, str]]] = {
    "MS": [
        ("coffin", "a closed wooden coffin in a quiet room, soft window light, photograph",
         "a closed wooden storage chest in a quiet room, soft window light, photograph"),
        ("grave", "a weathered gravestone on green grass in a park, overcast day, photograph",
         "a weathered stone bench on green grass in a park, overcast day, photograph"),
        ("hearse", "a black hearse parked on a quiet city street, photograph",
         "a black delivery van parked on a quiet city street, photograph"),
        ("skull", "a human skull on a wooden table, plain background, photograph",
         "a round ceramic vase on a wooden table, plain background, photograph"),
        ("funeral", "mourners in black clothes standing around an open grave, photograph",
         "people in dark coats standing around a garden flower bed, photograph"),
        ("wreath", "a funeral wreath of white lilies with a black ribbon on a stand, photograph",
         "a decorative wreath of white flowers with a green ribbon on a stand, photograph"),
    ],
    "LT": [
        ("hourglass", "an hourglass with almost all the sand already in the bottom bulb, close-up photograph",
         "an hourglass with almost all the sand still in the top bulb, close-up photograph"),
        ("candle", "a candle burned down to a tiny stub, flame about to go out, photograph",
         "a tall new candle burning steadily, photograph"),
        ("flower", "a wilted drooping flower with fallen petals in a glass vase, photograph",
         "a fresh upright flower in full bloom in a glass vase, photograph"),
        ("tree", "a nearly bare tree with its last few autumn leaves, photograph",
         "a tree full of fresh green spring leaves, photograph"),
        ("calendar", "a wall calendar with every day crossed out except the last one, photograph",
         "a wall calendar with no days crossed out, photograph"),
        ("battery", "a phone screen showing a nearly empty red battery icon, close-up photograph",
         "a phone screen showing a full green battery icon, close-up photograph"),
    ],
    # Negative-affect control (sensory unpleasantness, nothing final): the image
    # counterpart of the neg_sensory sentence pairs, for the text-vs-image gate.
    "NEG": [
        ("bread", "a slice of bread covered in green and grey mold on a plate, photograph",
         "a fresh slice of bread on a plate, photograph"),
        ("fruit", "rotten brown apples with soft dark spots in a bowl, photograph",
         "fresh shiny red apples in a bowl, photograph"),
        ("sink", "a dirty kitchen sink full of greasy dishes and food scraps, photograph",
         "a clean empty kitchen sink, photograph"),
        ("milk", "a glass of spoiled curdled milk with lumps, photograph",
         "a glass of fresh smooth milk, photograph"),
        ("trash", "an overflowing trash bin with garbage spilling onto the floor, photograph",
         "an empty trash bin on a clean floor, photograph"),
        ("toilet", "a filthy stained public toilet, photograph",
         "a clean white public toilet, photograph"),
    ],
}


# --------------------------------------------------------------------------
# Stimuli
# --------------------------------------------------------------------------


def generate_pairs_diffusers(
    out_dir: str | Path,
    model_id: str = "stabilityai/sdxl-turbo",
    seeds: tp.Sequence[int] = (0, 1),
    size: int = 512,
    steps: int = 2,
    device: str = "cuda",
    prompts: dict[str, list[tuple[str, str, str]]] = PAIR_PROMPTS,
) -> pd.DataFrame:
    """Generate target/control image pairs with a text-to-image model.

    Target and control share the seed, so the layout tends to match and the
    prompt difference carries most of the change. Inspect the contact sheet:
    discard pairs where the generator ignored the critical detail.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    out_dir = Path(out_dir)
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to(device)
    rows = []
    for cond, families in prompts.items():
        for family, target, control in families:
            for seed in seeds:
                for role, prompt in (("target", target), ("control", control)):
                    path = out_dir / cond / f"{family}_s{seed}_{role}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        gen = torch.Generator(device).manual_seed(int(seed))
                        img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0,
                                   generator=gen, height=size, width=size).images[0]
                        img.convert("RGB").save(path)
                    rows.append(dict(cond=cond, family=family, seed=seed, role=role,
                                     pair_id=f"{cond}_{family}_s{seed}", path=str(path), prompt=prompt))
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def draw_pairs_procedural(out_dir: str | Path, size: int = 256, seeds: tp.Sequence[int] = (0, 1)) -> pd.DataFrame:
    """Offline fallback: crude line drawings of minimal pairs.

    Only for checking that the pipeline runs (no GPU or downloads needed).
    TRIBE's video model was trained on natural video, so drawings like these
    say little about mortality salience. Do not interpret their results.
    """
    from PIL import Image, ImageDraw

    out_dir = Path(out_dir)
    rng_master = np.random.default_rng(0)

    def canvas():
        img = Image.new("RGB", (size, size), (200, 200, 195))
        return img, ImageDraw.Draw(img)

    def jitter(seed):
        r = np.random.default_rng(seed)
        return int(r.integers(-12, 12)), int(r.integers(-12, 12))

    s = size / 256

    def grave(d, dx, dy, target):
        if target:  # rounded headstone with a cross
            d.rectangle([90 * s + dx, 100 * s + dy, 166 * s + dx, 210 * s + dy], fill=(110, 110, 115))
            d.ellipse([90 * s + dx, 70 * s + dy, 166 * s + dx, 130 * s + dy], fill=(110, 110, 115))
            d.line([128 * s + dx, 95 * s + dy, 128 * s + dx, 160 * s + dy], fill=(40, 40, 40), width=int(6 * s))
            d.line([108 * s + dx, 115 * s + dy, 148 * s + dx, 115 * s + dy], fill=(40, 40, 40), width=int(6 * s))
        else:  # stone bench of similar mass
            d.rectangle([60 * s + dx, 140 * s + dy, 196 * s + dx, 165 * s + dy], fill=(110, 110, 115))
            d.rectangle([75 * s + dx, 165 * s + dy, 95 * s + dx, 210 * s + dy], fill=(110, 110, 115))
            d.rectangle([161 * s + dx, 165 * s + dy, 181 * s + dx, 210 * s + dy], fill=(110, 110, 115))
        d.rectangle([0, 210 * s + dy, size, size], fill=(90, 140, 80))

    def coffin(d, dx, dy, target):
        if target:  # tapered hexagon
            pts = [(128, 40), (165, 70), (155, 215), (101, 215), (91, 70)]
        else:  # rectangular chest
            pts = [(91, 60), (165, 60), (165, 215), (91, 215)]
        d.polygon([(x * s + dx, y * s + dy) for x, y in pts], fill=(120, 80, 45), outline=(60, 35, 20))

    def hourglass(d, dx, dy, target):
        frame = [(80, 40), (176, 40), (136, 128), (176, 216), (80, 216), (120, 128)]
        d.polygon([(x * s + dx, y * s + dy) for x, y in frame], outline=(60, 60, 60), width=int(4 * s))
        top_level, bottom_level = (120, 150) if target else (55, 205)  # little sand left vs full top
        d.polygon([(x * s + dx, y * s + dy) for x, y in
                   [(80 + (top_level - 40) * 40 / 88, top_level), (176 - (top_level - 40) * 40 / 88, top_level),
                    (136, 128), (120, 128)]], fill=(210, 170, 90))
        d.polygon([(x * s + dx, y * s + dy) for x, y in
                   [(120 + (bottom_level - 128) * 40 / 88, bottom_level), (136 - (bottom_level - 128) * 40 / 88 + 0, bottom_level),
                    (176, 216), (80, 216)]], fill=(210, 170, 90))

    def candle(d, dx, dy, target):
        top = 185 if target else 60
        d.rectangle([110 * s + dx, top * s + dy, 146 * s + dx, 215 * s + dy], fill=(240, 235, 220), outline=(120, 120, 110))
        d.ellipse([120 * s + dx, (top - 30) * s + dy, 136 * s + dx, (top - 4) * s + dy], fill=(250, 190, 60))

    def calendar(d, dx, dy, target):
        d.rectangle([50 * s + dx, 50 * s + dy, 206 * s + dx, 206 * s + dy], fill=(250, 250, 245), outline=(60, 60, 60))
        for i, j in itertools.product(range(5), range(5)):
            x0, y0 = 58 + i * 30, 60 + j * 28
            d.rectangle([x0 * s + dx, y0 * s + dy, (x0 + 24) * s + dx, (y0 + 22) * s + dy], outline=(150, 150, 150))
            if target and (i, j) != (4, 4):
                d.line([x0 * s + dx, y0 * s + dy, (x0 + 24) * s + dx, (y0 + 22) * s + dy], fill=(200, 30, 30), width=int(3 * s))

    def apple(d, dx, dy, target):
        d.ellipse([78 * s + dx, 78 * s + dy, 178 * s + dx, 178 * s + dy],
                  fill=(120, 85, 40) if target else (200, 30, 40))
        if target:  # rot spots
            for x, y in ((100, 110), (140, 125), (118, 150)):
                d.ellipse([x * s + dx, y * s + dy, (x + 18) * s + dx, (y + 14) * s + dy], fill=(60, 45, 25))

    def bread(d, dx, dy, target):
        d.rounded_rectangle([70 * s + dx, 70 * s + dy, 186 * s + dx, 190 * s + dy], radius=int(20 * s),
                            fill=(225, 190, 130), outline=(150, 110, 60))
        if target:  # mould
            for x, y in ((90, 95), (130, 140), (150, 90)):
                d.ellipse([x * s + dx, y * s + dy, (x + 26) * s + dx, (y + 20) * s + dy], fill=(110, 140, 100))

    families = {"MS": [("grave", grave), ("coffin", coffin)],
                "LT": [("hourglass", hourglass), ("candle", candle), ("calendar", calendar)],
                "NEG": [("apple", apple), ("bread", bread)]}
    rows = []
    for cond, fams in families.items():
        for family, draw in fams:
            for seed in seeds:
                dx, dy = jitter(int(rng_master.integers(1 << 30)) + seed)
                for role in ("target", "control"):
                    img, d = canvas()
                    draw(d, dx, dy, role == "target")
                    path = out_dir / cond / f"{family}_s{seed}_{role}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(path)
                    rows.append(dict(cond=cond, family=family, seed=seed, role=role,
                                     pair_id=f"{cond}_{family}_s{seed}", path=str(path), prompt="procedural"))
    return pd.DataFrame(rows)


def scan_image_folder(root: str | Path) -> pd.DataFrame:
    """Manifest for your own images, laid out as ``root/<COND>/<family>_s<k>_<target|control>.<ext>``."""
    rows = []
    for path in sorted(Path(root).glob("*/*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        stem = path.stem
        try:
            pair, role = stem.rsplit("_", 1)
            family, seed = pair.rsplit("_s", 1)
        except ValueError:
            LOGGER.warning("Skipping %s: expected <family>_s<k>_<target|control>", path)
            continue
        if role not in ("target", "control"):
            continue
        rows.append(dict(cond=path.parent.name, family=family, seed=int(seed), role=role,
                         pair_id=f"{path.parent.name}_{family}_s{seed}", path=str(path), prompt=""))
    df = pd.DataFrame(rows)
    complete = df.groupby("pair_id").role.nunique() == 2
    if (~complete).any():
        LOGGER.warning("Dropping incomplete pairs: %s", list(complete[~complete].index))
        df = df[df.pair_id.isin(complete[complete].index)]
    return df.reset_index(drop=True)


def lowlevel_stats(manifest: pd.DataFrame) -> pd.DataFrame:
    """Luminance, RMS contrast, edge density and colourfulness per image.

    The meaning of a pair should be the only systematic difference between its
    members; ``paired_lowlevel_tests`` flags properties that differ anyway.
    """
    from PIL import Image

    rows = []
    for r in manifest.itertuples():
        rgb = np.asarray(Image.open(r.path).convert("RGB").resize((256, 256)), dtype=float) / 255
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
        gy, gx = np.gradient(lum)
        rg, yb = rgb[..., 0] - rgb[..., 1], 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
        rows.append(dict(path=r.path, luminance=lum.mean(), contrast=lum.std(),
                         edges=np.hypot(gx, gy).mean(),
                         colourfulness=np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())))
    return manifest.merge(pd.DataFrame(rows), on="path")


def paired_lowlevel_tests(stats: pd.DataFrame) -> pd.DataFrame:
    """Paired t-tests of target - control for each low-level property, per condition."""
    from scipy import stats as st

    out = []
    for cond, g in stats.groupby("cond"):
        wide = g.pivot(index="pair_id", columns="role")
        for prop in ("luminance", "contrast", "edges", "colourfulness"):
            d = (wide[(prop, "target")] - wide[(prop, "control")]).to_numpy()
            t, p = st.ttest_1samp(d, 0.0) if len(d) > 1 else (np.nan, np.nan)
            out.append(dict(cond=cond, property=prop, mean_diff=d.mean(), t=t, p=p, n_pairs=len(d)))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Videos and TRIBE predictions
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Presentation:
    lead: float = 6.0  # grey before the image (s)
    show: float = 3.0  # image on screen (s)
    tail: float = 9.0  # grey after the image (s)
    fps: int = 16

    @property
    def total(self) -> float:
        return self.lead + self.show + self.tail


def make_image_video(image_path: str | Path | None, out_path: str | Path, pres: Presentation = Presentation(),
                     size: int = 512) -> Path:
    """grey | image | grey, as an mp4. ``image_path=None`` gives the all-grey baseline."""
    from moviepy import ColorClip, ImageClip, concatenate_videoclips
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    grey = lambda d: ColorClip(size=(size, size), color=GREY, duration=d)  # noqa: E731
    if image_path is None:
        middle = grey(pres.show)
    else:
        rgb = np.asarray(Image.open(image_path).convert("RGB").resize((size, size)))
        middle = ImageClip(rgb, duration=pres.show)
    clip = concatenate_videoclips([grey(pres.lead), middle, grey(pres.tail)])
    clip.write_videofile(str(out_path), fps=pres.fps, codec="libx264", audio=False, logger=None)
    clip.close()
    return out_path


def predict_timelines(tribe, videos: dict[str, Path], verbose: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One TRIBE ``predict`` call for all videos, each on its own timeline.

    Returns ``{timeline: (times, preds[n_tr, n_vertices])}`` with ``times`` in
    stimulus time (see module docstring for the 5-s convention).
    """
    events = pd.DataFrame([dict(type="Video", filepath=str(p), start=0.0, timeline=name, subject="default")
                           for name, p in videos.items()])
    preds, segments = tribe.predict(events=events, verbose=verbose)
    tl = np.array([str(s.timeline) for s in segments])
    st = np.array([float(s.start) for s in segments])
    out = {}
    for name in videos:
        sel = np.flatnonzero(tl == name)
        order = sel[np.argsort(st[sel])]
        out[name] = (st[order], np.asarray(preds[order], dtype=np.float32))
    return out


def estimate_peak_lag(responses: dict[str, tuple[np.ndarray, np.ndarray]], grey_key: str, onset: float,
                      image_keys: tp.Sequence[str]) -> tuple[float, pd.DataFrame]:
    """Lag (s, relative to image onset, in ``predict()`` time) at which the
    image-evoked response is largest, averaged over all images.

    Uses RMS over vertices of (image - grey), pooled across conditions, so the
    choice of window cannot favour any contrast.
    """
    gt, gp = responses[grey_key]
    curves = []
    for k in image_keys:
        t, p = responses[k]
        n = min(len(t), len(gt))
        curves.append(np.sqrt(((p[:n] - gp[:n]) ** 2).mean(1)))
    n = min(len(c) for c in curves)
    curve = np.mean([c[:n] for c in curves], 0)
    times = gt[:n]
    lag = float(times[int(np.argmax(curve))] - onset)
    return lag, pd.DataFrame(dict(time=times, rel_onset=times - onset, rms_response=curve))


def image_maps(responses, grey_key: str, onset: float, lag: float, image_keys: tp.Sequence[str],
               half_width: float = 1.0) -> np.ndarray:
    """``[n_images, n_vertices]``: mean over the peak window of (image - grey)."""
    gt, gp = responses[grey_key]
    lo, hi = onset + lag - half_width, onset + lag + half_width
    out = []
    for k in image_keys:
        t, p = responses[k]
        sel = (t >= lo) & (t <= hi)
        gsel = (gt >= lo) & (gt <= hi)
        out.append(p[sel].mean(0) - gp[gsel].mean(0))
    return np.stack(out)


def suggested_pred_offset(peak_lag: float, show: float) -> float:
    """BOLD for a ``show``-s stimulus peaks ~5 s + show/2 after onset. If
    TRIBE's response peaks ``peak_lag`` s after onset in ``predict()`` time,
    ``predict()`` row t corresponds to BOLD time t + (5 + show/2 - peak_lag).
    Use the rounded value as ``analysis.pred_offset`` in configs/default.yaml.
    """
    return 5.0 + show / 2 - peak_lag


# --------------------------------------------------------------------------
# Comparing MS and LT
# --------------------------------------------------------------------------


def pair_differences(maps: np.ndarray, manifest: pd.DataFrame) -> dict[str, tuple[list[str], np.ndarray]]:
    """``{cond: (pair_ids, target - control maps [n_pairs, V])}``."""
    idx = {(r.pair_id, r.role): i for i, r in enumerate(manifest.itertuples())}
    out = {}
    for cond, g in manifest.groupby("cond"):
        pairs = sorted(g.pair_id.unique())
        out[cond] = (pairs, np.stack([maps[idx[(p, "target")]] - maps[idx[(p, "control")]] for p in pairs]))
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def compare_contrasts(d_a: np.ndarray, d_b: np.ndarray, mask: np.ndarray | None = None,
                      n_perm: int = 5000, seed: int = 0) -> dict[str, float]:
    """How similar are two meaning contrasts (e.g. MS - control vs LT - control)?

    * ``r``: spatial correlation of the two mean contrast maps.
    * ``p_perm``: sign-flip null. Each pair difference is flipped at random
      (independently in both conditions), which keeps every map's spatial
      smoothness but destroys any consistent meaning effect.
    * ``rel_a`` / ``rel_b``: split-half reliability of each contrast
      (Spearman-Brown corrected; NaN if the halves do not agree at all).
      ``r`` cannot meaningfully exceed sqrt(rel_a * rel_b).
    * ``r_disatt``: r corrected for that ceiling; unstable when reliabilities are low.
    """
    # float64: TRIBE predicts float32, and the Gram route below must reproduce r exactly
    d_a, d_b = (np.asarray(d if mask is None else d[:, mask], dtype=np.float64) for d in (d_a, d_b))
    r = _corr(d_a.mean(0), d_b.mean(0))
    # Correlations of sign-flipped means only need the pair x pair Gram
    # matrices of the vertex-centred differences (exact, and tiny in memory).
    ca, cb = d_a - d_a.mean(1, keepdims=True), d_b - d_b.mean(1, keepdims=True)
    g_ab, g_aa, g_bb = ca @ cb.T, ca @ ca.T, cb @ cb.T
    rng = np.random.default_rng(seed)
    sa = rng.choice([-1.0, 1.0], size=(n_perm, len(d_a)))
    sb = rng.choice([-1.0, 1.0], size=(n_perm, len(d_b)))
    sa[0] = sb[0] = 1.0  # the observed pattern counts, so p >= 1 / n_perm (as in analysis.sign_flip_test)
    num = np.einsum("ki,ij,kj->k", sa, g_ab, sb)
    den = np.sqrt(np.einsum("ki,ij,kj->k", sa, g_aa, sa) * np.einsum("ki,ij,kj->k", sb, g_bb, sb))
    null = num / (den + 1e-12)

    def split_half(d):
        if len(d) < 4:
            return np.nan
        h1, h2 = d[0::2].mean(0), d[1::2].mean(0)
        x = _corr(h1, h2)
        return 2 * x / (1 + x) if x > 0 else np.nan

    rel_a, rel_b = split_half(d_a), split_half(d_b)
    ceiling = np.sqrt(rel_a * rel_b) if rel_a == rel_a and rel_b == rel_b else np.nan
    return dict(r=r, p_perm=float((np.abs(null) >= abs(r) - 1e-12).mean()), rel_a=rel_a, rel_b=rel_b,
                r_disatt=r / ceiling if ceiling == ceiling and ceiling > 0 else np.nan,
                n_pairs_a=len(d_a), n_pairs_b=len(d_b))


def contrast_reliability_null(d: np.ndarray, mask: np.ndarray | None = None, n_perm: int = 2000,
                              seed: int = 0) -> dict[str, float]:
    """Is a single contrast (target - control) consistent across pairs at all?

    Sign-flip test on the norm of the mean map: under the null the mean map is
    as large as a randomly signed average of the same pair differences.
    """
    # float64, or rounding drops the observed pattern from its own null (p = 0 with 4 pairs)
    d = np.asarray(d if mask is None else d[:, mask], dtype=np.float64)
    obs = np.linalg.norm(d.mean(0))
    if len(d) <= 12:  # exact: every sign pattern; all-flipped equals the observed, so p >= 2 / 2**n
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=len(d))))
    else:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_perm, len(d)))
        signs[0] = 1.0
    gram = d @ d.T  # |s.d / n|^2 = s G s / n^2: no [n_patterns x n_vertices] matrix needed
    null = np.sqrt(np.maximum(np.einsum("ki,ij,kj->k", signs, gram, signs), 0)) / len(d)
    return dict(norm=float(obs), p_perm=float((null >= obs * (1 - 1e-9)).mean()))


def top_fraction_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.05) -> float:
    """Dice overlap of the top ``frac`` most positive vertices of two maps."""
    k = max(1, int(frac * len(a)))
    ta, tb = set(np.argsort(a)[-k:]), set(np.argsort(b)[-k:])
    return 2 * len(ta & tb) / (2 * k)


EARLY_VISUAL_HCP = ["V1", "V2", "V3", "V4", "V3A", "V3B", "V6", "V6A", "V7", "V8", "LO1", "LO2", "LO3",
                    "V3CD", "V4t", "VMV1", "VMV2", "VMV3", "PIT", "MT", "MST", "FST"]


def cortex_mask_without(rois: tp.Sequence[str], n_vertices: int = 20484) -> np.ndarray:
    """Boolean mask that excludes the given HCP-MMP areas (both hemispheres)."""
    mask = np.ones(n_vertices, bool)
    if not rois:
        return mask
    from tribev2.utils import get_hcp_roi_indices

    for roi in rois:
        try:
            mask[get_hcp_roi_indices(roi, hemi="both", mesh="fsaverage5")] = False
        except ValueError:
            LOGGER.warning("HCP area %s not found; not excluded", roi)
    return mask


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def contact_sheet(manifest: pd.DataFrame, out_path: str | Path, thumb: int = 128) -> Path:
    """Targets on the top row of each pair, controls below: check the critical detail."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    pairs = list(dict.fromkeys(manifest.pair_id))
    fig, axes = plt.subplots(2, len(pairs), figsize=(1.3 * len(pairs), 3), squeeze=False)
    for j, pid in enumerate(pairs):
        for i, role in enumerate(("target", "control")):
            ax = axes[i, j]
            ax.imshow(Image.open(manifest[(manifest.pair_id == pid) & (manifest.role == role)].path.iloc[0])
                      .convert("RGB").resize((thumb, thumb)))
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(pid.replace("_", "\n", 1), fontsize=6)
        axes[0, 0].set_ylabel("target", fontsize=7)
        axes[1, 0].set_ylabel("control", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return Path(out_path)
