"""Figures: cortical contrast maps, ROI effect bars, and onset-locked time courses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_maps(map_dir: str | Path, out_dir: str | Path, pattern: str = "*.npy") -> list[Path]:
    """Render every saved contrast map on the inflated fsaverage5 surface."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tribev2.plotting import PlotBrainNilearn

    plotter = PlotBrainNilearn(mesh="fsaverage5")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(Path(map_dir).glob(pattern)):
        vec = np.load(path)
        if vec.shape[0] != 20484:
            continue  # fake / non-fsaverage5 data
        vmax = np.percentile(np.abs(vec), 99) or 1.0
        plotter.plot_surf(  # creates its own figure and returns the colour mappable
            vec, views=["left", "right", "medial_left", "medial_right"], cmap="cold_hot",
            vmin=-vmax, vmax=vmax, symmetric_cbar=True, colorbar=True,
        )
        fig = plt.gcf()
        fig.suptitle(path.stem)
        out = out_dir / f"{path.stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def plot_roi_effects(table: pd.DataFrame, out_path: str | Path, title: str, est: str = "mean") -> Path:
    """Grouped bars of ROI effects; stars mark p_fwe < .05."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_cols = [c for c in ("contrast", "delay", "category") if c in table]
    table = table.assign(group=table[group_cols].astype(str).agg(" | ".join, axis=1))
    groups, rois = table.group.unique(), table.roi.unique()
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(rois) * max(1, len(groups) / 3)), 4))
    width = 0.8 / len(groups)
    for i, g in enumerate(groups):
        sub = table[table.group == g].set_index("roi").loc[rois]
        x = np.arange(len(rois)) + i * width
        ax.bar(x, sub[est], width, label=g)
        for xi, (v, p) in zip(x, zip(sub[est], sub.p_fwe)):
            if p < 0.05:
                ax.text(xi, v, "*", ha="center", va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(np.arange(len(rois)) + 0.4 - width / 2, rois, rotation=45, ha="right")
    ax.set_ylabel("Δ predicted BOLD (SD)")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_timecourses(tc: pd.DataFrame, out_path: str | Path, lag: float) -> Path:
    """Probe-locked ROI time courses per prime condition (check the lag choice)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rois = tc.roi.unique()
    fig, axes = plt.subplots(1, len(rois), figsize=(2.2 * len(rois), 2.4), sharex=True)
    for ax, roi in zip(np.atleast_1d(axes), rois):
        sub = tc[tc.roi == roi].groupby(["prime_cond", "lag"]).value.mean().unstack(0)
        sub.plot(ax=ax, legend=False, lw=1)
        ax.axvspan(lag, lag + 4, color="0.9", zorder=0)
        ax.set_title(roi, fontsize=8)
        ax.set_xlabel("s from probe onset", fontsize=7)
    np.atleast_1d(axes)[-1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)
