"""Command line entry point.

    python -m mortality_tribe design   --config configs/default.yaml
    python -m mortality_tribe synth    --config configs/default.yaml
    python -m mortality_tribe predict  --config configs/default.yaml
    python -m mortality_tribe analyze  --config configs/default.yaml
    python -m mortality_tribe all      --config configs/default.yaml

``--dry-run`` swaps TTS for silence and TRIBE for a synthetic predictor with
known planted effects, so the whole pipeline can be checked on a laptop.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import pandas as pd
import yaml

from . import analysis, audio, predict, rois, stimuli

LOGGER = logging.getLogger("mortality_tribe")


def load_config(path: str | Path, dry_run: bool = False) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if dry_run:
        cfg["output_dir"] = str(Path(cfg["output_dir"]) / "dry_run")
        cfg["tts"] = "silent"
    return cfg


def _paths(cfg: dict) -> dict[str, Path]:
    out = Path(cfg["output_dir"])
    return dict(design=out / "design.yaml", audio=out / "audio", timings=out / "timings.csv",
                tts_cache=out / "tts_cache", preds=out / "preds", results=out / "results")


def cmd_design(cfg: dict, args) -> None:
    stim = stimuli.load_stimuli(cfg["stimuli_dir"])
    d = cfg["design"]
    trials = stimuli.build_design(
        stim, stimuli.Timing(**cfg.get("timing", {})), delays=d["delays"],
        crossing=d["crossing"], n_probe_sets=d.get("n_probe_sets"), seed=d["seed"],
    )
    stimuli.save_design(trials, _paths(cfg)["design"])
    LOGGER.info("Wrote %d trials to %s", len(trials), _paths(cfg)["design"])


def cmd_synth(cfg: dict, args) -> None:
    p = _paths(cfg)
    trials = stimuli.load_design(p["design"])
    cache = audio.SegmentCache(p["tts_cache"], audio.get_backend(cfg["tts"]))
    tables = []
    for t in trials:
        _, tab = audio.render_trial(t, cache, p["audio"])
        tables.append(tab)
    timings = pd.concat(tables, ignore_index=True)
    timings.to_csv(p["timings"], index=False)
    total = timings.groupby("trial_id").apply(lambda g: g.onset.max() + g.duration.iloc[-1]).sum()
    LOGGER.info("Rendered %d trials (%.1f min of audio)", len(trials), total / 60)


def _predictor(cfg: dict, args):
    if args.dry_run:
        n = cfg.get("dry_run_vertices", 400)
        return predict.FakePredictor(n, rois.fake_roi_indices(n))
    m = cfg["model"]
    return predict.TribePredictor(m["checkpoint"], m["cache_folder"], m["device"])


def cmd_predict(cfg: dict, args) -> None:
    p = _paths(cfg)
    trials = stimuli.load_design(p["design"])
    timings = pd.read_csv(p["timings"])
    model = _predictor(cfg, args)
    for i, t in enumerate(trials):
        out = p["preds"] / f"{t.trial_id}.npz"
        if out.exists() and not args.overwrite:
            continue
        LOGGER.info("[%d/%d] %s", i + 1, len(trials), t.trial_id)
        times, preds = model.predict(p["audio"] / f"{t.trial_id}.wav", t, timings[timings.trial_id == t.trial_id])
        predict.save_prediction(out, times, preds)


def cmd_analyze(cfg: dict, args) -> None:
    p = _paths(cfg)
    trials = stimuli.load_design(p["design"])
    timings = pd.read_csv(p["timings"])
    a = cfg["analysis"]
    acfg = analysis.AnalysisConfig(
        method=a["method"], lag=a["lag"], extra=a["extra"], pairs=[tuple(x) for x in a["pairs"]],
        control_category=a["control_category"], n_perm=a["n_perm"], seed=a["seed"],
        hypotheses=a.get("hypotheses", []),
    )
    n_vertices = predict.load_prediction(p["preds"] / f"{trials[0].trial_id}.npz")[1].shape[1]
    roi_idx = rois.hcp_roi_indices() if n_vertices == 20484 else rois.fake_roi_indices(n_vertices)
    meta, resp = analysis.load_responses(trials, timings, p["preds"], acfg)
    results = analysis.run_all(meta, resp, roi_idx, acfg, p["results"])
    tc = analysis.evoked_timecourses(trials, timings, p["preds"], roi_idx)
    tc.to_csv(p["results"] / "probe_timecourses.csv", index=False)
    (p["results"] / "analysis_config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(acfg)))
    if not args.no_plots:
        try:
            from . import plotting

            fig_dir = p["results"] / "figures"
            fig_dir.mkdir(exist_ok=True)
            plotting.plot_roi_effects(results["prime_contrasts"], fig_dir / "prime_contrasts.png", "Prime-evoked")
            for (c, d), g in results["probe_modulation"].groupby(["contrast", "delay"]):
                plotting.plot_roi_effects(g, fig_dir / f"probe_{c}_{d}.png", f"Probe modulation {c} ({d})")
            plotting.plot_timecourses(tc, fig_dir / "probe_timecourses.png", acfg.lag)
            plotting.plot_maps(p["results"] / "maps", fig_dir / "maps")
        except ImportError as e:
            LOGGER.warning("Skipping figures (%s). Install the [plot] extra.", e)
    LOGGER.info("Results in %s\n%s", p["results"], (p["results"] / "summary.md").read_text())


COMMANDS = dict(design=cmd_design, synth=cmd_synth, predict=cmd_predict, analyze=cmd_analyze)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="mortality_tribe", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="offline run with synthetic data")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, args.dry_run)
    for name in (list(COMMANDS) if args.command == "all" else [args.command]):
        COMMANDS[name](cfg, args)


if __name__ == "__main__":
    main()
