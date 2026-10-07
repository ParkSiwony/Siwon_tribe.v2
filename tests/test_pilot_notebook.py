"""Executes notebooks/pilot_image_mortality_steering.ipynb end to end in
DRY_RUN mode (fake TRIBE, tiny random Llama, procedural images)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "pilot_image_mortality_steering.ipynb"

# Smaller settings so the rehearsal takes seconds, not minutes.
FAST = dict(SEEDS=(0,), PRESENTATION=dict(lead=3.0, show=2.0, tail=5.0, fps=8), INVERSION_STEPS=20,
            N_NULL_TARGETS=1, EPS_LIST=[-0.1, 0.1], MAX_NEW_TOKENS=6, TEXT_PAIRS_PER_SET=2, CAA_MAX_PAIRS=6,
            ITEMS_MAX=6)


def test_notebook_runs_in_dry_run_mode(monkeypatch, tmp_path):
    for mod in ("torch", "transformers", "moviepy", "tokenizers"):
        pytest.importorskip(mod)
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.setenv("PILOT_DRY_RUN", "1")
    monkeypatch.chdir(ROOT)
    cells = [c for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(cells):
        exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}[{i}]", "exec"), ns)
        if i == 0:  # configuration cell
            assert ns["DRY_RUN"] is True
            ns.update(FAST, QUESTIONS=ns["QUESTIONS"][:1], OUT=tmp_path, IMAGE_DIR=tmp_path / "images")
    # A: the gate ran on real sentence pairs and image pairs and made a decision
    assert ns["recommended"] in {"text", "image", "both", "caa_only"}
    assert {"finality|text", "negative|text", "finality|image", "negative|image"} <= set(ns["contrasts"])
    # C: brain-route and sentence-pair vectors in one space
    assert {"u_text_self", "u_MS", "u_LT", "u_null", "u_rand", "v_mort_self", "v_mort_self_clean",
            "v_def_norm", "v_def_excl"} <= set(ns["VECTORS"])
    assert {"r_text_self", "r_MS", "r_LT"} <= set(ns["closed"].columns)
    # D: steering on the user's items and chat
    assert set(ns["summary"].vector) == set(ns["STEER_VECTORS"])
    assert set(ns["summary"].axis) == {"ingroup_norm", "outgroup_exclusion"}
    assert len(ns["chat"]) == len(ns["QUESTIONS"]) * (1 + 2 * len(ns["CHAT_VECTORS"]))
    for name in ("manifest.csv", "image_maps.npy", "text_maps.npy", "text_index.csv", "contrasts.npz",
                 "vectors.npz", "caa_quality.csv", "defense_summary.csv", "chat_demo.csv"):
        assert (tmp_path / name).exists(), name
