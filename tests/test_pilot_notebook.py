"""Executes notebooks/pilot_image_mortality_steering.ipynb end to end in
DRY_RUN mode (fake TRIBE, tiny random Llama, procedural images)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "pilot_image_mortality_steering.ipynb"

# Smaller settings so the rehearsal takes seconds, not minutes.
FAST = dict(SEEDS=(0,), PRESENTATION=dict(lead=3.0, show=2.0, tail=5.0, fps=8), INVERSION_STEPS=25,
            N_NULL_TARGETS=1, EPS_LIST=[-0.1, 0.1], MAX_NEW_TOKENS=6)


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
    assert set(ns["VECTORS"]) == {"u_MS", "u_LT", "u_null", "u_rand"}
    assert set(ns["summary"].vector) == {"u_MS", "u_LT", "u_null", "u_rand"}
    assert set(ns["summary"].axis) == {"ingroup_norm", "outgroup_exclusion"}
    assert len(ns["chat"]) == 1 + 2 * 3  # unsteered + 3 vectors x (+eps, -eps)
    assert {"r_MS", "r_LT"} <= set(ns["closed"].columns)
    for name in ("manifest.csv", "image_maps.npy", "contrasts.npz", "vectors.npz", "defense_summary.csv",
                 "chat_demo.csv"):
        assert (tmp_path / name).exists(), name
