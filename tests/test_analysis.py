import numpy as np
import pandas as pd
import pytest

from mortality_tribe import analysis, audio, predict, rois, stimuli
from tests.test_design import STIM


def test_sign_flip_detects_effect_and_controls_null():
    rng = np.random.default_rng(0)
    null = rng.normal(size=(10, 5))
    effect = null.copy()
    effect[:, 0] += 3
    res = analysis.sign_flip_test(effect)
    assert res["p_fwe"][0] < 0.01
    assert (res["p_unc"][1:] > 0.01).all()
    assert analysis.sign_flip_test(null)["p_fwe"].min() > 0.05


def test_false_positive_rate_is_nominal():
    rng = np.random.default_rng(1)
    hits = sum(analysis.sign_flip_test(rng.normal(size=(8, 9)))["p_fwe"].min() < 0.05 for _ in range(200))
    assert hits / 200 < 0.1


def test_bh_and_holm_are_monotone():
    p = np.array([0.001, 0.04, 0.03, 0.5])
    assert np.all(analysis.bh_fdr(p) >= p)
    assert np.all(analysis.holm(p) >= p)
    assert analysis.holm(p)[0] == pytest.approx(0.004)


def test_glm_separates_overlapping_responses():
    times = np.arange(120.0)
    X = analysis.glm_design(times, [5, 40], [30, 3])
    y = X[:, :2] @ np.array([[2.0, -1.0], [0.0, 3.0]])
    beta = analysis.glm_betas(times, y, [5, 40], [30, 3])
    assert np.allclose(beta, [[2.0, -1.0], [0.0, 3.0]], atol=1e-6)


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dry")
    stim = stimuli.load_stimuli(STIM)
    trials = stimuli.build_design(stim)
    cache = audio.SegmentCache(tmp / "cache", audio.SilentBackend())
    timings = pd.concat([audio.render_trial(t, cache, tmp / "audio")[1] for t in trials], ignore_index=True)
    n = 200
    roi_idx = rois.fake_roi_indices(n)
    model = predict.FakePredictor(n, roi_idx, seed=0)
    for t in trials:
        times, preds = model.predict(None, t, timings[timings.trial_id == t.trial_id])
        predict.save_prediction(tmp / "preds" / f"{t.trial_id}.npz", times, preds)
    cfg = analysis.AnalysisConfig(
        pairs=[("MS", "PAIN"), ("LT", "ET")], n_perm=1000,
        hypotheses=[
            dict(name="planted", table="probe_interaction", contrast="MS-PAIN", delay="delayed",
                 category="WV-vs-NEUT", roi="vmPFC", direction="+"),
            dict(name="null", table="probe_interaction", contrast="LT-ET", delay="immediate",
                 category="SOC-vs-NEUT", roi="vmPFC", direction="+"),
        ],
    )
    meta, resp = analysis.load_responses(trials, timings, tmp / "preds", cfg)
    return analysis.run_all(meta, resp, roi_idx, cfg, tmp / "results"), tmp


def test_pipeline_recovers_planted_probe_effect(dry_run):
    results, _ = dry_run
    hyp = results["hypotheses"].set_index("name")
    assert hyp.loc["planted", "supported"]
    assert not hyp.loc["null", "supported"]
    inter = results["probe_interaction"]
    sig = inter[inter.p_fwe < 0.01]  # 1,000 permutations: too coarse for the global FDR here
    assert set(sig.roi) == {"vmPFC"}
    assert set(sig.category) == {"WV-vs-NEUT"}
    assert set(sig.contrast) == {"MS-PAIN"}


def test_pipeline_recovers_planted_prime_effect(dry_run):
    results, _ = dry_run
    pc = results["prime_contrasts"].set_index(["contrast", "roi"])
    # the salience boost is the largest MS-PAIN prime effect, in the right direction
    ms = pc.loc["MS-PAIN"]
    assert ms["mean"].idxmax() == "salience"
    assert ms.loc["salience", "mean"] > 0.5


def test_outputs_written(dry_run):
    _, tmp = dry_run
    out = tmp / "results"
    for name in ["prime_contrasts", "probe_modulation", "probe_interaction", "carryover", "decoding", "hypotheses"]:
        assert (out / f"{name}.csv").exists()
    assert "Pre-registered hypotheses" in (out / "summary.md").read_text()
    assert any((out / "maps").glob("*.npy"))
