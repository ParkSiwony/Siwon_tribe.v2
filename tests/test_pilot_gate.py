"""Text -> brain predictions and the text-vs-image modality gate."""

import numpy as np
import pandas as pd
import pytest

from mortality_tribe import pilot_gate as pg
from mortality_tribe import pilot_text as pt


def _pairs(n=3, set_name="mort_self"):
    return pd.DataFrame([dict(id=f"p{i}", set=set_name, family=f"f{i % 2}", gen=("gpt", "kimi")[i % 2],
                              context=f"The session number {i} is closing now",
                              pos="Nothing can bring it back", neg="It can come back later", mode="mc")
                         for i in range(n)])


def test_pair_text_events_windows():
    events, index = pt.pair_text_events(_pairs(2), use_neuralset=False)
    assert len(index) == 4 and set(index.role) == {"pos", "neg"}
    for r in index.itertuples():
        words = events[events.timeline == r.timeline]
        option = words[words.start >= r.opt_on]
        assert option.text.iloc[-1].endswith(".")  # a full stop was added for sentence parsing
        assert words.start.min() < r.opt_on < r.opt_off
        assert words.context.iloc[-1].startswith("The session")  # left context includes the context sentence


def test_text_responses_through_fake_tribe(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mortality_tribe import pilot_fakes as pf

    llm, tok = pf.tiny_llama(tmp_path / "tiny", "The session number is closing now Nothing can bring it back later")
    tribe = pf.FakeTribe(llm, tok, tmp_path / "tiny", n_vertices=300)
    events, index = pt.pair_text_events(_pairs(3), use_neuralset=False)
    maps = pt.text_responses(tribe, events, index)
    assert maps.shape == (6, 300) and np.isfinite(maps).all()
    diffs = pt.text_pair_differences(index, maps)
    meta, d = diffs["mort_self"]
    assert d.shape == (3, 300) and list(meta.pair_id) == ["p0", "p1", "p2"]
    assert not np.allclose(d, 0)  # different options -> different predicted responses
    assert set(pt.family_maps(meta, d)) == {"f0", "f1"}


def _contrasts(shared=1.0, image_consistent=True, seed=0):
    rng = np.random.default_rng(seed)
    v = 500
    fin, neg = rng.normal(size=v), rng.normal(size=v)
    fin_img = shared * fin + (1 - shared) * rng.normal(size=v)
    img_effect = fin_img if image_consistent else np.zeros(v)
    return {"finality|text": fin + rng.normal(0, 1, (16, v)),
            "finality|image": img_effect + rng.normal(0, 1, (12, v)),
            "negative|text": neg + rng.normal(0, 1, (16, v)),
            "negative|image": neg + rng.normal(0, 1, (12, v))}


def _decide(c):
    _, info = pg.mtmm(c)
    conv = pg.convergence(c, pg.mtmm_pairs(), n_perm=500)
    return pg.recommend_modality(info, conv)


def test_gate_recommends_text_when_modalities_converge():
    choice, reasons = _decide(_contrasts(shared=1.0))
    assert choice == "text" and any("convergent" in r for r in reasons)


def test_gate_keeps_both_when_modalities_disagree():
    assert _decide(_contrasts(shared=0.0))[0] == "both"


def test_gate_falls_back_to_the_consistent_modality():
    assert _decide(_contrasts(image_consistent=False))[0] == "text"
    rng = np.random.default_rng(1)
    noise = {k: rng.normal(size=(12, 300)) for k in ("finality|text", "finality|image", "negative|text")}
    assert _decide(noise)[0] == "caa_only"


def test_mtmm_shapes():
    corr, info = pg.mtmm(_contrasts())
    assert corr.shape == (4, 4) and np.allclose(np.diag(corr), 1)
    assert {"consistency_p", "split_half", "n_pairs"} <= set(info.columns)
    assert len(pg.mtmm_pairs()) == 6
