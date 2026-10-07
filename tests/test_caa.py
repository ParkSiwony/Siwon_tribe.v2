"""Contrast-pair ingestion and CAA vectors (mirrors the steering study)."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from mortality_tribe import caa


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_load_study_schema_and_sentence_pairs(tmp_path):
    _write_jsonl(tmp_path / "mort_self_pairs.jsonl", [
        dict(id="a1", family="f1", gen="GPT", context="c", question="Which fits?", opt_pos="I will die.",
             opt_neg="I will rest."),
        dict(id="a2", family="f2", gen="kimi", question="Which fits?", opt_pos="p2", opt_neg="n2")])
    (tmp_path / "arb_pairs.yaml").write_text(yaml.safe_dump({"pairs": [dict(pos="big house", neg="small house")]}))
    (tmp_path / "notes.md").write_text("ignored")
    df = caa.load_pairs(tmp_path)
    assert list(df.set) == ["arb", "mort_self", "mort_self"]
    assert list(df["mode"]) == ["sentence", "mc", "mc"]
    assert df.gen.tolist()[1:] == ["gpt", "kimi"]  # lower-cased
    assert df.loc[df.id == "a1", "pos"].item() == "I will die."
    with pytest.raises(KeyError):
        _write_jsonl(tmp_path / "bad_pairs.jsonl", [dict(question="q", opt_pos="only one side")])
        caa.load_pairs(tmp_path / "bad_pairs.jsonl")


def test_validate_flags_problems():
    df = pd.DataFrame([
        dict(id="1", set="mort_struct", family="f", gen="gpt", axis="", person="", context="", question="q",
             pos="I lose my house forever", neg="The house stays", mode="mc"),
        dict(id="2", set="mort_struct", family="f", gen="kimi", axis="", person="", context="", question="q",
             pos="The house stays", neg="The house stays", mode="mc"),
        dict(id="3", set="self_nonfinal", family="g", gen="gpt", axis="", person="", context="", question="q",
             pos="This is the last time I see it", neg="I will see it again", mode="mc"),
    ])
    out = caa.validate_pairs(df)
    kinds = set(out["issues"].issue)
    assert {"first person in options of impersonal set", "identical options",
            "finality word in a non-finality set", "duplicate neg"} <= kinds
    assert out["counts"].loc["mort_struct", "pairs"] == 2
    assert {"pos_words", "neg_words", "diff"} <= set(out["lengths"].columns)


def test_mc_prompt_matches_study_format():
    p = caa.mc_prompt("ctx", "Which fits?", "yes", "no")
    assert p == "ctx\nWhich fits?\n(A) yes\n(B) no\nAnswer: ("
    assert caa.mc_prompt("", "Q?", "a", "b").startswith("Q?\n(A) a")


def test_orthogonalize_removes_controls():
    rng = np.random.default_rng(0)
    b1, b2, v = rng.normal(size=(3, 50))
    clean, rho = caa.orthogonalize(v, b1, b2)
    assert abs(clean @ caa.unit(b1)) < 1e-10 and abs(clean @ caa.unit(b2)) < 1e-10
    assert 0 < rho < 1 and np.linalg.norm(clean) == pytest.approx(1.0)
    assert caa.orthogonalize(v)[1] == pytest.approx(1.0)


def test_quality_detects_content_and_generator_style():
    rng = np.random.default_rng(1)
    n, dim = 80, 64
    fams = np.repeat([f"f{i}" for i in range(8)], 10)
    gens = np.tile(["gpt", "kimi"], n // 2)
    content = caa.unit(rng.normal(size=dim))
    good = 2 * content + rng.normal(0, 0.5, (n, dim))
    q = caa.vector_quality(good, fams, gens)
    assert q["dz_holdout"] > 1.0 and q["split_half"] > 0.8 and q["gen_cos"] > 0.8
    # each generator writes its own direction: the vector is mostly style
    style = {g: caa.unit(rng.normal(size=dim)) for g in ("gpt", "kimi")}
    styled = np.stack([2 * style[g] for g in gens]) + rng.normal(0, 0.5, (n, dim))
    assert caa.vector_quality(styled, fams, gens)["gen_cos"] < 0.6
    # without family labels it falls back to pair-level splits
    assert np.isfinite(caa.vector_quality(good, ["_none"] * n)["split_half"])


def test_family_holdout_holds_out_whole_families():
    fams = np.repeat(["a", "b", "c", "d", "e"], 4)
    ho = caa.family_holdout(fams, frac=0.2)
    held = set(fams[ho])
    assert held and all(ho[fams == f].all() for f in held) and not ho.all()


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mortality_tribe import pilot_fakes as pf

    corpus = "Which statement fits better ? Nothing I do now can bring this moment back . Situation"
    return pf.tiny_llama(__import__("tempfile").mkdtemp(), corpus)


def test_item_differences_and_extract_vectors(tiny, tmp_path):
    from mortality_tribe import pilot_fakes as pf

    llm, tok = tiny
    pairs = caa.load_pairs(pf.write_dry_pairs(tmp_path / "caa"))
    sub = pairs[pairs.set == "mort_self"]
    d = caa.item_differences(llm, tok, sub, layer=3, verbose=False)
    assert d.shape == (len(sub), llm.config.hidden_size) and np.isfinite(d).all()
    # both letter orders are averaged: a pair whose options are identical gives exactly zero
    same = sub.head(1).assign(neg=sub.head(1).pos)
    assert np.allclose(caa.item_differences(llm, tok, same, layer=3, verbose=False), 0, atol=1e-5)
    vecs, quality = caa.extract_vectors(llm, tok, pairs, 3, ["mort_self", "mort_struct"],
                                        ["neg_sensory", "neg_affect"])
    assert {"v_mort_self", "v_mort_self_clean", "v_mort_struct_clean", "v_neg_sensory"} <= set(vecs)
    assert abs(vecs["v_mort_self_clean"] @ vecs["v_neg_sensory"]) < 1e-6
    assert {"dz_holdout", "split_half", "gen_cos", "rho", "passes"} <= set(quality.columns)
    sentence = pairs.head(2).assign(question="", mode="sentence")
    assert caa.item_differences(llm, tok, sentence, layer=3, verbose=False).shape == (2, llm.config.hidden_size)
