"""Tests for the image -> brain -> Llama pilot.

TRIBE, V-JEPA and Llama-3.2 are not available here, so the GPU parts are
exercised with stand-ins: a toy differentiable "brain model" for the
inversion and a tiny random Llama with a word-level tokenizer for steering.
"""

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from mortality_tribe import pilot_bridge as pb
from mortality_tribe import pilot_images as pi


# --------------------------------------------------------------------------
# Layer bookkeeping
# --------------------------------------------------------------------------


def test_layer_groups_match_neuralset_convention():
    # TRIBE training default: layers_to_use [0.5, 0.75, 1.0], group_mean; Llama-3.2-3B has 29 hidden states
    groups = pb.hidden_state_groups([0.5, 0.75, 1.0], "group_mean", 29)
    assert groups == [(14, 21), (21, 29)]
    assert pb.injection_layer(groups) == 13  # layers[13] outputs hidden_states[14]
    assert pb.hidden_state_groups(0.5, "mean", 29) == [(14, 15)]
    assert len(pb.hidden_state_groups([0, 0.2, 0.4, 0.6, 0.8, 1.0], "group_mean", 29)) == 5


# --------------------------------------------------------------------------
# Stimuli and videos
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def procedural(tmp_path_factory):
    out = tmp_path_factory.mktemp("img")
    return pi.draw_pairs_procedural(out), out


def test_procedural_pairs_are_complete(procedural):
    manifest, _ = procedural
    assert set(manifest.cond) == {"MS", "LT"}
    roles = manifest.groupby("pair_id").role.apply(set)
    assert all(r == {"target", "control"} for r in roles)


def test_scan_folder_reproduces_manifest(procedural):
    manifest, root = procedural
    scanned = pi.scan_image_folder(root)
    assert set(scanned.pair_id) == set(manifest.pair_id)
    assert len(scanned) == len(manifest)


def test_lowlevel_stats(procedural):
    manifest, _ = procedural
    stats = pi.lowlevel_stats(manifest)
    assert np.isfinite(stats[["luminance", "contrast", "edges", "colourfulness"]].to_numpy()).all()
    tests = pi.paired_lowlevel_tests(stats)
    assert set(tests.cond) == {"MS", "LT"} and len(tests) == 8


def test_make_image_video(procedural, tmp_path):
    pytest.importorskip("moviepy")
    from moviepy import VideoFileClip

    manifest, _ = procedural
    pres = pi.Presentation(lead=1.0, show=1.0, tail=1.0, fps=8)
    path = pi.make_image_video(manifest.path.iloc[0], tmp_path / "x.mp4", pres, size=64)
    grey = pi.make_image_video(None, tmp_path / "g.mp4", pres, size=64)
    for p in (path, grey):
        with VideoFileClip(str(p)) as clip:
            assert clip.duration == pytest.approx(pres.total, abs=0.2)


# --------------------------------------------------------------------------
# Responses, lag, contrasts
# --------------------------------------------------------------------------


def _fake_responses(n_vertices=300, onset=6.0, peak=2.0, n_tr=18, seed=0):
    """Grey = noise; images add a pattern with a bump peaking at onset + peak."""
    rng = np.random.default_rng(seed)
    times = np.arange(n_tr, dtype=float)
    bump = np.exp(-0.5 * ((times - onset - peak) / 1.2) ** 2)
    base = rng.normal(0, 0.05, size=(n_tr, n_vertices))
    shared = rng.normal(size=n_vertices)  # "seeing an image"
    meaning = {"MS": rng.normal(size=n_vertices), "LT": rng.normal(size=n_vertices)}
    responses, keys, rows = {"grey": (times, base)}, [], []
    for cond in ("MS", "LT"):
        for k in range(6):
            for role in ("target", "control"):
                pat = shared + rng.normal(0, 0.5, n_vertices)
                if role == "target":
                    pat = pat + meaning[cond]
                key = f"{cond}_f{k}_s0_{role}"
                responses[key] = (times, base + bump[:, None] * pat[None] + rng.normal(0, 0.05, (n_tr, n_vertices)))
                keys.append(key)
                rows.append(dict(cond=cond, family=f"f{k}", seed=0, role=role, pair_id=f"{cond}_f{k}_s0", path=key))
    return responses, keys, pd.DataFrame(rows), meaning


def test_peak_lag_and_maps():
    responses, keys, manifest, meaning = _fake_responses(peak=2.0)
    lag, curve = pi.estimate_peak_lag(responses, "grey", 6.0, keys)
    assert lag == 2.0
    maps = pi.image_maps(responses, "grey", 6.0, lag, keys)
    d = pi.pair_differences(maps, manifest)
    assert pb.cosine(d["MS"][1].mean(0), meaning["MS"]) > 0.9
    assert pi.suggested_pred_offset(2.0, show=3.0) == pytest.approx(4.5)


def test_compare_contrasts_detects_shared_meaning():
    rng = np.random.default_rng(1)
    common = rng.normal(size=500)
    d_a = common + rng.normal(0, 1, (12, 500))
    d_b = common + rng.normal(0, 1, (12, 500))
    res = pi.compare_contrasts(d_a, d_b, n_perm=500)
    assert res["r"] > 0.5 and res["p_perm"] < 0.01
    assert res["rel_a"] > 0.5
    indep = pi.compare_contrasts(rng.normal(size=(12, 500)), rng.normal(size=(12, 500)), n_perm=500)
    assert indep["p_perm"] > 0.01
    mask = np.zeros(500, bool)
    mask[:250] = True
    assert pi.compare_contrasts(d_a, d_b, mask=mask, n_perm=100)["r"] > 0.5


def test_contrast_reliability_and_overlap():
    rng = np.random.default_rng(2)
    eff = rng.normal(size=400)
    assert pi.contrast_reliability_null(eff + rng.normal(0, 1, (10, 400)), n_perm=300)["p_perm"] < 0.01
    assert pi.contrast_reliability_null(rng.normal(size=(10, 400)), n_perm=300)["p_perm"] > 0.01
    assert pi.top_fraction_overlap(eff, eff) == 1.0


def test_sign_flip_targets_are_balanced():
    d = np.ones((8, 5))
    for t in pb.sign_flip_targets(d, 4):
        assert np.allclose(t, 0)


# --------------------------------------------------------------------------
# Inversion through a toy brain model
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _Batch:
    data: dict
    segments: list


def _toy_brain(dim=24, n_out=300, seed=0, nonlinear=True):
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.w1 = torch.nn.Parameter(torch.randn(dim, 64, generator=g) / dim ** 0.5)
            self.w2 = torch.nn.Parameter(torch.randn(64, n_out, generator=g) / 8)

        def forward(self, batch):
            x = batch.data["text"].float().mean(1).transpose(1, 2)  # B, T, D
            h = x @ self.w1
            h = torch.tanh(h) if nonlinear else h
            return (h @ self.w2).transpose(1, 2)  # B, O, T

    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(3, 2, dim, 20, generator=g)
    x[..., ::4] = 0  # some empty bins
    return Toy(), [_Batch(data={"text": x}, segments=[None] * 3)]


def test_inversion_recovers_planted_direction():
    torch = pytest.importorskip("torch")
    model, batches = _toy_brain()
    u_true = np.random.default_rng(3).normal(size=24)
    u_true /= np.linalg.norm(u_true)
    scale = 0.5 * pb.text_feature_norm(batches)
    target = pb.delta_for(model, batches, u_true, scale)
    res = pb.invert_brain_map(model, batches, target, n_steps=300, lr=0.05, seed=0, verbose=False)
    assert res.r > 0.99
    assert pb.cosine(res.u, u_true) > 0.95
    assert all(not p.requires_grad for p in model.parameters())
    with torch.no_grad():
        assert np.isfinite(res.delta).all()


def test_symmetry_separates_linear_steering():
    pytest.importorskip("torch")
    model, batches = _toy_brain(nonlinear=False)
    u = np.random.default_rng(4).normal(size=24)
    u /= np.linalg.norm(u)
    target = pb.delta_for(model, batches, u, 1.0)
    sym = pb.symmetry(model, batches, u, 1.0, target)
    assert sym["r_plus"] > 0.99 and sym["r_minus"] < -0.99
    assert sym["directional_over_symmetric"] > 1e3  # a linear map has no symmetric part


# --------------------------------------------------------------------------
# Steering a tiny random Llama
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_llm():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from mortality_tribe import pilot_steer as ps

    corpus = " ".join(f"{it.scenario} {it.defensive} {it.inclusive}" for it in ps.load_items())
    corpus += " Answer with A or B . ( ) : Which response is closer to what you think ? user assistant system"
    words = sorted(set(re.findall(r"\w+|[^\w\s]", corpus)))
    vocab = {w: i for i, w in enumerate(["<unk>", "<s>", "</s>", "<pad>", "<|user|>", "<|assistant|>"] + words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  unk_token="<unk>")
    tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|> {% endif %}")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                      bos_token_id=1, eos_token_id=2, pad_token_id=3)
    model = LlamaForCausalLM(cfg).eval()
    return model, tok


def test_letters_and_scoring(tiny_llm):
    from mortality_tribe import pilot_steer as ps

    model, tok = tiny_llm
    letters = ps.letter_token_ids(tok)
    assert letters["A"] != letters["B"]
    items = ps.load_items()[:3]
    df = ps.score_items(model, tok, items, letters)
    assert len(df) == 6 and np.isfinite(df.delta).all()
    # swapping the letters swaps which log-prob counts as defensive
    a = df[df.order == "def_is_A"].set_index("item")
    b = df[df.order == "def_is_B"].set_index("item")
    assert not np.allclose(a.lp_def, b.lp_def)


def test_steering_hook_changes_and_restores(tiny_llm):
    from mortality_tribe import pilot_steer as ps

    model, tok = tiny_llm
    letters = ps.letter_token_ids(tok)
    items = ps.load_items()[:2]
    layers = pb.decoder_layers(model)
    v = np.random.default_rng(0).normal(size=32).astype(np.float32)
    base = ps.score_items(model, tok, items, letters).delta.to_numpy()
    with pb.residual_hook(layers, 1, v, 0.0):
        same = ps.score_items(model, tok, items, letters).delta.to_numpy()
    with pb.residual_hook(layers, 1, v, 5.0):
        steered = ps.score_items(model, tok, items, letters).delta.to_numpy()
    after = ps.score_items(model, tok, items, letters).delta.to_numpy()
    assert np.allclose(base, same, atol=1e-5)
    assert not np.allclose(base, steered, atol=1e-3)
    assert np.allclose(base, after, atol=1e-5)  # hook removed


def test_sweep_summary_and_decomposition(tiny_llm):
    from mortality_tribe import pilot_steer as ps

    model, tok = tiny_llm
    items = ps.load_items()
    items = [it for it in items if it.axis == "ingroup_norm"][:3] + [it for it in items if it.axis == "outgroup_exclusion"][:3]
    rng = np.random.default_rng(1)
    vectors = {"u_a": rng.normal(size=32), "u_b": rng.normal(size=32)}
    h = ps.residual_norm(model, tok, ["Answer with A or B ."], layer=1)
    assert h > 0
    sweep = ps.steering_sweep(model, tok, items, vectors, [-0.5, 0.5], layer=1, h_norm=h, verbose=False)
    assert set(sweep.vector) == {"none", "u_a", "u_b"}
    summary = ps.summarize_sweep(sweep)
    assert len(summary) == 2 * 2 * 2  # vectors x axes x eps
    dec = ps.decompose(summary)
    assert len(dec) == 4 and np.isfinite(dec.directional).all()


def test_chat_demo_and_alignment(tiny_llm):
    from mortality_tribe import pilot_steer as ps

    model, tok = tiny_llm
    vec = {"u_a": np.random.default_rng(2).normal(size=32)}
    chat = ps.chat_demo(model, tok, ["Which response is closer to what you think ?"], vec, eps=0.5, layer=1,
                        h_norm=1.0, max_new_tokens=5)
    assert list(chat.vector) == ["none", "u_a", "u_a"]
    assert "Q:" in ps.format_chat(chat)
    al = ps.hidden_state_alignment(model, model, tok, ["Answer with A or B ."], layer=1)
    assert al["token_cos_mean"] == pytest.approx(1.0, abs=1e-5)


def test_distinct_ngrams():
    from mortality_tribe import pilot_steer as ps

    assert ps.distinct_ngrams("a b c d e") == 1.0
    assert ps.distinct_ngrams("a b a b a b a b") < 0.5


def test_load_items_accepts_study_schema(tmp_path):
    import json

    from mortality_tribe import pilot_steer as ps

    rec = [{"id": "def_0001", "axis": "outgroup_exclusion", "context": "c", "question": "q?",
            "opt_pos": "exclude them", "opt_neg": "include them"}]
    p = tmp_path / "items.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rec))
    it = ps.load_items(p)[0]
    assert (it.defensive, it.inclusive, it.question) == ("exclude them", "include them", "q?")
    assert "q?" in ps.mc_text(it, it.defensive, it.inclusive)


def test_gram_shortcuts_match_direct_computation():
    rng = np.random.default_rng(5)
    d_a, d_b = rng.normal(size=(7, 300)), rng.normal(size=(9, 300))
    sa, sb = rng.choice([-1.0, 1.0], size=7), rng.choice([-1.0, 1.0], size=9)
    direct = pi._corr((d_a * sa[:, None]).mean(0), (d_b * sb[:, None]).mean(0))
    ca, cb = d_a - d_a.mean(1, keepdims=True), d_b - d_b.mean(1, keepdims=True)
    gram = (sa @ (ca @ cb.T) @ sb) / np.sqrt((sa @ ca @ ca.T @ sa) * (sb @ cb @ cb.T @ sb))
    assert gram == pytest.approx(direct, abs=1e-10)
    s = rng.choice([-1.0, 1.0], size=7)
    assert np.sqrt(s @ (d_a @ d_a.T) @ s) / 7 == pytest.approx(np.linalg.norm((d_a * s[:, None]).mean(0)))
