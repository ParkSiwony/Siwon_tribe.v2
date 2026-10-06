import re
from pathlib import Path

import numpy as np
import pytest

from mortality_tribe import audio, stimuli

STIM = Path(__file__).parents[1] / "stimuli"

# Vocabulary of the manipulations: must never leak into the probes.
FORBIDDEN_IN_PROBES = {
    "death", "die", "dies", "dying", "dead", "mortality", "mortal", "funeral", "grave", "coffin",
    "pain", "tooth", "teeth", "dentist", "drill", "ache", "time", "years", "weeks", "months",
    "future", "limited", "last", "final", "end", "ending", "remaining", "remain", "forever",
}


@pytest.fixture(scope="module")
def stim():
    return stimuli.load_stimuli(STIM)


def test_stimuli_are_balanced(stim):
    assert set(stim.primes) == {"MS", "PAIN", "LT", "ET", "NEU"}
    assert len({len(v) for v in stim.primes.values()}) == 1
    assert len({len(v) for v in stim.probes.values()}) == 1


def test_prime_lengths_are_matched(stim):
    lengths = {c: [len(i["text"].split()) for i in items] for c, items in stim.primes.items()}
    for c, ls in lengths.items():
        assert all(80 <= n <= 115 for n in ls), (c, ls)
    means = [np.mean(ls) for ls in lengths.values()]
    assert max(means) - min(means) < 15, lengths


def test_probes_avoid_manipulation_vocabulary(stim):
    for cat, items in stim.probes.items():
        for text in items:
            words = set(re.findall(r"[a-z]+", text.lower()))
            assert not words & FORBIDDEN_IN_PROBES, (cat, text, words & FORBIDDEN_IN_PROBES)
            assert 9 <= len(text.split()) <= 13, text


def test_latin_design_yokes_probes_across_conditions(stim):
    trials = stimuli.build_design(stim)
    assert len(trials) == 5 * 4 * 2
    by_key = {}
    for t in trials:
        probes = tuple(s.text for s in t.segments if s.kind == "probe")
        by_key.setdefault((t.item_index, t.delay), set()).add(probes)
    # every condition hears exactly the same probes in the same order
    assert all(len(v) == 1 for v in by_key.values())
    # every probe sentence is used exactly once per condition and delay
    used = [s.text for t in trials if t.prime_cond == "MS" and t.delay == "immediate"
            for s in t.segments if s.kind == "probe"]
    assert sorted(used) == sorted(p for v in stim.probes.values() for p in v)


def test_probe_order_has_no_category_repeats(stim):
    for t in stimuli.build_design(stim):
        cats = [s.category for s in t.segments if s.kind == "probe"]
        assert all(a != b for a, b in zip(cats, cats[1:]))


def test_delayed_trials_contain_filler(stim):
    for t in stimuli.build_design(stim):
        kinds = [s.kind for s in t.segments]
        assert kinds[0] == "prime"
        assert ("filler" in kinds) == (t.delay == "delayed")


def test_full_crossing(stim):
    trials = stimuli.build_design(stim, crossing="full", delays=["immediate"])
    assert len(trials) == 5 * 4 * 4


def test_design_roundtrip(stim, tmp_path):
    trials = stimuli.build_design(stim)
    stimuli.save_design(trials, tmp_path / "d.yaml")
    assert stimuli.load_design(tmp_path / "d.yaml") == trials


def test_render_trial_onsets_and_identical_probe_audio(stim, tmp_path):
    trials = stimuli.build_design(stim)
    cache = audio.SegmentCache(tmp_path / "cache", audio.SilentBackend())
    import soundfile as sf

    t0 = trials[0]
    wav, tab = audio.render_trial(t0, cache, tmp_path / "audio")
    data, sr = sf.read(wav)
    assert sr == audio.SAMPLE_RATE
    assert tab.onset.iloc[0] == pytest.approx(t0.lead_in)
    ends = tab.onset + tab.duration + [s.gap_after for s in t0.segments]
    assert np.allclose(tab.onset.iloc[1:].to_numpy(), ends.iloc[:-1].to_numpy(), atol=1e-3)
    assert len(data) / sr == pytest.approx(ends.iloc[-1] + t0.tail, abs=0.01)
    # same probe text -> same cached waveform
    text = tab[tab.kind == "probe"].text.iloc[0]
    assert np.array_equal(cache.get(text), cache.get(text))
