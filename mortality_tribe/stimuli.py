"""Stimulus loading and experimental design.

A *trial* is one continuous audio stream fed to TRIBE v2 as its own timeline::

    [lead-in] PRIME [gap] (FILLER [gap])? PROBE [gap] PROBE [gap] ... [tail]

The same probe sentences, in the same order, follow every prime condition
(the probes are "yoked" across conditions). Any difference in the predicted
response to a probe can therefore only come from the preceding prime.
"""

from __future__ import annotations

import dataclasses
import itertools
import random
import typing as tp
from pathlib import Path

import yaml

DELAYS = ("immediate", "delayed")


@dataclasses.dataclass(frozen=True)
class Timing:
    """Silences (seconds) inserted between stimulus segments."""

    lead_in: float = 2.0
    # >= 12 s so the prime's hemodynamic response (and undershoot) has mostly
    # decayed before the first probe; shorter gaps leak the prime-evoked
    # pattern into the first probe window (a pure hemodynamic confound).
    after_prime: float = 12.0
    after_filler: float = 12.0
    between_probes: float = 8.0  # IBC language protocols use 8 s between items
    tail: float = 10.0  # lets the last probe's hemodynamic response unfold


@dataclasses.dataclass(frozen=True)
class Segment:
    kind: tp.Literal["prime", "filler", "probe"]
    text: str
    item_id: str
    category: str  # prime condition, "FILLER", or probe category
    gap_after: float


@dataclasses.dataclass
class Trial:
    trial_id: str
    prime_cond: str
    prime_id: str
    item_index: int  # index of the prime item within its condition (pairing unit)
    delay: str
    probe_set: int
    filler_id: str | None
    lead_in: float
    tail: float
    segments: list[Segment]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trial":
        d = dict(d)
        d["segments"] = [Segment(**s) for s in d["segments"]]
        return cls(**d)


@dataclasses.dataclass
class Stimuli:
    conditions: dict[str, dict]
    primes: dict[str, list[dict]]
    probe_categories: dict[str, str]
    probes: dict[str, list[str]]
    fillers: list[dict]


def load_stimuli(stim_dir: str | Path) -> Stimuli:
    stim_dir = Path(stim_dir)
    primes = yaml.safe_load((stim_dir / "primes.yaml").read_text())
    probes = yaml.safe_load((stim_dir / "probes.yaml").read_text())
    fillers = yaml.safe_load((stim_dir / "fillers.yaml").read_text())
    stim = Stimuli(
        conditions=primes["conditions"],
        primes={k: list(v) for k, v in primes["items"].items()},
        probe_categories=probes["categories"],
        probes={k: list(v) for k, v in probes["items"].items()},
        fillers=list(fillers["items"]),
    )
    validate_stimuli(stim)
    return stim


def validate_stimuli(stim: Stimuli) -> None:
    missing = set(stim.conditions) ^ set(stim.primes)
    if missing:
        raise ValueError(f"Conditions and prime items disagree on: {sorted(missing)}")
    n_items = {k: len(v) for k, v in stim.primes.items()}
    if len(set(n_items.values())) != 1:
        raise ValueError(f"Every condition needs the same number of items: {n_items}")
    n_probes = {k: len(v) for k, v in stim.probes.items()}
    if len(set(n_probes.values())) != 1:
        raise ValueError(f"Every probe category needs the same size: {n_probes}")
    if set(stim.probe_categories) != set(stim.probes):
        raise ValueError("Probe categories and probe items disagree")
    all_probes = [p for v in stim.probes.values() for p in v]
    if len(set(all_probes)) != len(all_probes):
        raise ValueError("Duplicate probe sentences")


def make_probe_sets(
    probes: dict[str, list[str]], n_sets: int, seed: int
) -> list[list[tuple[str, str]]]:
    """Split probes into ``n_sets`` sets, each with the same number per category.

    Each set is returned in a fixed pseudo-random order that never places two
    probes of the same category back to back. The order is shared by every
    prime condition, so serial position is matched across conditions.
    """
    rng = random.Random(seed)
    n_per_cat = len(next(iter(probes.values())))
    if n_per_cat % n_sets:
        raise ValueError(f"{n_per_cat} probes per category not divisible by {n_sets}")
    k = n_per_cat // n_sets
    shuffled = {c: rng.sample(items, len(items)) for c, items in probes.items()}
    sets = []
    for s in range(n_sets):
        pool = [(c, p) for c, items in shuffled.items() for p in items[s * k : (s + 1) * k]]
        sets.append(_order_without_repeats(pool, rng))
    return sets


def _order_without_repeats(
    pool: list[tuple[str, str]], rng: random.Random, max_tries: int = 10_000
) -> list[tuple[str, str]]:
    for _ in range(max_tries):
        order = rng.sample(pool, len(pool))
        if all(a[0] != b[0] for a, b in zip(order, order[1:])):
            return order
    raise RuntimeError("Could not find a probe order without category repeats")


def build_design(
    stim: Stimuli,
    timing: Timing = Timing(),
    delays: tp.Sequence[str] = DELAYS,
    n_probe_sets: int | None = None,
    crossing: tp.Literal["latin", "full"] = "latin",
    seed: int = 0,
) -> list[Trial]:
    """Create the list of trials.

    ``latin``: prime item *j* of every condition is followed by probe set *j*
    (so all conditions see exactly the same probes; 1 trial per prime item).
    ``full``: every prime item is followed by every probe set (more trials,
    lets probe and prime-item variance be separated).
    """
    for d in delays:
        if d not in DELAYS:
            raise ValueError(f"Unknown delay {d!r}; expected one of {DELAYS}")
    n_items = len(next(iter(stim.primes.values())))
    n_probe_sets = n_probe_sets or n_items
    probe_sets = make_probe_sets(stim.probes, n_probe_sets, seed)
    if crossing == "latin" and n_probe_sets != n_items:
        raise ValueError("Latin crossing needs as many probe sets as prime items")

    trials = []
    for cond, items in stim.primes.items():
        for j, item in enumerate(items):
            set_ids = [j] if crossing == "latin" else range(n_probe_sets)
            for s, delay in itertools.product(set_ids, delays):
                segs = [Segment("prime", _clean(item["text"]), item["id"], cond, timing.after_prime)]
                filler_id = None
                if delay == "delayed":
                    filler = stim.fillers[j % len(stim.fillers)]
                    filler_id = filler["id"]
                    segs.append(
                        Segment("filler", _clean(filler["text"]), filler_id, "FILLER", timing.after_filler)
                    )
                for i, (cat, text) in enumerate(probe_sets[s]):
                    segs.append(
                        Segment("probe", text, f"{cat}_s{s}_{i:02d}", cat, timing.between_probes)
                    )
                trials.append(
                    Trial(
                        trial_id=f"{item['id']}_set{s}_{delay}",
                        prime_cond=cond,
                        prime_id=item["id"],
                        item_index=j,
                        delay=delay,
                        probe_set=s,
                        filler_id=filler_id,
                        lead_in=timing.lead_in,
                        tail=timing.tail,
                        segments=segs,
                    )
                )
    return trials


def _clean(text: str) -> str:
    return " ".join(text.split())


def save_design(trials: list[Trial], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump([t.to_dict() for t in trials], sort_keys=False))


def load_design(path: str | Path) -> list[Trial]:
    return [Trial.from_dict(d) for d in yaml.safe_load(Path(path).read_text())]
