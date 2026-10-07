"""Steering vectors from your own contrast sentences (CAA), for the pilot.

Mirrors the method of the steering study (Mortality_Steering_v7.ipynb) so the
vectors are comparable:

* Each pair is shown as an A/B question, **in both letter orders**:
  ``{context}\\n{question}\\n(A) ...\\n(B) ...\\nAnswer: (`` followed by the
  letter, and ``d_i = 1/2 [h(AB, A) - h(AB, B) + h(BA, B) - h(BA, A)]``.
  The letter's identity cancels within every pair.
* ``h`` is the hidden state of the letter token after decoder layer
  ``layer`` (``hidden_states[layer + 1]``), i.e. exactly where the pilot
  injects. The vector is the mean of ``d_i`` over training pairs.
* Quality: dz on held-out *families* (>= 1.0), family-wise split-half cosine,
  and, new for two-generator data, the agreement between the vector built
  from one generator's sentences and the other's.
* "clean" vectors: QR projection away from control directions (negative
  affect etc.), with rho = share of the vector that survives.

Pairs without a question (``pos`` / ``neg`` sentences only) are also accepted:
then ``h`` is the mean hidden state over each sentence's tokens.
"""

from __future__ import annotations

import json
import logging
import re
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

POS_KEYS = ("opt_pos", "pos", "positive", "sent_pos")
NEG_KEYS = ("opt_neg", "neg", "negative", "sent_neg")
FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|i'm|i've|i'd|i'll)\b", re.I)
FINALITY = re.compile(r"\b(die|dies|died|dying|dead|death|mortal|mortality|grave|funeral|coffin|buried|"
                      r"forever|never again|final|last|ends?|ending|gone for good|irreversible|"
                      r"cannot be undone|extinct|vanish(?:es|ed)?)\b", re.I)


# --------------------------------------------------------------------------
# Loading and checking
# --------------------------------------------------------------------------


SUFFIXES = (".json", ".jsonl", ".yaml", ".yml", ".csv")


def pair_files(folder: str | Path, include_korean: bool = False) -> list[Path]:
    """``<set>_pairs.<ext>`` files in a folder; ``*_kor`` translations are
    skipped unless asked for (other files, e.g. LoRA chat corpora, are ignored)."""
    return sorted(p for p in Path(folder).glob("*_pairs*") if p.suffix in SUFFIXES
                  and (include_korean or not p.stem.endswith("_kor")))


def load_pairs(path: str | Path, include_korean: bool = False) -> pd.DataFrame:
    """Contrast pairs from one file or all ``*_pairs.*`` files of a folder.

    One row per pair with columns: id, set, family, gen, axis, person,
    exploratory, context, question, pos, neg, mode ("mc" if there is a
    question, else "sentence"). ``set`` comes from ``set_tag`` or the file
    name (``mort_self_pairs.jsonl`` -> ``mort_self``).
    """
    path = Path(path)
    files = pair_files(path, include_korean) if path.is_dir() else [path]
    rows = []
    for f in files:
        default_set = re.sub(r"_pairs(_kor)?$", "", f.stem)
        for i, d in enumerate(_read_records(f)):
            pos = next((d[k] for k in POS_KEYS if d.get(k)), None)
            neg = next((d[k] for k in NEG_KEYS if d.get(k)), None)
            if pos is None or neg is None:
                raise KeyError(f"{f.name} record {i}: need one of {POS_KEYS} and one of {NEG_KEYS}")
            question = str(d.get("question", "") or "").strip()
            rows.append(dict(
                id=str(d.get("id", f"{default_set}_{i:04d}")), set=str(d.get("set_tag") or default_set),
                family=str(d.get("family", "_none")), gen=str(d.get("gen", "_none")).lower(),
                axis=str(d.get("axis", "") or ""), person=str(d.get("person", "") or ""),
                exploratory=bool(d.get("exploratory", False)),
                context=str(d.get("context", "") or "").strip(), question=question,
                pos=str(pos).strip(), neg=str(neg).strip(), mode="mc" if question else "sentence",
                source=f.name,
            ))
    if not rows:
        raise FileNotFoundError(f"No contrast pairs found in {path}")
    return pd.DataFrame(rows)


def _read_records(f: Path) -> list[dict]:
    text = f.read_text(encoding="utf-8")
    if f.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if f.suffix == ".csv":
        return pd.read_csv(f).fillna("").to_dict("records")
    raw = yaml.safe_load(text) if f.suffix in (".yaml", ".yml") else json.loads(text)
    if isinstance(raw, dict):
        raw = raw.get("pairs") or raw.get("items") or next(iter(raw.values()))
    return list(raw)


def sample_pairs(df: pd.DataFrame, n_per_set: int | None, seed: int = 0) -> pd.DataFrame:
    """Up to ``n_per_set`` pairs per set, spread evenly over families
    (round-robin after shuffling within each family)."""
    if not n_per_set:
        return df
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby("set", sort=False):
        queues = [list(rng.permutation(gf.index)) for _, gf in g.groupby("family")]
        picked = []
        while len(picked) < min(n_per_set, len(g)):
            for q in queues:
                if q and len(picked) < n_per_set:
                    picked.append(q.pop())
        keep += picked
    return df.loc[sorted(keep)]


def validate_pairs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Text-level checks before any model is run.

    * ``counts``: pairs per set x generator, and families per set.
    * ``lengths``: mean words of the positive and negative option per set
      (a systematic length difference ends up in the vector).
    * ``issues``: duplicates, identical options, empty fields, first-person
      words in impersonal sets, finality words in sets meant to lack them.
    """
    counts = df.pivot_table(index="set", columns="gen", values="id", aggfunc="count", fill_value=0)
    counts["families"] = df.groupby("set").family.nunique()
    counts["pairs"] = df.groupby("set").size()
    words = lambda s: s.str.split().str.len()  # noqa: E731
    lengths = df.assign(pos_words=words(df.pos), neg_words=words(df.neg)).groupby("set")[
        ["pos_words", "neg_words"]].mean()
    lengths["diff"] = lengths.pos_words - lengths.neg_words

    issues = []

    def flag(row, kind, detail=""):
        issues.append(dict(set=row.set, id=row.id, issue=kind, detail=detail))

    for r in df.itertuples():
        if not r.pos or not r.neg:
            flag(r, "empty option")
        elif r.pos == r.neg:
            flag(r, "identical options")
        impersonal = any(k in r.set for k in ("struct", "impersonal")) or r.person in ("third", "impersonal")
        if impersonal:  # the options carry the contrast; contexts may legitimately say "I"
            hits = FIRST_PERSON.findall(" ".join([r.pos, r.neg]))
            if hits:
                flag(r, "first person in options of impersonal set", ", ".join(sorted(set(h.lower() for h in hits))))
        if any(k in r.set for k in ("nonfinal", "neg_", "sens", "aff", "arb")):
            hits = FINALITY.findall(r.pos)
            if hits:
                flag(r, "finality word in a non-finality set", ", ".join(sorted(set(h.lower() for h in hits))))
    for col in ("pos", "neg"):
        dup = df[df.duplicated(["set", col], keep=False)]
        for r in dup.itertuples():
            flag(r, f"duplicate {col}", r.pos if col == "pos" else r.neg)
    return dict(counts=counts, lengths=lengths, issues=pd.DataFrame(issues, columns=["set", "id", "issue", "detail"]))


# --------------------------------------------------------------------------
# Hidden-state differences
# --------------------------------------------------------------------------


def mc_prompt(context: str, question: str, opt_a: str, opt_b: str) -> str:
    """The steering study's A/B format (``build_mc``)."""
    head = f"{context}\n" if context else ""
    return f"{head}{question}\n(A) {opt_a}\n(B) {opt_b}\nAnswer: ("


def _prompt_ids(tok, text: str, use_chat: bool):
    """Prompt as a user turn (chat template) or plain text, then the letter."""
    from .pilot_steer import build_ids

    if use_chat and getattr(tok, "chat_template", None):
        return build_ids(tok, text, use_chat=True)
    return tok(text, return_tensors="pt").input_ids


def item_differences(model, tok, df: pd.DataFrame, layer: int, use_chat: bool = False,
                     verbose: bool = True) -> np.ndarray:
    """``[n_pairs, hidden]`` letter-cancelled differences, in ``df`` order.

    ``layer`` is the decoder layer after which the pilot injects; hidden
    states are read from ``hidden_states[layer + 1]``.
    """
    import torch

    dev = next(model.parameters()).device
    letter_ids = {}
    for letter in "AB":
        ids = tok(letter, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Letter {letter!r} is not a single token")
        letter_ids[letter] = ids[0]

    def h_letter(prompt_ids, letter):
        full = torch.cat([prompt_ids, torch.tensor([[letter_ids[letter]]])], dim=1).to(dev)
        hs = model(full, output_hidden_states=True).hidden_states[layer + 1]
        return hs[0, -1].float().cpu().numpy()

    def h_mean(text):
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states[layer + 1]
        return hs[0, 1:].float().mean(0).cpu().numpy()  # skip the begin-of-text token

    out = []
    with torch.no_grad():
        for k, r in enumerate(df.itertuples()):
            if r.mode == "mc":
                d = []
                for pos_letter, (a, b) in (("A", (r.pos, r.neg)), ("B", (r.neg, r.pos))):
                    p = _prompt_ids(tok, mc_prompt(r.context, r.question, a, b), use_chat)
                    neg_letter = "B" if pos_letter == "A" else "A"
                    d.append(h_letter(p, pos_letter) - h_letter(p, neg_letter))
                out.append(np.mean(d, 0))
            else:
                head = f"{r.context} " if r.context else ""
                out.append(h_mean(head + r.pos) - h_mean(head + r.neg))
            if verbose and (k + 1) % 100 == 0:
                LOGGER.info("%d / %d pairs", k + 1, len(df))
    return np.stack(out)


# --------------------------------------------------------------------------
# Vectors and their quality
# --------------------------------------------------------------------------


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def orthogonalize(v: np.ndarray, *bases: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove the span of ``bases`` (QR, order-independent). Returns the unit
    residual and rho = |residual| / |v|, the share of the vector that survives."""
    w = np.asarray(v, dtype=np.float64).copy()
    n0 = np.linalg.norm(w)
    bases = [b for b in bases if b is not None]
    if bases:
        q, _ = np.linalg.qr(np.stack([unit(b) for b in bases], axis=1))
        w = w - q @ (q.T @ w)
    return unit(w), float(np.linalg.norm(w) / (n0 + 1e-12))


def _units(families: tp.Sequence[str]) -> np.ndarray:
    """Families as the unit of splitting; every pair is its own unit if there
    is only one family (the study's fallback, at the price of possible
    sentence-pattern leakage between halves)."""
    fams = np.asarray([str(f) for f in families])
    if len(set(fams)) < 2:
        LOGGER.warning("No family labels: splitting by pair instead (pattern leakage possible)")
        return np.array([f"pair{i}" for i in range(len(fams))])
    return fams


def family_holdout(families: tp.Sequence[str], frac: float = 0.2, seed: int = 0) -> np.ndarray:
    """Boolean mask of held-out pairs: whole families, ~``frac`` of the pairs."""
    fams = _units(families)
    uniq = sorted(set(fams))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(uniq))
    held, n = set(), 0
    for f in order:
        if n >= frac * len(fams) and held:
            break
        held.add(f)
        n += int((fams == f).sum())
    if len(held) == len(uniq):  # keep at least one training family
        held.discard(order[-1])
    return np.isin(fams, list(held))


def vector_quality(diffs: np.ndarray, families: tp.Sequence[str], gens: tp.Sequence[str] | None = None,
                   seed: int = 0) -> dict[str, float]:
    """Gate-1 style quality of the mean-difference vector.

    * ``dz_holdout``: projections of held-out-family pairs onto the vector
      built from the other families, mean / sd (study criterion: >= 1.0).
    * ``dz_loo``: the same with every pair left out in turn (stable with few pairs).
    * ``split_half``: cosine of vectors from two disjoint halves of the
      training families (criterion: >= 0.8).
    * ``gen_cos`` / ``dz_cross_gen``: vector from one generator's pairs vs the
      other's, and dz of each generator's pairs on the other's vector. Low
      values mean the direction carries the generator's style, not the content.
    """
    fams = _units(families)
    ho = family_holdout(fams, seed=seed)
    v_tr = unit(diffs[~ho].mean(0))
    p = diffs[ho] @ v_tr
    total = diffs.sum(0)
    loo = np.array([d @ unit(total - d) for d in diffs])  # each pair on the vector of all the others
    out = dict(n_pairs=len(diffs), n_families=len(set(fams)), dz_holdout=float(p.mean() / (p.std(ddof=1) + 1e-12)),
               dz_loo=float(loo.mean() / (loo.std(ddof=1) + 1e-12)))
    tr_fams = sorted(set(fams[~ho]))
    rng = np.random.default_rng(seed)
    rng.shuffle(tr_fams)
    half = set(tr_fams[: len(tr_fams) // 2])
    in_a = np.isin(fams, list(half)) & ~ho
    in_b = ~np.isin(fams, list(half)) & ~ho
    out["split_half"] = float(unit(diffs[in_a].mean(0)) @ unit(diffs[in_b].mean(0))) \
        if in_a.any() and in_b.any() else np.nan
    if gens is not None:
        g = np.asarray(gens)
        names = sorted(set(g))
        if len(names) == 2:
            va, vb = unit(diffs[g == names[0]].mean(0)), unit(diffs[g == names[1]].mean(0))
            pa, pb = diffs[g == names[0]] @ vb, diffs[g == names[1]] @ va
            out.update(gen_cos=float(va @ vb),
                       dz_cross_gen=float(np.mean([pa.mean() / (pa.std(ddof=1) + 1e-12),
                                                   pb.mean() / (pb.std(ddof=1) + 1e-12)])))
    return out


def build_vector(diffs: np.ndarray, families: tp.Sequence[str], seed: int = 0) -> np.ndarray:
    """Unit mean-difference vector from the training families only (as in the
    study), so ``dz_holdout`` stays an out-of-sample estimate."""
    ho = family_holdout(_units(families), seed=seed)
    return unit(diffs[~ho].mean(0))


def extract_vectors(model, tok, pairs: pd.DataFrame, layer: int, targets: tp.Sequence[str],
                    controls: tp.Sequence[str] = (), extra: tp.Sequence[str] = (), use_chat: bool = False,
                    max_pairs: int | None = None, seed: int = 0) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """One vector ``v_<set>`` per available set, plus ``v_<target>_clean`` with
    the control directions projected out. Returns (vectors, quality table)."""
    available = set(pairs.set)
    vecs, rows = {}, []
    for name in dict.fromkeys([*targets, *controls, *extra]):
        if name not in available:
            continue
        sub = pairs[pairs.set == name]
        if max_pairs and len(sub) > max_pairs:
            sub = sub.sample(max_pairs, random_state=seed).sort_index()
        LOGGER.info("CAA %s: %d pairs", name, len(sub))
        d = item_differences(model, tok, sub, layer, use_chat=use_chat)
        vecs[f"v_{name}"] = build_vector(d, sub.family, seed)
        rows.append(dict(vector=f"v_{name}", **vector_quality(d, sub.family, sub.gen, seed)))
    ctrl = [vecs[f"v_{c}"] for c in controls if f"v_{c}" in vecs]
    for t in targets:
        if f"v_{t}" in vecs and ctrl:
            vecs[f"v_{t}_clean"], rho = orthogonalize(vecs[f"v_{t}"], *ctrl)
            rows.append(dict(vector=f"v_{t}_clean", rho=rho))
    quality = pd.DataFrame(rows)
    if len(quality):
        quality["passes"] = (quality.get("dz_holdout", np.nan) >= 1.0) & (quality.get("split_half", np.nan) >= 0.8)
    return vecs, quality


def split_def_by_axis(pairs: pd.DataFrame) -> pd.DataFrame:
    """Defense pairs as two sets, ``def_norm`` and ``def_excl``: the study
    orthogonalises against both axes separately (a single v_def is mostly the
    exclusion axis)."""
    out = pairs.copy()
    is_def = out.set == "def"
    out.loc[is_def & (out.axis == "ingroup_norm"), "set"] = "def_norm"
    out.loc[is_def & (out.axis == "outgroup_exclusion"), "set"] = "def_excl"
    return out
