"""Pilot step 3: steer Llama with the brain-derived directions.

Two read-outs:

1. **Defense items**, as in the steering study: in-group-norm and
   out-group-exclusion scenarios with a defensive and an inclusive option.
   Each item is asked twice with the letters swapped, and
   ``delta = logP(defensive letter) - logP(inclusive letter)`` is averaged
   over both orders, which cancels any letter preference.
2. **Chat answers** to open questions about humanity at -eps / 0 / +eps, to
   see what the direction does in plain language.

Steering adds ``eps * |h| * u`` to the residual stream after one decoder
layer at every position, where |h| is the median residual-stream norm at that
layer (so eps is relative, as in the steering notebook). Compare every target
direction with the null directions at the same eps: any vector added to the
residual stream changes the text somewhat.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .pilot_bridge import decoder_layers, residual_hook

LOGGER = logging.getLogger(__name__)

DEFAULT_QUESTION = "Which response is closer to what you think?"
ANSWER_PREFIX = "Answer: ("
SAMPLE_ITEMS = Path(__file__).parents[1] / "stimuli" / "defense_items_sample.yaml"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def load_llm(name_or_path: str, device: str = "cuda", dtype: str = "bfloat16", local_files_only: bool = False):
    """Causal LM + tokenizer, frozen, in eval mode (transformers 4.x and 5.x)."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = getattr(torch, dtype)
    major = int(transformers.__version__.split(".")[0])
    kw = {"dtype": torch_dtype} if major >= 5 else {"torch_dtype": torch_dtype}
    model = AutoModelForCausalLM.from_pretrained(name_or_path, local_files_only=local_files_only, **kw)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def build_ids(tok, user_text: str, assistant_prefix: str = "", system: str | None = None, use_chat: bool = True):
    """Token ids for one user turn (chat template if the tokenizer has one),
    optionally followed by the start of the assistant's answer."""
    import torch

    if use_chat and getattr(tok, "chat_template", None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_text}]
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        ids = out["input_ids"] if hasattr(out, "keys") else out
        ids = torch.as_tensor(ids)
        if ids.ndim == 1:
            ids = ids[None]
    else:
        text = (f"{system}\n\n" if system else "") + user_text + "\n"
        ids = tok(text, return_tensors="pt").input_ids
    if assistant_prefix:
        extra = tok(assistant_prefix, add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([ids, extra], dim=1)
    return ids


def letter_token_ids(tok, prefix: str = ANSWER_PREFIX) -> dict[str, int]:
    """Ids of 'A' and 'B' as the single token that follows ``prefix``."""
    base = tok(prefix, add_special_tokens=False).input_ids
    out = {}
    for letter in "AB":
        ids = tok(prefix + letter, add_special_tokens=False).input_ids
        if ids[: len(base)] != base or len(ids) != len(base) + 1:
            raise ValueError(f"{letter!r} is not a single token after {prefix!r}; change the prefix")
        out[letter] = ids[-1]
    return out


def residual_norm(model, tok, texts: tp.Sequence[str], layer: int, use_chat: bool = True) -> float:
    """Median norm of the residual stream after decoder ``layers[layer]``.

    The first position is excluded: Llama's begin-of-text token has a hidden
    state many times larger than any other (an attention sink) and would
    dominate the scale.
    """
    import torch

    dev = next(model.parameters()).device
    norms = []
    with torch.no_grad():
        for t in texts:
            ids = build_ids(tok, t, use_chat=use_chat).to(dev)
            hs = model(ids, output_hidden_states=True).hidden_states[layer + 1][0, 1:]
            norms.append(hs.float().norm(dim=-1).cpu().numpy())
    return float(np.median(np.concatenate(norms)))


# --------------------------------------------------------------------------
# Defense items
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DefenseItem:
    id: str
    axis: str  # "ingroup_norm" or "outgroup_exclusion"
    scenario: str
    defensive: str
    inclusive: str
    question: str = ""


def load_items(path: str | Path | None = None, defensive_key: str | None = None,
               inclusive_key: str | None = None) -> list[DefenseItem]:
    """Defense items from YAML / JSON / JSONL.

    Accepted field names: axis; scenario | context | prompt (+ optional
    question); defensive | opt_def | opt_pos; inclusive | opt_inc | opt_neg.
    In the steering study's files ``opt_pos`` is the defensive option: check
    the printed example before trusting the mapping, or pass the keys.
    """
    path = Path(path) if path is not None else SAMPLE_ITEMS
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
        raw = raw["items"] if isinstance(raw, dict) else raw
    elif path.suffix == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
        raw = raw["items"] if isinstance(raw, dict) else raw

    def pick(d, keys, what):
        for k in keys:
            if k and d.get(k):
                return str(d[k])
        raise KeyError(f"No {what} field in item {d.get('id', d)}; tried {keys}")

    items = []
    for i, d in enumerate(raw):
        items.append(DefenseItem(
            id=str(d.get("id", f"item{i:03d}")),
            axis=pick(d, ["axis"], "axis"),
            scenario=pick(d, ["scenario", "context", "prompt"], "scenario"),
            question=str(d.get("question", "") or ""),
            defensive=pick(d, [defensive_key, "defensive", "opt_def", "opt_pos"], "defensive option"),
            inclusive=pick(d, [inclusive_key, "inclusive", "opt_inc", "opt_neg"], "inclusive option"),
        ))
    return items


def mc_text(item: DefenseItem, first: str, second: str, instruction: str = "Answer with A or B.") -> str:
    question = item.question or DEFAULT_QUESTION
    return f"{item.scenario}\n\n{question}\n(A) {first}\n(B) {second}\n\n{instruction}"


def score_items(model, tok, items: tp.Sequence[DefenseItem], letters: dict[str, int],
                use_chat: bool = True) -> pd.DataFrame:
    """One row per item and letter order with the defensive-minus-inclusive log-prob."""
    import torch

    dev = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for it in items:
            for order in ("def_is_A", "def_is_B"):
                a, b = (it.defensive, it.inclusive) if order == "def_is_A" else (it.inclusive, it.defensive)
                ids = build_ids(tok, mc_text(it, a, b), assistant_prefix=ANSWER_PREFIX, use_chat=use_chat).to(dev)
                lp = torch.log_softmax(model(ids).logits[0, -1].float(), dim=-1)
                lp_a, lp_b = float(lp[letters["A"]]), float(lp[letters["B"]])
                lp_def, lp_inc = (lp_a, lp_b) if order == "def_is_A" else (lp_b, lp_a)
                rows.append(dict(item=it.id, axis=it.axis, order=order, lp_def=lp_def, lp_inc=lp_inc,
                                 delta=lp_def - lp_inc, mass=float(np.exp(lp_a) + np.exp(lp_b))))
    return pd.DataFrame(rows)


def steering_sweep(model, tok, items, vectors: dict[str, np.ndarray], eps_list: tp.Sequence[float], layer: int,
                   h_norm: float, use_chat: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Score all items unsteered (once) and under every vector x eps."""
    letters = letter_token_ids(tok)
    layers = decoder_layers(model)
    base = score_items(model, tok, items, letters, use_chat).assign(vector="none", eps_rel=0.0)
    out = [base]
    for name, v in vectors.items():
        unit = np.asarray(v, dtype=np.float32) / np.linalg.norm(v)
        for eps in eps_list:
            if eps == 0:
                continue
            with residual_hook(layers, layer, unit, eps * h_norm):
                df = score_items(model, tok, items, letters, use_chat)
            out.append(df.assign(vector=name, eps_rel=eps))
            if verbose:
                LOGGER.info("%s eps=%+.2f: mean delta %.3f", name, eps, df.delta.mean())
    return pd.concat(out, ignore_index=True)


def summarize_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Per vector x axis x eps: change of the item-level delta vs unsteered
    (paired over items, letter orders averaged first), t-test, format mass."""
    from scipy import stats

    per_item = df.groupby(["vector", "eps_rel", "axis", "item"], as_index=False).agg(delta=("delta", "mean"),
                                                                                    mass=("mass", "mean"))
    base = per_item[per_item.vector == "none"].set_index("item").delta
    rows = []
    for (vec, eps, axis), g in per_item[per_item.vector != "none"].groupby(["vector", "eps_rel", "axis"]):
        diff = g.set_index("item").delta - base.reindex(g.item).to_numpy()
        t, p = stats.ttest_1samp(diff.to_numpy(), 0.0) if len(diff) > 1 else (np.nan, np.nan)
        rows.append(dict(vector=vec, axis=axis, eps_rel=eps, change=diff.mean(), se=diff.std(ddof=1) / np.sqrt(len(diff)),
                         t=t, p=p, n_items=len(diff), toward_defense=float((diff > 0).mean()), mass=g.mass.mean()))
    return pd.DataFrame(rows).sort_values(["vector", "axis", "eps_rel"]).reset_index(drop=True)


def decompose(summary: pd.DataFrame) -> pd.DataFrame:
    """Directional [f(+e) - f(-e)]/2 vs symmetric [f(+e) + f(-e)]/2 parts.

    Steering flips with the sign of the vector; a generic perturbation moves
    the answers the same way whichever sign is used.
    """
    rows = []
    for (vec, axis), g in summary.groupby(["vector", "axis"]):
        s = g.set_index("eps_rel").change
        for e in sorted({abs(x) for x in s.index if x > 0}):
            if e in s.index and -e in s.index:
                d, m = (s[e] - s[-e]) / 2, (s[e] + s[-e]) / 2
                rows.append(dict(vector=vec, axis=axis, eps=e, directional=d, symmetric=m,
                                 ratio=abs(d) / (abs(m) + 1e-12)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Chat demo
# --------------------------------------------------------------------------


def generate(model, tok, question: str, system: str | None = None, max_new_tokens: int = 120) -> str:
    import torch

    dev = next(model.parameters()).device
    ids = build_ids(tok, question, system=system).to(dev)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                             do_sample=False, repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def distinct_ngrams(text: str, n: int = 3) -> float:
    """Share of distinct word n-grams; values well below ~0.6 signal degenerate loops."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def chat_demo(model, tok, questions: tp.Sequence[str], vectors: dict[str, np.ndarray], eps: float, layer: int,
              h_norm: float, system: str | None = None, max_new_tokens: int = 120) -> pd.DataFrame:
    """Answers per question: unsteered, then each vector at +eps and -eps."""
    layers = decoder_layers(model)
    rows = []
    for q in questions:
        conds = [("none", 0.0)] + [(name, s * eps) for name in vectors for s in (1, -1)]
        for name, e in conds:
            if name == "none":
                text = generate(model, tok, q, system, max_new_tokens)
            else:
                v = np.asarray(vectors[name], dtype=np.float32)
                with residual_hook(layers, layer, v / np.linalg.norm(v), e * h_norm):
                    text = generate(model, tok, q, system, max_new_tokens)
            rows.append(dict(question=q, vector=name, eps_rel=e, answer=text, distinct3=distinct_ngrams(text)))
    return pd.DataFrame(rows)


def format_chat(df: pd.DataFrame, width: int = 100) -> str:
    """Readable side-by-side dump of ``chat_demo`` output."""
    import textwrap

    lines = []
    for q, g in df.groupby("question", sort=False):
        lines += ["=" * width, f"Q: {q}", "=" * width]
        for r in g.itertuples():
            tag = "unsteered" if r.vector == "none" else f"{r.vector} {r.eps_rel:+.2f}"
            flag = "  [!! repetitive]" if r.distinct3 < 0.6 else ""
            lines.append(f"--- {tag}{flag}")
            lines += textwrap.wrap(r.answer, width) or ["(empty)"]
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Base vs Instruct
# --------------------------------------------------------------------------


def hidden_state_alignment(model_a, model_b, tok, texts: tp.Sequence[str], layer: int) -> dict[str, float]:
    """How similar are two models' residual streams at one layer on the same text?

    TRIBE's directions are found in the *base* model's space. If they are
    injected into Instruct, the two spaces must line up at that layer.
    ``token_cos`` is the mean cosine between the two models' hidden states,
    token by token (first token excluded).
    """
    import torch

    cos = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids
            ha = model_a(ids.to(next(model_a.parameters()).device), output_hidden_states=True).hidden_states[layer + 1][0, 1:]
            hb = model_b(ids.to(next(model_b.parameters()).device), output_hidden_states=True).hidden_states[layer + 1][0, 1:]
            ha, hb = ha.float().cpu(), hb.float().cpu()
            cos.append(torch.nn.functional.cosine_similarity(ha, hb, dim=-1).numpy())
    c = np.concatenate(cos)
    return dict(token_cos_mean=float(c.mean()), token_cos_p10=float(np.percentile(c, 10)), n_tokens=len(c))
