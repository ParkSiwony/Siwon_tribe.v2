"""Offline stand-ins for the pilot notebook's ``DRY_RUN`` mode.

``FakeTribe`` mimics the parts of ``tribev2.TribeModel`` the pilot touches:
``predict`` for video and word events, ``data.get_loaders`` returning batches
of text features, ``data.text_feature`` (with a Llama ``model`` and
``tokenizer``) and the brain network ``_model``. Its text features come from a
real but tiny, randomly initialised Llama, so steering that Llama changes the
fake brain in the same way steering Llama-3.2-3B changes TRIBE.

Every number a dry run prints is meaningless; it only shows that each step
runs on this machine.
"""

from __future__ import annotations

import dataclasses
import re
import types
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from .pilot_bridge import hidden_state_groups


def tiny_llama(save_dir: str | Path, corpus: str, hidden: int = 32, n_layers: int = 8, seed: int = 0):
    """Random ``LlamaForCausalLM`` + word-level tokenizer with a chat template.

    The config is saved to ``save_dir`` so ``AutoConfig.from_pretrained`` works
    offline, as it does for the real model.
    """
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    corpus += " Answer with A or B . ( ) : user assistant system"
    words = sorted(set(re.findall(r"\w+|[^\w\s]", corpus)))
    specials = ["<unk>", "<s>", "</s>", "<pad>", "<|user|>", "<|assistant|>", "<|system|>"]
    vocab = {w: i for i, w in enumerate(specials + [w for w in words if w not in specials])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  unk_token="<unk>")
    tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|> {% endif %}")
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=3)
    model = LlamaForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(save_dir)
    return model, tok


@dataclasses.dataclass
class FakeBatch:
    data: dict
    segments: list

    def to(self, device):
        return FakeBatch({k: v.to(device) for k, v in self.data.items()}, self.segments)


class _FakeBrain:
    """Builds the toy brain network lazily (keeps torch an optional import)."""

    @staticmethod
    def build(dim: int, n_vertices: int, seed: int):
        import torch

        class FakeBrain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                g = torch.Generator().manual_seed(seed)
                self.w1 = torch.nn.Parameter(torch.randn(dim, 64, generator=g) / dim ** 0.5)
                self.w2 = torch.nn.Parameter(torch.randn(64, n_vertices, generator=g) / 8)

            @property
            def device(self):
                return self.w1.device

            def forward(self, batch):
                x = batch.data["text"].float().mean(1).transpose(1, 2)  # B, T, D
                y = (torch.tanh(x @ self.w1) @ self.w2).transpose(1, 2)  # B, V, T (2 Hz)
                return torch.nn.functional.avg_pool1d(y, 2)  # 1 Hz, like TRIBE's pooler

        return FakeBrain()


class FakeTribe:
    """See module docstring. ``llm`` is the CausalLM whose base model plays
    TRIBE's text extractor; steer the same CausalLM in step 3."""

    def __init__(self, llm, tok, config_dir: str | Path, n_vertices: int = 2000, seed: int = 0,
                 peak_lag: float = 1.0):
        base = llm.model  # LlamaModel, like neuralset's AutoModel
        self._model = _FakeBrain.build(llm.config.hidden_size, n_vertices, seed)
        self.remove_empty_segments = True
        self.n_vertices, self.peak_lag = n_vertices, peak_lag
        self._visual = np.random.default_rng(seed + 1).normal(size=(16 * 16 * 3, n_vertices)) / 30
        feat = types.SimpleNamespace(layers=[0.5, 0.75, 1.0], layer_aggregation="group_mean",
                                     model_name=str(config_dir), batch_size=1, model=base, tokenizer=tok)
        self.data = types.SimpleNamespace(text_feature=feat, get_loaders=self._get_loaders, TR=1.0)

    # -- text ----------------------------------------------------------------
    def _text_tensor(self, words: pd.DataFrame, n_bins: int = 200):
        import torch

        feat = self.data.text_feature
        llm, tok = feat.model, feat.tokenizer
        groups = hidden_state_groups(feat.layers, feat.layer_aggregation, llm.config.num_hidden_layers + 1)
        x = torch.zeros(len(groups), llm.config.hidden_size, n_bins)
        dev = next(llm.parameters()).device
        with torch.no_grad():
            for w in words.itertuples():
                ids = tok(w.context, return_tensors="pt", add_special_tokens=False).input_ids[:, -256:].to(dev)
                hs = llm(ids, output_hidden_states=True).hidden_states
                vec = torch.stack([torch.stack(hs[a:b])[:, 0, -1].float().mean(0) for a, b in groups]).cpu()
                b = int(w.start * 2)
                if b < n_bins:
                    x[:, :, b] += vec
        return x

    def _get_loaders(self, events: pd.DataFrame, split_to_build: str = "all"):
        words = events[events.type == "Word"]
        batches = []
        for tl, g in words.groupby("timeline", sort=False):
            x = self._text_tensor(g)[None]
            batches.append(FakeBatch({"text": x}, [types.SimpleNamespace(timeline=tl, start=0.0, duration=100.0)]))
        return {split_to_build: batches}

    # -- predict -------------------------------------------------------------
    def predict(self, events: pd.DataFrame, verbose: bool = True):
        import torch

        preds, segs = [], []
        for r in events[events.type == "Video"].itertuples():
            p = self._video_response(r.filepath)
            preds.append(p)
            segs += [types.SimpleNamespace(timeline=r.timeline, start=float(t), duration=1.0) for t in range(len(p))]
        words = events[events.type == "Word"]
        if len(words):
            for b in self._get_loaders(events)["all"]:
                with torch.no_grad():
                    out = self._model(b.to(self._model.device))[0].T.cpu().numpy()  # T, V
                tl = b.segments[0].timeline
                n = int(np.ceil(words[words.timeline == tl].start.max())) + 10
                preds.append(out[:n])
                segs += [types.SimpleNamespace(timeline=tl, start=float(t), duration=1.0) for t in range(n)]
        return np.concatenate(preds), segs

    def _video_response(self, path: str) -> np.ndarray:
        from moviepy import VideoFileClip
        from PIL import Image

        with VideoFileClip(str(path)) as clip:
            n = int(np.floor(clip.duration))
            frames = [np.asarray(Image.fromarray(clip.get_frame(t + 0.5)).resize((16, 16)), float).ravel() / 255
                      for t in range(n)]
        drive = np.stack(frames) @ self._visual  # n, V
        t = np.arange(n)
        kernel = np.exp(-0.5 * ((t[:, None] - t[None, :] - self.peak_lag) / 1.5) ** 2)  # response, frame
        return (kernel @ drive).astype(np.float32)


def corpus_for(*texts: tp.Iterable[str]) -> str:
    return " ".join(" ".join(t) for t in texts)


def write_dry_pairs(folder: str | Path, n_families: int = 3, per_family: int = 4) -> Path:
    """Tiny synthetic contrast-pair files in the steering study's format (dry run only)."""
    import json

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    templates = {
        "mort_self": ("Nothing I do now can bring this moment back.", "I can return to this moment whenever I like."),
        "mort_struct": ("The old bridge is gone and will never stand again.", "The old bridge is closed and will open again."),
        "neg_sensory": ("The room smells of burnt rubber and sour milk.", "The room smells of fresh bread and coffee."),
        "neg_affect": ("A heavy sadness settles over the whole evening.", "A light calm settles over the whole evening."),
        "def": ("The newcomers should be kept out of the club.", "The newcomers should be welcomed into the club."),
    }
    for name, (pos, neg) in templates.items():
        with open(folder / f"{name}_pairs.jsonl", "w", encoding="utf-8") as f:
            for fam in range(n_families):
                for k in range(per_family):
                    rec = dict(id=f"{name}_{fam}_{k}", family=f"family{fam}", gen=("gpt", "kimi")[k % 2],
                               context=f"Situation {fam} {k}.", question="Which statement fits better?",
                               opt_pos=pos, opt_neg=neg)
                    if name == "def":
                        rec["axis"] = ("ingroup_norm", "outgroup_exclusion")[fam % 2]
                    f.write(json.dumps(rec) + "\n")
    return folder
