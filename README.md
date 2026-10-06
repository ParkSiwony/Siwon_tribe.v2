# Mortality or limited-time salience induces *what*? An in-silico study with TRIBE v2

This project uses Meta's **TRIBE v2** brain encoder ([paper](https://arxiv.org/abs/2605.04326),
[code](https://github.com/facebookresearch/tribev2), [weights](https://huggingface.co/facebook/tribev2))
to run fMRI experiments *in silico* on two related questions:

* **Terror Management Theory (TMT):** when people are reminded of their own death
  (*mortality salience*, MS), do they invest more in their cultural worldview and self-esteem?
* **Socioemotional Selectivity Theory (SST):** when the future feels *limited* (LT), do
  people shift from knowledge and achievement goals toward close relationships and
  savoring the present?

TRIBE v2 predicts the cortical fMRI response of an *average* listener (fsaverage5,
20,484 vertices, 1 Hz) to any audio, video or text. Its creators showed it reproduces
classic localizer contrasts zero-shot (faces, places, language, emotional vs physical
pain). This project applies the same approach to a social-psychology paradigm.

## Core idea: same probes, different primes

```
[prime ~35 s] ─12 s─ ( [neutral filler ~35 s] ─12 s─ )? [probe] ─8 s─ [probe] ─8 s─ … (12 probes)
     MS / PAIN / LT / ET / NEU          only in "delayed" trials       identical in every condition
```

Each trial starts with a guided-imagery **prime**. A set of **probe sentences** follows,
and those probes are identical in content, audio waveform and order across every
prime condition. So any difference in the predicted response to a probe comes from
the prime alone. "What does mortality salience induce?" becomes a measurable question:
**which kinds of content (worldview, self-worth, social bonds, savoring, achievement)
does the prime make the brain respond to differently, and in which networks?**

The TMT primes mirror the vector sets of the companion LLM steering study
(`Mortality_Steering_v7.ipynb`), so each brain-level contrast has an LLM-level twin:

| Prime | Content | LLM twin | Role |
|---|---|---|---|
| **MS** | imagine your own death | `v_self` | target: first-person finality |
| **MS_STRUCT** | irreversible endings of things/others (a felled tree, a dead star, an extinct bird) | `v_struct` | second target: impersonal finality |
| **PAIN** | dental pain | `neg_sensory` | canonical TMT control: aversive, not existential |
| **DISTRESS** | anxiety, shame, loneliness, anger: no ending | `neg_self` | first-person negative affect without finality |
| **NONFINAL** | attending to your own body/self as it changes: no ending | `self_nonfinal` | self focus without finality |
| **LT** / **ET** | the same situations with a short / long time horizon | — | Socioemotional Selectivity Theory |
| **NEU** | neutral routine | — | low-arousal baseline |

| Probe category | Prediction |
|---|---|
| **WV_NORM** in-group norm defense | TMT/LLM: MS increases the response |
| **WV_EXCL** out-group exclusion | LLM: no finality-specific effect (kept separate from WV_NORM: the axes behaved differently, even oppositely, in the LLM study) |
| **SELF** self-worth | TMT: MS increases the response (self-esteem buffer) |
| **SOC** close relationships | SST: LT increases the response |
| **SAV** present-moment savoring | SST: LT increases the response |
| **ACH** achievement and novelty | SST: LT decreases the response |
| **NEUT** neutral facts | control: removes generic, content-unspecific carry-over |

The **delay** factor (prime → probes immediately, or after a neutral filler) mirrors
TMT's distinction between proximal and distal defenses. In TRIBE terms, the filler
pushes the prime further back in the language model's context (Llama-3.2 sees the
preceding 1,024 words) and partly out of the model's 100-second transformer window.

## What gets measured

1. **Prime-evoked maps:** what each prime evokes while it is heard (e.g. MS − PAIN).
2. **Probe modulation:** A − B on identical probes, per category, region of interest (ROI) and delay.
3. **Category × prime interaction:** whether the shift is larger for category *c* than for neutral probes. This is the confirmatory test.
4. **Carry-over:** whether the probe-period shift resembles the prime's own pattern (a lingering
   state) or not (selective re-weighting of specific content).
5. **Decoding:** whether the preceding prime can be decoded from responses to probes that share
   no words with it.

ROIs come from the HCP-MMP parcellation (the atlas used in the paper): salience
(anterior insula/dACC), vmPFC, PCC/precuneus, TPJ, anterior temporal lobe, MTL,
dlPFC, OFC, and a language control region. See `mortality_tribe/rois.py`.

**Pre-registered hypotheses** (`configs/default.yaml`) are one-sided and Holm-corrected within each family:

| | Family | Prediction (immediate probes, vmPFC unless noted) | Source |
|---|---|---|---|
| H1 | TMT | MS − PAIN raises WV_NORM (vs NEUT) | LLM: `self_clean` steers in-group norm defense (+) |
| H2 | TMT | MS − NONFINAL raises WV_NORM | LLM: `self_nonfinal` null → finality is necessary |
| H3 | TMT | MS − DISTRESS raises WV_NORM | LLM: negative-affect controls moved the other way |
| H4 | TMT | MS − PAIN shifts WV_NORM more than WV_EXCL | LLM: exclusion axis not finality-specific |
| H5–H7 | SST | LT − ET: SOC ↑, SAV ↑ (OFC), ACH ↓ | Socioemotional Selectivity Theory |

Steering activates the representation directly, which is closest to the *immediate* condition, so H1–H4 use immediate probes.
The delayed condition is exploratory: the LoRA pilot suggests the sign may differ once mortality is no longer in focus.
**Convergence** of self and impersonal finality (MS−PAIN vs MS_STRUCT−PAIN) is exploratory. It is computed cross-half
(disjoint prime items), reported with split-half reliabilities, and compared against a shared-baseline reference
(W−PAIN for every other condition W). This mirrors the notebook's attenuation correction and calibration curve.

## Install and run

```bash
pip install -e ".[tribe,plot,test]"          # pulls tribev2 from GitHub; needs ffmpeg
huggingface-cli login                        # TRIBE uses Llama-3.2-3B (gated: accept its licence)

# 1) Offline check of the whole pipeline (no GPU or network): silent audio plus a
#    synthetic predictor with known planted effects. It must recover them.
python -m mortality_tribe all --dry-run
pytest

# 2) The real experiment (a GPU is strongly recommended)
python -m mortality_tribe design    # trial list        -> outputs/design.yaml
python -m mortality_tribe synth     # gTTS audio        -> outputs/audio/*.wav, timings.csv
python -m mortality_tribe predict   # TRIBE v2          -> outputs/preds/*.npz (resumable)
python -m mortality_tribe analyze   # stats + figures   -> outputs/results/
```

The default design has 8 primes × 4 items × 2 delays = 64 trials, about 4 hours of audio.
Start with `outputs/results/summary.md`. Next, check `figures/probe_timecourses.png`
to confirm the predicted responses look hemodynamically sensible.

## Layout

```
stimuli/            primes.yaml, probes.yaml, fillers.yaml (writing rules in each header)
configs/            default.yaml (design, timing, model, analysis, hypotheses)
mortality_tribe/
  stimuli.py        loading, validation, yoked/Latin-square design
  audio.py          per-segment TTS (cached, so probes are byte-identical), trial assembly, onsets
  predict.py        TribeModel wrapper + FakePredictor (planted effects, for dry runs)
  analysis.py       GLM betas, paired permutation tests, interaction, carry-over, decoding
  rois.py           hypothesis ROIs on HCP-MMP
  plotting.py       surface maps, ROI bars, onset-locked time courses
tests/              design invariants + end-to-end recovery of planted effects
```

## Design decisions found by the dry run

* **12 s gaps after the prime and filler.** With 4 s gaps, the prime's hemodynamic
  undershoot leaked into the first probes and produced spurious "carry-over".
* **GLM, not window averaging.** One HRF-convolved regressor per segment, plus temporal
  derivatives and drift terms, fitted within each trial. This follows the paper's GLM
  for language experiments and separates overlapping responses.
* **Stimuli are the random factor.** TRIBE output is deterministic, with no subject noise,
  so inference is over probe sentences and prime items. Probes from the same trial share
  an item-specific offset. For that reason, decoding cross-validates over held-out *prime
  items*, probe effects also report `p_fwe_items`, and the confirmatory tests use the
  within-trial category × prime interaction.

## What this can and cannot tell you (read before interpreting)

* TRIBE models a **passive listener's stimulus-driven response**, averaged over
  people. It does not model physiology, mood, motivation, or behavior. A result like
  "MS increases the vmPFC response to worldview sentences" means: *in the model's learned
  mapping, mortality context changes how worldview content is represented, in a way that
  maps onto vmPFC*. That is a **hypothesis generator** for a real fMRI or behavioral
  study, not evidence that people defend their worldview.
* In the model, the prime can only act through **context**: Llama's 1,024-word text
  context and the 100-second multimodal transformer window. Effects that need longer
  delays, sleep, or consolidation are out of reach by construction. The "delayed"
  condition shows how quickly the model's context effect decays.
* Text features come from Llama-3.2. Some "effects" may reflect the language model's
  semantics, for example death and nationalism co-occurring in text. A useful control
  is to compute the same contrasts directly on Llama embeddings and ask what the brain
  mapping adds.
* There are only 4 items per prime condition, which is enough for the probe-level design but weak
  for prime-level contrasts (the smallest exact p is 0.125). Add items to
  `stimuli/primes.yaml` (the tests enforce length matching) before drawing conclusions
  about the primes themselves.
* Stimuli are English (whisperX in tribev2 supports en/fr/es/nl/zh). A Korean version
  would need a new stimulus set and a check of tribev2's transcription step.
* TRIBE v2 is licensed CC-BY-NC-4.0 (non-commercial).

## Possible next steps

* **Steer the brain model's language stream directly.** TRIBE's text features come from base
  `meta-llama/Llama-3.2-3B`. Injecting the notebook's `v_self` into that Llama (same layer and ε) while TRIBE
  encodes *neutral* probes gives the predicted cortical footprint of the LLM's mortality direction. It also
  tests whether "steering" and "priming with mortality text" land in the same brain regions. Check first that the
  vector was extracted from the base model rather than Instruct, or re-extract it on base.
* Port the notebook's in-group-norm / out-group-exclusion items (translated) as probes, so both projects use the
  same outcome items.

* Validate: show that PAIN − NEU recovers the paper's emotional vs physical pain map before trusting MS − PAIN.
* Use video primes (e.g. end-of-life film clips vs dental-procedure clips); TRIBE's video stream is then used too.
* Compare the maps with NeuroSynth terms ("death", "self-referential", "fear"), as the paper does in its ICA section.
* Run the preregistered human fMRI or behavioral study using the contrasts that survived here.
