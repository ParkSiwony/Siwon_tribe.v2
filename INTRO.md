# Mortality Salience × TRIBE v2: Brief Introduction
*필멸성·시간 제한 현저성은 무엇을 유도하는가? — TRIBE v2 기반 in-silico 뇌 실험*

## One-sentence summary
We use Meta's TRIBE v2, a model that predicts whole-cortex fMRI responses to speech, text and video, to ask **how reminders of death or of limited time change the way the brain processes the same sentences afterwards**.

## Why
* **Terror Management Theory (TMT)** predicts that mortality salience strengthens attachment to one's cultural worldview and self-esteem.
* **Socioemotional Selectivity Theory (SST)** predicts that a limited future shifts priorities toward close relationships and the present moment, away from knowledge and achievement.
* Both are behavioral theories. TRIBE v2 makes it possible to pilot their *neural* predictions cheaply, before a costly human fMRI study.

## How (in one picture)
```
[ prime ]  ──►  ( neutral filler )?  ──►  [ probe sentences: identical in every condition ]
 MS (self finality) · MS_STRUCT (impersonal     in-group norms · out-group exclusion · self-worth ·
 finality) · dental pain · distress · non-final close relationships · savoring · achievement ·
 self focus · limited / expansive time · neutral  neutral facts
```
Probe content, audio and order are held constant, so any change in the predicted brain response comes from the prime alone.

## What we measure
| Question | Analysis |
|---|---|
| What does the prime itself evoke? | Prime contrast maps (e.g. MS − dental pain) |
| Which content does it re-weight? | Category × prime interaction on identical probes, in 9 hypothesis ROIs |
| A lingering mood, or selective re-weighting? | Spatial carry-over of the prime pattern into probes |
| Is the prime still "readable" later? | Decoding the prime from probe responses alone |

Seven pre-registered hypotheses are tested with Holm correction within each family. **H1–H4** take their predictions from the LLM steering study:
* first-person finality raises in-group *norm* processing (H1);
* the effect needs finality, not just self focus (H2);
* it isn't negative affect (H3);
* it's specific to the norm axis rather than exclusion (H4).

**H5–H7** test Socioemotional Selectivity Theory. Everything else is exploratory.

## Status
* The full pipeline is implemented: stimuli, TTS, TRIBE inference, GLM, permutation statistics and figures.
* An offline dry run with planted effects recovers the true effect and rejects the nulls (36 tests pass, including a CPU dry run of the pilot notebook).
* **Pilot notebook** (`notebooks/pilot_image_mortality_steering.ipynb`):
  - shows mortality and limited-time *images* to TRIBE and compares the predicted brain contrasts;
  - converts each contrast into a Llama-3.2-3B steering direction through TRIBE's text pathway;
  - steers Llama on the in-group-norm / out-group-exclusion items and on open questions about humanity.
* **Pending:** running the real model. This needs a GPU and access to Llama-3.2-3B.

## Relation to the "AI mortality" LLM project
A companion project studies mortality inside a language model: LoRA training and activation steering of a *mortality representation* in **Llama-3.2-3B**, measured on in-group-norm vs out-group-exclusion defense items. TRIBE v2 builds its text features from **the same Llama-3.2-3B**. Together the two projects ask one question at two levels:

* **LLM level:** what does a mortality representation do to the model's defensive behavior?
* **Brain level (this repo):** what does mortality content do to predicted human cortical responses?

This opens a shared measurement space, for example comparing the LLM's mortality steering direction with the Llama features that drive TRIBE's predicted MS effect.

## Caveats
TRIBE predicts a passive, averaged listener's stimulus-driven response. It does not model mood, motivation or behavior. Results are **hypotheses for human studies**, not evidence about people.

See `README.md` for design details, commands and limitations.
