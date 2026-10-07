# Steering sentences (ChatGPT + Kimi)

English contrast pairs and defense items from the LLM steering study, generated with ChatGPT and
Kimi, with Korean versions alongside. The pilot notebook
(`notebooks/pilot_image_mortality_steering.ipynb`) reads them as follows.

| File | Used for |
|---|---|
| `mort_self_pairs.jsonl` | **A3** text → brain (`finality|text` in the gate) and **C3** CAA target `v_mort_self` |
| `mort_struct_pairs.jsonl` | A3 (shown in the matrix) and C3 target `v_mort_struct` |
| `neg_sensory_pairs.jsonl` | A3 (`negative|text` in the gate) and C3 orthogonalisation basis |
| `neg_affect_pairs.jsonl` | A3 (matrix) and C3 orthogonalisation basis |
| `def_pairs.jsonl` | C3 orthogonalisation bases `def_norm` / `def_excl` (split by `axis`) |
| `self_nonfinal_pairs.jsonl`, `neg_self_pairs.jsonl`, `arb_pairs.jsonl` | C3 discriminant controls; `arb` is also a steering control |
| `items.jsonl` | **D** the 408 in-group-norm / out-group-exclusion items scored under steering |
| `*_kor.jsonl` | not read by default (`caa.load_pairs(..., include_korean=True)`) |
| `control.jsonl`, `emotion_only*.jsonl` | not read: chat-format training corpora, not contrast pairs |

Pair fields follow the study (`context`, `question`, `opt_pos`, `opt_neg`, `family`, `gen`,
`person`, `axis`, `exploratory`, …). The notebook prints a text check before using them:
- counts per generator and family;
- option lengths;
- duplicates;
- first-person words in the options of impersonal sets;
- finality words in sets that should have none.

Notes from the first check:
- **Mortality sets.** In `mort_self` and `mort_struct`, `opt_pos` is about one word shorter than `opt_neg`. The two sets match each other in this, and negation words are balanced (~50 % in both options).
- **`neg_sensory` person label.** Its options are labelled `third` but are mostly first-person bodily statements ("…churns my stomach").
