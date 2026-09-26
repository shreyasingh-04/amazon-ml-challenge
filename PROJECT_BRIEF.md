# Amazon ML Challenge 2026 — Business Entity Resolution
_(Updated against the official problem statement — supersedes the deck-only version)_

## 1. Problem statement

We receive business records from **three independent sources** (Source 1, Source 2,
Source 3). Each record has: `entity_id` (prefixed `S1-`/`S2-`/`S3-`), `business_name`,
`business_address`, `country`.

Source 1 is the **deduplicated reference source**. For every Source 1 entity we must
find all matching records in Source 2 and Source 3 — zero, one, or many.

No shared identifier exists across sources. Matching must be done from name +
address text alone, robust to: abbreviations (Corp/Corporation, Pvt/Private,
Ltd/Limited, Rd/Road, St/Street), legal suffix inconsistencies, DBA/trade names,
punctuation (`&` vs "and"), word-order transpositions, typos, transliteration
variants, missing address components, landmark references ("Near SBI ATM"),
municipal numbering formats, component reordering.

**`country`**: training covers US and India; **test additionally includes France**,
unseen in training. Treat country as an open string set — do not hardcode, filter,
or one-hot-encode against only {US, India}. Every test entity, France included,
must appear in the submission.

**External data lookups, APIs, geocoding services are strictly prohibited.**
**Final model must be MIT/Apache 2.0 licensed and ≤8B parameters.**

## 2. Pipeline shape — two stages, BOTH now scored

1. **Blocking / candidate generation** — cuts the search space to a small candidate
   set per Source 1 entity. Output: `candidate_pairs.tsv`.
   - **This is no longer just an internal recall-safety step — it counts toward
     final ranking directly.** Reviewers look at `candidate_pairs.tsv` and the
     code producing it. Smaller candidate sets per Source 1 entity rank higher,
     all else equal. The objective is **recall vs. candidate-set-size trade-off**,
     not "grab everything plausible."
   - `candidate_pairs.tsv` must be the **exact set fed to the matcher at
     inference** — the last filtering stage, not an early loose pass. Every ID in
     `matching_results.tsv` must appear in `candidate_pairs.tsv` (validator warns
     if not — signals a pipeline bug).
2. **Matching** — for each candidate, decide true match vs. look-alike; aggregate
   to one row per Source 1 entity in `matching_results.tsv`.

## 3. Data & exact file schema

All files tab-separated (`.tsv`); always `sep="\t"`.

**Source files** (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`):
| column | notes |
|---|---|
| `entity_id` | unique, prefixed `S1-`/`S2-`/`S3-` |
| `business_name` | noisy: abbreviations, typos, transliterations |
| `business_address` | noisy: partial, format variations, landmarks |
| `country` | open string set — US, India in train; +France in test |

**`train_ground_truth.tsv`**:
| column | notes |
|---|---|
| `source1_entity_id` | |
| `matched_entity_ids` | comma-separated S2-/S3- ids; empty = singleton |

No validation split is provided for test — carve our own hold-out from train to
self-score with F₀.₅ before submitting.

## 4. Output format (exact)

**`matching_results.tsv`** — only file scored on the live leaderboard.
| column | notes |
|---|---|
| `source1_entity_id` | |
| `matched_entity_ids` | comma-separated, no quoting, empty if none |

**`candidate_pairs.tsv`** — not leaderboard-scored, but reviewed for final ranking.
| column | notes |
|---|---|
| `source1_entity_id` | |
| `candidate_entity_ids` | comma-separated candidates fed to the matcher |

Hard rules (validator enforces):
- Exactly one row per Source 1 **test** entity in both files — no missing rows.
- Empty string when no matches/candidates (not "NA", not omitted row).
- No duplicate entity IDs within one row's list; no duplicate `source1_entity_id` rows.
- IDs must reference existing Source 2/3 **test** records only; no self-matches to Source 1.
- `matching_results.tsv` ids must be a subset of that row's `candidate_pairs.tsv` ids.

Run before every submission:
```
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## 5. Scoring

Macro-averaged **F₀.₅** per Source 1 entity, then averaged over all entities:
```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```
- Precision weighted 2× over recall — a false merge costs far more than a miss.
- Singletons count fully: correct empty prediction = 1.0, any predicted match on
  a true singleton = 0.0.
- Public leaderboard uses a subset of test during the challenge; private
  leaderboard (full remaining test set) decides final rank after the challenge ends.
- **Final ranking also factors in `candidate_pairs.tsv` quality** (recall ceiling
  it preserves vs. how small it is) — reviewed separately from the leaderboard score.

## 6. Repository / final package structure (matches official spec exactly)

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/                 # all source code
│       ├── README.md            # exact reproduce instructions: data -> blocking -> matching -> output
│       └── requirements.txt     # pinned deps
└── Documentation_template.md    # methodology: approach, blocking strategy,
                                  # model architecture, feature engineering
```

Working repo (before zipping) should mirror this, plus a local `data/` and
`dataset/` layout matching the download, and a `utils/validate_submission.py`.

## 7. Task split

### Person A — Blocking / Candidate Generation (this is my part)
Objective: maximize recall **per unit of candidate-set size** — this is scored
directly, not just an internal safety margin.

- [ ] Load `train_source1/2/3.tsv` + `train_ground_truth.tsv`; confirm schema
      matches spec exactly (don't assume — print and check).
- [ ] Build normalization for `business_name` and `business_address` separately:
      lowercase, strip punctuation, expand `&`, standardize legal-suffix
      abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited...) and address
      abbreviations (Rd/Road, St/Street...), strip/standardize landmark phrases.
      Keep normalized fields as new columns; never discard originals.
- [ ] Do **not** build country-specific hardcoded branches — write normalization
      rules as data-driven lookups so a new country (France) degrades gracefully
      rather than crashing or being silently dropped.
- [ ] Design blocking key(s): token-based (sorted significant tokens), phonetic
      (Soundex/Metaphone) on name, address-token key, optionally n-gram/character
      shingles for typo tolerance. Union multiple strategies if needed for recall,
      but now weigh each added strategy against how much it inflates candidate size.
- [ ] Generate `candidate_pairs.tsv` in the **exact schema** above — one row per
      Source 1 test entity, `candidate_entity_ids` comma-separated, empty when none.
- [ ] Build a **recall-ceiling audit** against `train_ground_truth.tsv`: % of true
      matches surviving blocking, AND average/median candidate-set size per
      entity — report both, they're now a single trade-off to optimize together.
- [ ] Iterate: tighten blocking keys / add filters that cut candidate-set size
      with minimal recall loss (this is the scored lever, not just a nice-to-have).
- [ ] Confirm with teammate: `candidate_pairs.tsv` is the FINAL filtered set
      actually used for inference, not an early loose pass — if there are
      multiple blocking/filtering stages, this file must reflect the last one.
- [ ] Write the "candidate generation / blocking strategy" section of
      `Documentation_template.md`.

### Person B — Matching, Aggregation & Evaluation
- [ ] Build pairwise similarity features per candidate (name and address
      similarity computed separately): Jaccard, Levenshtein, TF-IDF cosine, etc.
- [ ] Build match/no-match decision (rule-based threshold or trained classifier,
      MIT/Apache-licensed, ≤8B params if using any pretrained model), tuned
      toward precision given the F₀.₅ penalty on false merges.
- [ ] Explicit empty-list path for singletons — precision-first, prefer no-match
      when unsure.
- [ ] Aggregate into `matching_results.tsv` exactly as spec'd; every id must be a
      subset of that row's candidates in `candidate_pairs.tsv`.
- [ ] Build `evaluate.py`: macro F₀.₅ scorer against a self-carved hold-out split
      from train (no test ground truth is provided).
- [ ] Build/adapt `utils/validate_submission.py` per the exact rules in section 4.
- [ ] Write the "model architecture / feature engineering" section of
      `Documentation_template.md`.

### Shared
- [ ] Agree on `candidate_pairs.tsv` schema/contract before writing code (done —
      see section 4 above, now fixed by the official spec).
- [ ] Agree on the repo layout in section 6 up front so both branches merge cleanly.
- [ ] Decide the train/validation hold-out split together so both of you evaluate
      against the same numbers.
- [ ] Joint pass on `Documentation_template.md`, `README.md`, `requirements.txt`,
      and final `validate_submission.py` run before packaging the zip.
