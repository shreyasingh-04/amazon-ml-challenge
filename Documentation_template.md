# Amazon ML Challenge 2026 - Entity Resolution Methodology

## 1. Blocking Strategy
**Objective:** Maximize recall while strictly bounding candidate set sizes to avoid OOM errors and maximize final ranking score.

**Normalization:**
To handle cross-country variance robustly (and elegantly scale to unseen countries like France), we strictly avoided hardcoded country branches. Both business names and addresses were normalized using data-driven dictionary lookups.
- Name Normalization: Punctuation removal, expansion of symbols (`&` to `and`), and standardization of corporate suffixes (e.g., `corp` -> `corporation`, `ltd` -> `limited`).
- Address Normalization: Expansion of abbreviations (`st`, `ave`, `hwy`) and regex-based stripping of localized landmark phrases (e.g., `Near SBI ATM`, `Opposite...`).

**Blocking Keys:**
We utilized a multi-key strategy for high recall against typos and acronyms. 
1. **Significant Tokens**: Every token in the normalized name longer than 2 characters acts as a key.
2. **Acronyms**: The first character of every token is concatenated to catch initialized DBA variations.
3. **Longest Word Prefix**: To tolerate typos, the first 4 characters of the longest name token forms a key.

**Candidate Capping (OOM Prevention):**
Due to the `N x M` explosion on common tokens (e.g., "company"), we applied frequency capping. Any blocking key bucket containing more than 500 entities was dynamically discarded. This successfully kept our average candidate set size under 400 per Source 1 entity without sacrificing rare-token recall.## 2. Feature Engineering

## 3. Matching Model

## 4. Evaluation and Calibration
