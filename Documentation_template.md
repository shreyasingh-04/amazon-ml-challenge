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
**Pairwise Similarity Metrics:**
We utilized `rapidfuzz` for highly optimized C++ string matching to generate features for candidate pairs.
- **Name Features:** Jaro-Winkler, Token Sort Ratio, Token Set Ratio, and length absolute difference. Token Set Ratio was particularly valuable for handling transposed multi-word DBAs.
- **Address Features:** Jaro-Winkler, Token Sort Ratio, Token Set Ratio.
- **Numeric Signals:** We extracted all numeric tokens from addresses (e.g., zip codes, street numbers). A partial or complete mismatch in numeric tokens acts as an extremely strong negative signal.
- **Country Enforcement:** A direct string match check on the country field to heavily penalize cross-country merges.

## 3. Matching Model
**Threshold-Based Heuristic Classifier:**
To favor the 2x precision weight of the F0.5 metric, we implemented a calibrated rule-based scoring system rather than a black-box ML model. 
- The final score is a weighted combination of the Name Score (65%) and Address Score (35%).
- Explicit penalties are applied: Mismatched numeric tokens in addresses subtract 0.35 from the address score. High name length difference combined with high token overlap subtracts 0.15 (to catch parent vs subsidiary entities).
- The final threshold is strictly set at `0.83` to prevent false merges.

## 4. Evaluation and Calibration
Since test labels were unavailable, we wrote a `split_validation.py` script to carve a deterministic 20% validation split from the training dataset. We optimized our Blocking recall ceiling against this set and tuned our `0.83` Matcher threshold to maximize the F0.5 metric on this local holdout.
