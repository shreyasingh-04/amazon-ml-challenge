# Amazon ML Challenge 2026 - Business Entity Resolution

## Reproduction Instructions

To reproduce our results from the raw dataset, follow these exact steps:

1. **Environment Setup**
   Ensure you have Python 3.8+ installed. Install the pinned dependencies:
   ```bash
   pip install -r code/business_entity_resolution/requirements.txt
   ```

2. **Run Pipeline**
   From the root of the repository, execute the end-to-end pipeline:
   ```bash
   python code/business_entity_resolution/src/pipeline.py
   ```
   This script will automatically:
   - Read `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv` from the `dataset/test` directory.
   - Run our optimized blocking logic to output `candidate_pairs.tsv` to the `output/` directory.
   - Execute our Matcher across the generated candidates to produce the final `matching_results.tsv`.

3. **Validation**
   To independently verify our formatting, you can run the provided validator:
   ```bash
   python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```


 
