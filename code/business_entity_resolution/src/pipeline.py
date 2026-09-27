import os
import pandas as pd
from blocking import preprocess_dataframe, generate_candidate_pairs
from features import compute_features
from matcher import Matcher

def run_pipeline():
    DATASET_DIR = "dataset/test"
    print("1. Loading test data...")
    s1 = pd.read_csv(os.path.join(DATASET_DIR, 'test_source1.tsv'), sep='\t', dtype=str)
    s2 = pd.read_csv(os.path.join(DATASET_DIR, 'test_source2.tsv'), sep='\t', dtype=str)
    s3 = pd.read_csv(os.path.join(DATASET_DIR, 'test_source3.tsv'), sep='\t', dtype=str)
    
    print("2. Preprocessing...")
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    
    print("3. Generating candidate pairs (Blocking)...")
    candidates = generate_candidate_pairs(s1, s2, s3)
    
    os.makedirs("output", exist_ok=True)
    cand_path = "output/candidate_pairs.tsv"
    candidates.to_csv(cand_path, sep='\t', index=False)
    print(f"Saved candidates to {cand_path}")
    
    # Store dictionaries for O(1) fast lookup during matching
    # Done AFTER blocking so we don't hold them in memory alongside the massive cross-join tables!
    print("Building lookup dictionaries and freeing memory...")
    s1_dict = s1.set_index('entity_id').to_dict('index')
    s2_dict = s2.set_index('entity_id').to_dict('index')
    s3_dict = s3.set_index('entity_id').to_dict('index')
    
    # Delete original dataframes and force garbage collection
    del s1, s2, s3
    import gc
    gc.collect()
    
    print("4. Running Matching phase...")
    matcher = Matcher(threshold=0.83)
    
    results = []
    
    # Use itertuples for massive speedup over iterrows
    for row in candidates.itertuples(index=False):
        s1_id = row.source1_entity_id
        cands_str = str(row.candidate_entity_ids)
        
        # O(1) lookup instead of O(N) DataFrame filtering
        s1_row = s1_dict.get(s1_id)
        if not s1_row:
            results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ''})
            continue
            
        if not cands_str or cands_str == 'nan':
            results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ''})
            continue
            
        cand_list = cands_str.split(',')
        matched = []
        
        for cand_id in cand_list:
            if cand_id.startswith('S2-'):
                cand_row = s2_dict.get(cand_id)
            elif cand_id.startswith('S3-'):
                cand_row = s3_dict.get(cand_id)
            else:
                continue
                
            if not cand_row:
                continue
                
            feats = compute_features(s1_row, cand_row)
            if matcher.predict(feats):
                matched.append(cand_id)
                
        results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ','.join(matched)})
        
    res_df = pd.DataFrame(results)
    match_path = "output/matching_results.tsv"
    res_df.to_csv(match_path, sep='\t', index=False)
    print(f"Saved matches to {match_path}")

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == 'src':
        os.chdir('../../..')
    run_pipeline()
