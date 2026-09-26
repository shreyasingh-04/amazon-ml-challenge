import pandas as pd
import os
import sys
from features import compute_features
from matcher import Matcher
from pipeline import load_source_data, save_tsv

def run_matcher_pipeline(data_dir, candidate_pairs_path, output_dir):
    """
    Decoupled execution: reads the candidate_pairs.tsv contract from Person A's blocking output
    and evaluates each pair using the Matcher model.
    """
    print(f"Loading raw data from {data_dir}...")
    sources = load_source_data(data_dir)
    
    s1_df = sources['source1']
    s2_df = sources['source2']
    s3_df = sources['source3']
    
    if s1_df is None:
        print("Source 1 data not found. Cannot proceed.")
        return
        
    print(f"Loading candidate pairs from {candidate_pairs_path}...")
    try:
        cand_df = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str)
        cand_df.fillna("", inplace=True)
    except Exception as e:
        print(f"Error reading candidate pairs: {e}")
        return
        
    candidate_pairs = {}
    for _, row in cand_df.iterrows():
        s1_id = row['source1_entity_id']
        cands = row['candidate_entity_ids']
        candidate_pairs[s1_id] = cands.split(",") if cands else []
        
    print("Running matching model...")
    matcher = Matcher()
    matching_results = {}
    
    # Pre-index S2 and S3 for fast lookup
    candidates_dict = {}
    if s2_df is not None:
        for _, row in s2_df.iterrows():
            candidates_dict[row['entity_id']] = row
    if s3_df is not None:
        for _, row in s3_df.iterrows():
            candidates_dict[row['entity_id']] = row
            
    # Pre-index S1
    s1_dict = {row['entity_id']: row for _, row in s1_df.iterrows()}
    
    # Evaluate candidates
    for s1_id, candidates in candidate_pairs.items():
        matched = []
        if s1_id not in s1_dict:
            continue
            
        row_s1 = s1_dict[s1_id]
        
        for cand_id in candidates:
            if not cand_id or cand_id not in candidates_dict:
                continue
            row_s2 = candidates_dict[cand_id]
            features = compute_features(row_s1, row_s2)
            if matcher.predict(features):
                matched.append(cand_id)
                
        matching_results[s1_id] = matched
        
    match_path = os.path.join(output_dir, "matching_results.tsv")
    save_tsv(matching_results, match_path, "source1_entity_id", "matched_entity_ids")
    print(f"Matching results saved to {match_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_matcher.py <data_dir> <candidate_pairs.tsv_path> [output_dir]")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    candidate_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
    
    # Fix relative paths
    if not os.path.exists(data_dir) and os.path.exists(os.path.join("../..", data_dir)):
        data_dir = os.path.join("../..", data_dir)
        candidate_path = os.path.join("../..", candidate_path)
        output_dir = os.path.join("../..", output_dir)
        
    run_matcher_pipeline(data_dir, candidate_path, output_dir)
