import pandas as pd
import re
import os
from typing import Dict, Tuple

def load_and_verify_data(dataset_dir: str, prefix: str = 'train') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load source1, source2, source3 and ground truth TSV files, and verify the schema matches the spec.
    """
    s1_path = os.path.join(dataset_dir, prefix, f'{prefix}_source1.tsv')
    s2_path = os.path.join(dataset_dir, prefix, f'{prefix}_source2.tsv')
    s3_path = os.path.join(dataset_dir, prefix, f'{prefix}_source3.tsv')
    
    print(f"Loading data from {dataset_dir}/{prefix}...")
    # Use nrows=10 for fast schema verification if needed, or read full.
    # For now, we read all data to do the end-to-end audit.
    s1 = pd.read_csv(s1_path, sep='\t')
    s2 = pd.read_csv(s2_path, sep='\t')
    s3 = pd.read_csv(s3_path, sep='\t')
    
    expected_cols = {'entity_id', 'business_name', 'business_address', 'country'}
    for name, df in zip(['Source1', 'Source2', 'Source3'], [s1, s2, s3]):
        actual_cols = set(df.columns)
        print(f"--- {name} Schema ---")
        print(f"Columns: {df.columns.tolist()}")
        if not expected_cols.issubset(actual_cols):
            print(f"WARNING: Missing expected columns in {name}! Missing: {expected_cols - actual_cols}")
        else:
            print("Schema matches exact spec.")
            
    gt = None
    if prefix == 'train':
        gt_path = os.path.join(dataset_dir, prefix, f'{prefix}_ground_truth.tsv')
        gt = pd.read_csv(gt_path, sep='\t')
        print(f"\n--- Ground Truth Schema ---")
        print(f"Columns: {gt.columns.tolist()}")
        
    return s1, s2, s3, gt

def apply_data_driven_lookups(text: str, lookups: Dict[str, str]) -> str:
    """Apply dictionary-based standardizations using regex boundaries."""
    for pattern, replacement in lookups.items():
        text = re.sub(rf'\b{pattern}\b', replacement, text)
    return text

def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = name.replace('&', ' and ')
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Standardize legal-suffix abbreviations
    lookups = {
        'corp': 'corporation',
        'inc': 'incorporated',
        'pvt': 'private',
        'ltd': 'limited',
        'llc': 'limited liability company',
        'co': 'company',
        'bros': 'brothers'
    }
    name = apply_data_driven_lookups(name, lookups)
    
    # Extra whitespace removal
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_address(address: str) -> str:
    if pd.isna(address):
        return ""
    address = str(address).lower()
    address = address.replace('&', ' and ')
    
    # Strip landmark phrases ("Near SBI ATM", "Opposite", etc.)
    address = re.sub(r'\bnear\s+.*?(?:,|$)', ' ', address)
    address = re.sub(r'\bopp(?:osite|\.)?\s+.*?(?:,|$)', ' ', address)
    address = re.sub(r'\bbeside\s+.*?(?:,|$)', ' ', address)
    address = re.sub(r'\bbehind\s+.*?(?:,|$)', ' ', address)
    
    address = re.sub(r'[^\w\s]', ' ', address)
    
    # Address abbreviations
    lookups = {
        'rd': 'road',
        'st': 'street',
        'ave': 'avenue',
        'blvd': 'boulevard',
        'apt': 'apartment',
        'ste': 'suite',
        'bldg': 'building',
        'fl': 'floor',
        'hwy': 'highway'
    }
    address = apply_data_driven_lookups(address, lookups)
    
    address = re.sub(r'\s+', ' ', address).strip()
    return address

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns. Country is not used here to avoid hardcoded rules."""
    df['norm_name'] = df['business_name'].apply(normalize_name)
    df['norm_address'] = df['business_address'].apply(normalize_address)
    return df

def extract_blocking_keys(row: pd.Series) -> set:
    keys = set()
    norm_name = row['norm_name']
    norm_address = row['norm_address']
    
    name_tokens = norm_name.split()
    addr_tokens = norm_address.split()
    
    # Filter out very short tokens (like 'and', 'co') for blocking
    sig_name_tokens = [t for t in name_tokens if len(t) > 2]
    
    # Strategy 1: Every significant token in the name is a key
    for t in sig_name_tokens:
        keys.add(f"TK_{t}")
        
    # Strategy 2: First 4 characters of the longest word in the name
    if sig_name_tokens:
        longest_name_token = max(sig_name_tokens, key=len)
        if len(longest_name_token) >= 4:
            keys.add(f"L4_{longest_name_token[:4]}")
            
    # Strategy 3: Initials of the name (to catch acronyms)
    if name_tokens:
        initials = "".join([t[0] for t in name_tokens if t])
        if len(initials) >= 3:
            keys.add(f"IN_{initials}")

    return keys

def generate_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Generate multiple blocking keys per entity."""
    records = []
    for _, row in df.iterrows():
        keys = extract_blocking_keys(row)
        for k in keys:
            records.append({'entity_id': row['entity_id'], 'blocking_key': k})
    return pd.DataFrame(records)

def generate_candidate_pairs(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """
    Block and generate candidate pairs. 
    Returns DataFrame matching candidate_pairs.tsv schema.
    """
    # Generate keys
    keys1 = generate_blocking_keys(s1)
    keys2 = generate_blocking_keys(s2)
    keys3 = generate_blocking_keys(s3)
    
    # Merge candidates on blocking keys
    cands_1_2 = pd.merge(keys1, keys2, on='blocking_key', suffixes=('_1', '_cand'))
    cands_1_3 = pd.merge(keys1, keys3, on='blocking_key', suffixes=('_1', '_cand'))
    all_cands = pd.concat([cands_1_2, cands_1_3])
    
    # Group by source1_entity_id to get unique candidate lists
    grouped = all_cands.groupby('entity_id_1')['entity_id_cand'].unique().reset_index()
    grouped.columns = ['source1_entity_id', 'candidate_entity_ids']
    
    # Ensure no self-matches (though shouldn't happen cross-source) and format properly
    grouped['candidate_entity_ids'] = grouped['candidate_entity_ids'].apply(lambda x: ','.join(sorted(list(set(x)))))
    
    # Ensure all S1 entities exist in output (even singletons)
    res = pd.DataFrame({'source1_entity_id': s1['entity_id']})
    res = pd.merge(res, grouped, on='source1_entity_id', how='left')
    res['candidate_entity_ids'] = res['candidate_entity_ids'].fillna('')
    
    return res

def evaluate_blocking_recall(cands_df: pd.DataFrame, gt_df: pd.DataFrame):
    """
    Audit blocking recall and candidate size.
    """
    gt_dict = dict(zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']))
    
    total_true_matches = 0
    surviving_matches = 0
    candidate_sizes = []
    
    for _, row in cands_df.iterrows():
        s1_id = row['source1_entity_id']
        cands_str = str(row['candidate_entity_ids'])
        cand_list = set(cands_str.split(',')) if cands_str else set()
        
        candidate_sizes.append(len(cand_list))
        
        gt_str = str(gt_dict.get(s1_id, ''))
        gt_list = set(gt_str.split(',')) if gt_str and gt_str != 'nan' else set()
        
        total_true_matches += len(gt_list)
        surviving_matches += len(gt_list.intersection(cand_list))
        
    recall = surviving_matches / total_true_matches if total_true_matches > 0 else 1.0
    avg_size = sum(candidate_sizes) / len(candidate_sizes) if candidate_sizes else 0
    
    print(f"\n--- Blocking Audit ---")
    print(f"Total True Matches: {total_true_matches}")
    print(f"Surviving Matches:  {surviving_matches}")
    print(f"Recall Ceiling:     {recall:.4f} ({recall*100:.2f}%)")
    print(f"Avg Candidate Size: {avg_size:.2f} per S1 entity")

if __name__ == "__main__":
    DATASET_DIR = "dataset"
    print("1. Loading and verifying schema...")
    # NOTE: In production, load the full dataset. For rapid testing we could use nrows.
    # To do the full recall audit, we need the full dataset.
    # s1, s2, s3, gt = load_and_verify_data(DATASET_DIR, prefix='train')
    
    # We will run a 50,000 row test to establish a baseline recall ceiling
    s1_test = pd.read_csv(os.path.join(DATASET_DIR, 'train', 'train_source1.tsv'), sep='\t', nrows=50000)
    s2_test = pd.read_csv(os.path.join(DATASET_DIR, 'train', 'train_source2.tsv'), sep='\t', nrows=50000)
    s3_test = pd.read_csv(os.path.join(DATASET_DIR, 'train', 'train_source3.tsv'), sep='\t', nrows=50000)
    gt_test = pd.read_csv(os.path.join(DATASET_DIR, 'train', 'train_ground_truth.tsv'), sep='\t', nrows=50000)
    
    print("S1 schema:", s1_test.columns.tolist())
    print("S2 schema:", s2_test.columns.tolist())
    print("S3 schema:", s3_test.columns.tolist())
    print("GT schema:", gt_test.columns.tolist())
    
    print("\nPre-processing (Normalization)...")
    s1_test = preprocess_dataframe(s1_test)
    s2_test = preprocess_dataframe(s2_test)
    s3_test = preprocess_dataframe(s3_test)
    
    print("\nGenerating candidates...")
    candidates = generate_candidate_pairs(s1_test, s2_test, s3_test)
    print("\nEvaluating Blocking Strategy...")
    evaluate_blocking_recall(candidates, gt_test)
    
    print("\nPipeline check completed successfully.")
