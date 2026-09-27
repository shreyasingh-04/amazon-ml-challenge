import pandas as pd
import os
import sys
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score
from features import compute_features
from pipeline import load_source_data

def build_dataset(data_dir, candidate_pairs_path, ground_truth_path):
    print("Loading raw data...")
    sources = load_source_data(data_dir)
    s1_dict = {row['entity_id']: row for _, row in sources['source1'].iterrows()} if sources['source1'] is not None else {}
    
    candidates_dict = {}
    if sources['source2'] is not None:
        for _, row in sources['source2'].iterrows():
            candidates_dict[row['entity_id']] = row
    if sources['source3'] is not None:
        for _, row in sources['source3'].iterrows():
            candidates_dict[row['entity_id']] = row
            
    print("Loading ground truth...")
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str)
    gt_df.fillna("", inplace=True)
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids']
        gt_map[s1_id] = set(matches.split(",")) if matches else set()
        
    print("Loading candidate pairs...")
    cand_df = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str)
    cand_df.fillna("", inplace=True)
    
    X_list = []
    y_list = []
    
    print("Computing features and extracting hard negatives...")
    for _, row in cand_df.iterrows():
        s1_id = row['source1_entity_id']
        cands_str = row['candidate_entity_ids']
        candidates = cands_str.split(",") if cands_str else []
        
        if s1_id not in s1_dict:
            continue
            
        row_s1 = s1_dict[s1_id]
        true_matches = gt_map.get(s1_id, set())
        
        for cand_id in candidates:
            if not cand_id or cand_id not in candidates_dict:
                continue
                
            row_s2 = candidates_dict[cand_id]
            features = compute_features(row_s1, row_s2)
            
            # Label: 1 if it's a true match, 0 if it's a hard negative
            label = 1 if cand_id in true_matches else 0
            
            X_list.append(features)
            y_list.append(label)
            
    X_df = pd.DataFrame(X_list)
    y_arr = np.array(y_list)
    return X_df, y_arr

def train_and_tune(X, y, output_model_path="models/matcher_xgb.pkl"):
    print(f"Training XGBoost on {len(y)} candidate pairs ({sum(y)} positives)...")
    
    # Handle massive class imbalance (e.g. 1M negatives, 50k positives)
    num_pos = sum(y)
    num_neg = len(y) - num_pos
    scale_weight = (num_neg / num_pos) if num_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=scale_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X, y)
    
    print("Tuning threshold for F0.5...")
    probs = model.predict_proba(X)[:, 1]
    
    best_f05 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.1, 0.95, 0.02):
        preds = (probs >= thresh).astype(int)
        
        # Avoid division by zero warnings
        if sum(preds) == 0:
            continue
            
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        
        if (0.25 * precision + recall) == 0:
            continue
            
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    # Final check to see what the model actually predicts at the best threshold
    final_preds = (probs >= best_thresh).astype(int)
    print(f"Best Threshold: {best_thresh:.2f} (F0.5: {best_f05:.4f})")
    print(f"Total Positives Predicted: {sum(final_preds)} out of {len(y)}")
    
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    with open(output_model_path, 'wb') as f:
        pickle.dump({'model': model, 'threshold': best_thresh}, f)
        
    print(f"Model saved to {output_model_path}")
    
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python train_matcher.py <data_dir> <candidate_pairs.tsv> <ground_truth.tsv> [output_model_path]")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    cand_path = sys.argv[2]
    gt_path = sys.argv[3]
    model_path = sys.argv[4] if len(sys.argv) > 4 else "models/matcher_xgb.pkl"
    
    # Fix paths if run from root
    if not os.path.exists(data_dir) and os.path.exists("../../" + data_dir):
        data_dir = "../../" + data_dir
        cand_path = "../../" + cand_path
        gt_path = "../../" + gt_path
        model_path = "../../" + model_path
        
    X, y = build_dataset(data_dir, cand_path, gt_path)
    if len(X) > 0:
        train_and_tune(X, y, model_path)
    else:
        print("Error: No candidate pairs generated to train on.")
