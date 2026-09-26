import pandas as pd
import os
import sys

def evaluate(ground_truth_path, predictions_path):
    """
    Computes macro F0.5 per Source 1 entity against the ground truth.
    Formula: F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str)
    gt_df.fillna("", inplace=True)
    
    pred_df = pd.read_csv(predictions_path, sep="\t", dtype=str)
    pred_df.fillna("", inplace=True)
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids']
        gt_dict[s1_id] = set(matches.split(",")) if matches else set()
        
    pred_dict = {}
    for _, row in pred_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids']
        pred_dict[s1_id] = set(matches.split(",")) if matches else set()
        
    f05_scores = []
    for s1_id, true_matches in gt_dict.items():
        pred_matches = pred_dict.get(s1_id, set())
        
        # Singleton case: true_matches is empty
        if not true_matches:
            if not pred_matches:
                f05_scores.append(1.0) # correctly predicted singleton
            else:
                f05_scores.append(0.0) # falsely predicted match for singleton
            continue
            
        # Normal case
        true_positives = len(true_matches.intersection(pred_matches))
        if true_positives == 0:
            f05_scores.append(0.0)
            continue
            
        precision = true_positives / len(pred_matches)
        recall = true_positives / len(true_matches)
        
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        f05_scores.append(f05)
        
    macro_f05 = sum(f05_scores) / len(f05_scores) if f05_scores else 0.0
    print(f"Evaluated on {len(f05_scores)} Source 1 entities.")
    print(f"Macro F0.5 Score: {macro_f05:.4f}")
    return macro_f05

if __name__ == "__main__":
    import sys
    gt = "dataset/train/train_ground_truth.tsv"
    pred = "output/matching_results.tsv"
    
    if len(sys.argv) > 1:
        gt = sys.argv[1]
    if len(sys.argv) > 2:
        pred = sys.argv[2]
        
    if not os.path.exists(gt) and os.path.exists("../" + gt):
        gt = "../" + gt
        pred = "../" + pred
        
    if not os.path.exists(gt):
        print(f"Ground truth file {gt} not found.")
        sys.exit(1)
        
    if not os.path.exists(pred):
        print(f"Predictions file {pred} not found.")
        sys.exit(1)
        
    evaluate(gt, pred)
