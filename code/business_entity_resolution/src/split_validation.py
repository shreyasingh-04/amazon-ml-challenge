import pandas as pd
import os
import random
import shutil

def split_validation(data_dir="dataset/train", output_dir="dataset/split", split_ratio=0.2, seed=42):
    """
    Carves out a validation set from the training data by randomly selecting
    a subset of Source 1 entities and isolating their ground truth.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    
    if not os.path.exists(gt_path):
        print(f"Ground truth not found at {gt_path}.")
        return

    print(f"Loading data from {data_dir}...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    
    s1_ids = s1_df['entity_id'].unique().tolist()
    random.seed(seed)
    random.shuffle(s1_ids)
    
    val_size = int(len(s1_ids) * split_ratio)
    val_s1_ids = set(s1_ids[:val_size])
    train_s1_ids = set(s1_ids[val_size:])
    
    # Split S1
    val_s1_df = s1_df[s1_df['entity_id'].isin(val_s1_ids)]
    train_s1_df = s1_df[s1_df['entity_id'].isin(train_s1_ids)]
    
    # Split Ground Truth
    val_gt_df = gt_df[gt_df['source1_entity_id'].isin(val_s1_ids)]
    train_gt_df = gt_df[gt_df['source1_entity_id'].isin(train_s1_ids)]
    
    # Save splits
    val_s1_df.to_csv(os.path.join(output_dir, "val", "train_source1.tsv"), sep="\t", index=False)
    val_gt_df.to_csv(os.path.join(output_dir, "val", "train_ground_truth.tsv"), sep="\t", index=False)
    
    train_s1_df.to_csv(os.path.join(output_dir, "train", "train_source1.tsv"), sep="\t", index=False)
    train_gt_df.to_csv(os.path.join(output_dir, "train", "train_ground_truth.tsv"), sep="\t", index=False)
    
    # Source 2 and Source 3 are shared dictionaries, we can just copy them to both splits
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        src_path = os.path.join(data_dir, s_file)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(output_dir, "train", s_file))
            shutil.copy(src_path, os.path.join(output_dir, "val", s_file))

    print(f"Validation Split complete (Seed {seed})!")
    print(f"Validation Set S1 entities: {len(val_s1_df)}")
    print(f"Train Sub-split S1 entities: {len(train_s1_df)}")
    print(f"Splits saved to: {output_dir}")

if __name__ == "__main__":
    split_validation()
