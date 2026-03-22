#!/usr/bin/env python3
"""
Silver → Gold conversion (universal for all datasets).

Input:  silver/{dataset}.txt (UserID ItemID Timestamp CumulativeCount)
        silver/{dataset}_item_features.txt (item features from train)
Output: gold/{dataset}/{dataset}-time.txt
        gold/{dataset}/item_features.txt (reindexed item features)

Output format for {dataset}-time.txt (5 columns, space-separated):
    UserID ItemID Timestamp CumulativeCount Age

Output format for item_features.txt:
    Line 1: T_min T_max T_cutoff
    Line 2+: NewItemID FirstTimestamp LastTimestamp NumInteractions

Applies:
    - Reindexing (1 to N)
    - Age computation using first_ts from item_features
    - Reindex item_features.txt for model consumption

The silver file already contains CumulativeCount computed from train data.
The item_features file contains first_ts computed from train data.
Age = timestamp - first_ts (for items seen in train, else 0).

Usage:
    python to_gold.py ml-1m
    python to_gold.py aws-beauty
    python to_gold.py aws-health
    python to_gold.py 30m
"""

import os
import argparse
from collections import defaultdict


def load_item_features(path):
    """Load pre-computed item features.

    File format:
        Line 1: T_min T_max T_cutoff
        Line 2+: ItemID FirstTimestamp LastTimestamp NumInteractions

    Returns:
        header: (t_min, t_max, t_cutoff)
        item_features: dict original_item_id -> (first_ts, last_ts, num_interactions)
    """
    item_features = {}
    with open(path, 'r') as f:
        # Parse header line
        header = f.readline().strip().split()
        t_min, t_max, t_cutoff = int(header[0]), int(header[1]), int(header[2])
        print(f"  Header: t_min={t_min}, t_max={t_max}, t_cutoff={t_cutoff}")
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                item_id = parts[0]
                first_ts = int(parts[1])
                last_ts = int(parts[2])
                num_interactions = int(parts[3])
                item_features[item_id] = (first_ts, last_ts, num_interactions)
    return (t_min, t_max, t_cutoff), item_features


def load_silver(path):
    """Load silver format: UserID ItemID Timestamp CumulativeCount"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                user_id = parts[0]
                item_id = parts[1]
                timestamp = int(parts[2])
                cum_count = int(parts[3])
                data.append((user_id, item_id, timestamp, cum_count))
    return data


def main():
    parser = argparse.ArgumentParser(description='Silver → Gold conversion')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., ml-1m, aws-beauty)')
    args = parser.parse_args()

    silver_path = os.path.join('silver', f'{args.dataset}.txt')
    silver_features_path = os.path.join('silver', f'{args.dataset}_item_features.txt')
    gold_dir = os.path.join('gold', args.dataset)
    gold_path = os.path.join(gold_dir, f'{args.dataset}-time.txt')
    gold_features_path = os.path.join(gold_dir, 'item_features.txt')

    if not os.path.exists(silver_path):
        raise FileNotFoundError(f"Silver data not found: {silver_path}")

    if not os.path.exists(silver_features_path):
        raise FileNotFoundError(f"Item features not found: {silver_features_path}")

    os.makedirs(gold_dir, exist_ok=True)

    # Load pre-computed item features
    print(f"Loading item features from {silver_features_path}...")
    (t_min, t_max, t_cutoff), item_features = load_item_features(silver_features_path)
    print(f"  Loaded features for {len(item_features):,} items")

    # Load silver data
    print(f"Loading {silver_path}...")
    data = load_silver(silver_path)
    print(f"  Loaded {len(data):,} interactions")

    # Sort by timestamp
    print("Sorting by timestamp...")
    data.sort(key=lambda x: x[2])

    # Reindex
    print("Reindexing...")
    user_map = {}
    item_map = {}

    for row in data:
        user_id, item_id = row[0], row[1]
        if user_id not in user_map:
            user_map[user_id] = len(user_map) + 1
        if item_id not in item_map:
            item_map[item_id] = len(item_map) + 1

    print(f"  Users: {len(user_map):,}, Items: {len(item_map):,}")

    # Build output with age
    print("Computing age...")
    output = []
    cold_items = 0

    for user_id, item_id, timestamp, cum_count in data:
        new_user = user_map[user_id]
        new_item = item_map[item_id]

        # Age = timestamp - first_ts (cold items get age=0)
        if item_id in item_features:
            first_ts = item_features[item_id][0]
            age = timestamp - first_ts
        else:
            age = 0  # Cold item
            cold_items += 1

        output.append((new_user, new_item, timestamp, cum_count, age))

    print(f"  Cold item interactions: {cold_items:,} (kept, age=0)")

    # Save
    print(f"Saving to {gold_path}...")
    with open(gold_path, 'w') as f:
        for user, item, ts, count, age in output:
            f.write(f"{user} {item} {ts} {count} {age}\n")

    # Save reindexed item features
    print(f"Saving item features to {gold_features_path}...")
    with open(gold_features_path, 'w') as f:
        f.write(f"{t_min} {t_max} {t_cutoff}\n")
        for orig_item_id, new_item_id in sorted(item_map.items(), key=lambda x: x[1]):
            if orig_item_id in item_features:
                first_ts, last_ts, num_interactions = item_features[orig_item_id]
                f.write(f"{new_item_id} {first_ts} {last_ts} {num_interactions}\n")

    # Stats
    n_users, n_items = len(user_map), len(item_map)
    density = len(output) / (n_users * n_items) * 100
    print(f"\n=== Gold Dataset Stats ===")
    print(f"Users: {n_users:,}")
    print(f"Items: {n_items:,}")
    print(f"Interactions: {len(output):,}")
    print(f"Density: {density:.4f}%")
    print(f"Avg interactions/user: {len(output)/n_users:.1f}")
    print(f"Data format: UserID ItemID Timestamp CumulativeCount Age")
    print(f"\nSaved to:")
    print(f"  {gold_path}")
    print(f"  {gold_features_path}")


if __name__ == '__main__':
    main()
