#!/usr/bin/env python3
"""
Bronze → Silver conversion for MovieLens 1M dataset.

Input:  bronze/ml-1m/ratings.dat
Output: silver/ml-1m.txt, silver/ml-1m_item_features.txt

Output format for ml-1m.txt (4 columns, space-separated):
    UserID ItemID Timestamp CumulativeCount

Output format for ml-1m_item_features.txt:
    Line 1: T_min T_max T_train_cutoff
    Line 2+: ItemID FirstTimestamp LastTimestamp NumInteractions

Flow: Load -> Sort -> Split (90% train) -> Compute item features from train
(No time filter, no user core - ML-1M is already clean)

Usage:
    python to_silver_ml1m.py
"""

import os
from collections import defaultdict


TRAIN_RATIO = 0.90


def save_item_features(features, t_min, t_max, t_cutoff, output_path):
    """Save item features to file."""
    with open(output_path, 'w') as f:
        f.write(f"{t_min} {t_max} {t_cutoff}\n")
        for item_id in sorted(features.keys()):
            first_ts, last_ts, num_interactions = features[item_id]
            f.write(f"{item_id} {first_ts} {last_ts} {num_interactions}\n")


def main():
    bronze_path = os.path.join('bronze', 'ml-1m', 'ratings.dat')
    silver_dir = 'silver'
    silver_path = os.path.join(silver_dir, 'ml-1m.txt')
    item_features_path = os.path.join(silver_dir, 'ml-1m_item_features.txt')

    if not os.path.exists(bronze_path):
        raise FileNotFoundError(f"Bronze data not found: {bronze_path}")

    os.makedirs(silver_dir, exist_ok=True)

    print(f"Config: {TRAIN_RATIO:.0%} train (no time filter, no user core)")
    print(f"Converting {bronze_path} → {silver_path}")

    # Pass 1: Load all data
    print("\nPass 1: Loading data...")
    data = []
    with open(bronze_path, 'r') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 4:
                user_id = parts[0]
                item_id = parts[1]
                timestamp = int(parts[3])
                data.append((user_id, item_id, timestamp))

    print(f"  Loaded: {len(data):,} interactions")

    # Pass 2: Sort by timestamp and compute train cutoff
    print("\nPass 2: Sorting and computing train cutoff...")
    data.sort(key=lambda x: x[2])

    all_ts = [ts for _, _, ts in data]
    t_min, t_max = min(all_ts), max(all_ts)
    t_cutoff = t_min + int((t_max - t_min) * TRAIN_RATIO)

    train_count = sum(1 for _, _, ts in data if ts < t_cutoff)
    print(f"  Time range: {t_min} - {t_max}")
    print(f"  Train cutoff: {t_cutoff}")
    print(f"  Train interactions: {train_count:,} ({train_count/len(data):.1%})")

    # Pass 3: Compute item features from TRAIN portion only
    print("\nPass 3: Computing item dynamics from TRAIN only...")
    item_cumcount = defaultdict(int)
    item_first_ts = {}
    item_last_ts = {}
    data_with_count = []

    for user_id, item_id, ts in data:
        # Update item features (train only)
        if ts < t_cutoff:
            if item_id not in item_first_ts:
                item_first_ts[item_id] = ts
            item_last_ts[item_id] = ts
            item_cumcount[item_id] += 1

        # Record cumulative count at this point (from train only)
        count = item_cumcount[item_id]
        data_with_count.append((user_id, item_id, ts, count))

    # Build item features (from train only)
    item_features = {}
    for item_id in item_first_ts:
        item_features[item_id] = (
            item_first_ts[item_id],
            item_last_ts[item_id],
            item_cumcount[item_id]
        )

    print(f"  Items in train: {len(item_features):,}")

    # Save item features
    save_item_features(item_features, t_min, t_max, t_cutoff, item_features_path)
    print(f"  Saved to: {item_features_path}")

    # Pass 4: Write output
    print("\nPass 4: Writing output...")
    users = set(row[0] for row in data_with_count)
    items = set(row[1] for row in data_with_count)

    with open(silver_path, 'w') as f:
        for user_id, item_id, ts, count in data_with_count:
            f.write(f"{user_id} {item_id} {ts} {count}\n")

    print(f"\n=== Silver Dataset Stats ===")
    print(f"Users: {len(users):,}")
    print(f"Items: {len(items):,}")
    print(f"Items (in train): {len(item_features):,}")
    print(f"Interactions: {len(data_with_count):,}")
    print(f"Avg interactions/user: {len(data_with_count)/len(users):.1f}")
    print(f"Density: {len(data_with_count)/(len(users)*len(items))*100:.4f}%")
    print(f"Format: UserID ItemID Timestamp CumulativeCount")
    print(f"\nSaved to: {silver_path}")


if __name__ == '__main__':
    main()
