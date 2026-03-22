#!/usr/bin/env python3
"""
Bronze → Silver conversion for Amazon Health and Household dataset.

Input:  bronze/aws-health/Health_and_Household.jsonl
Output: silver/aws-health.txt, silver/aws-health_item_features.txt

Output format for aws-health.txt (4 columns, space-separated):
    UserID ItemID Timestamp CumulativeCount

Output format for aws-health_item_features.txt:
    Line 1: T_min T_max T_train_cutoff (timestamp range and train cutoff)
    Line 2+: ItemID FirstTimestamp LastTimestamp NumInteractions

Flow: Time filter -> Split (90% train) -> Compute item features from train -> User core

IMPORTANT: Item dynamics (first_ts, cumulative_count) are computed from
TRAIN portion only (first 90% by time) to avoid data leakage.

Default: 2020-2022, 20-core user filter, 90% train split

Usage:
    python to_silver_aws_health.py                         # Default
    python to_silver_aws_health.py --user_core 10          # 10-core filter
"""

import os
import json
import argparse
from datetime import datetime
from collections import defaultdict


TRAIN_RATIO = 0.90  # Fixed train split ratio


def save_item_features(features, t_min, t_max, t_cutoff, output_path):
    """Save item features to file."""
    with open(output_path, 'w') as f:
        f.write(f"{t_min} {t_max} {t_cutoff}\n")
        for item_id in sorted(features.keys()):
            first_ts, last_ts, num_interactions = features[item_id]
            f.write(f"{item_id} {first_ts} {last_ts} {num_interactions}\n")


def main():
    parser = argparse.ArgumentParser(description='Bronze → Silver for AWS-Health')
    parser.add_argument('--start_year', type=int, default=2020,
                        help='Start year inclusive (default: 2020)')
    parser.add_argument('--end_year', type=int, default=2022,
                        help='End year inclusive (default: 2022)')
    parser.add_argument('--user_core', type=int, default=20,
                        help='Minimum interactions per user (default: 20)')
    args = parser.parse_args()

    bronze_path = os.path.join('bronze', 'aws-health', 'Health_and_Household.jsonl')
    silver_dir = 'silver'
    silver_path = os.path.join(silver_dir, 'aws-health.txt')
    item_features_path = os.path.join(silver_dir, 'aws-health_item_features.txt')

    if not os.path.exists(bronze_path):
        raise FileNotFoundError(f"Bronze data not found: {bronze_path}")

    os.makedirs(silver_dir, exist_ok=True)

    print(f"Config: {args.start_year}-{args.end_year}, {args.user_core}-core, {TRAIN_RATIO:.0%} train")
    print(f"Converting {bronze_path} → {silver_path}")

    # Pass 1: Load and filter by year
    print("\nPass 1: Loading and filtering by year...")
    data = []
    n_read = 0
    with open(bronze_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            timestamp = int(record['timestamp'] // 1000)  # ms → seconds
            year = datetime.fromtimestamp(timestamp).year

            if args.start_year <= year <= args.end_year:
                user_id = record['user_id']
                item_id = record['asin']
                data.append((user_id, item_id, timestamp))

            n_read += 1
            if n_read % 5000000 == 0:
                print(f"  {n_read // 1000000}M read, {len(data):,} kept...")

    print(f"  After year filter: {len(data):,} interactions")

    # Pass 2: Sort by timestamp and compute train cutoff
    print("\nPass 2: Sorting and computing train cutoff...")
    data.sort(key=lambda x: x[2])  # Sort by timestamp

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
    data_with_counts = []

    for user_id, item_id, ts in data:
        if ts < t_cutoff:
            # Train: update item features
            if item_id not in item_first_ts:
                item_first_ts[item_id] = ts
            item_last_ts[item_id] = ts
            item_cumcount[item_id] += 1

        # Record cumulative count at this point (from train only)
        count = item_cumcount[item_id]  # 0 if item not seen in train yet
        data_with_counts.append((user_id, item_id, ts, count))

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

    # Pass 4: Apply user core filter
    print("\nPass 4: Applying user core filter...")
    user_counts = defaultdict(int)
    for user_id, _, _, _ in data_with_counts:
        user_counts[user_id] += 1

    valid_users = {u for u, c in user_counts.items() if c >= args.user_core}
    print(f"  Users with >={args.user_core} interactions: {len(valid_users):,}")

    # Pass 5: Filter and write
    print("\nPass 5: Filtering and writing...")
    filtered = [(u, i, ts, cnt) for u, i, ts, cnt in data_with_counts if u in valid_users]

    # Get unique items in filtered data
    items = set(i for _, i, _, _ in filtered)

    with open(silver_path, 'w') as f:
        for user_id, item_id, ts, count in filtered:
            f.write(f"{user_id} {item_id} {ts} {count}\n")

    print(f"\n=== Silver Dataset Stats ===")
    print(f"Users: {len(valid_users):,}")
    print(f"Items (in filtered): {len(items):,}")
    print(f"Items (in train): {len(item_features):,}")
    print(f"Interactions: {len(filtered):,}")
    print(f"Avg interactions/user: {len(filtered)/len(valid_users):.1f}")
    print(f"Density: {len(filtered)/(len(valid_users)*len(items))*100:.4f}%")
    print(f"Format: UserID ItemID Timestamp CumulativeCount")
    print(f"\nSaved to: {silver_path}")


if __name__ == '__main__':
    main()
