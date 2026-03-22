#!/usr/bin/env python3
"""
Preprocess 30music dataset.

Input:  gold/30m/30m-time.txt (preprocessed 30Music data)
Output: silver/30m.txt, silver/30m_item_features.txt

Output format for 30m.txt (4 columns, space-separated):
    UserID ItemID Timestamp CumulativeCount

Output format for 30m_item_features.txt:
    Line 1: T_min T_max T_train_cutoff
    Line 2+: ItemID FirstTimestamp LastTimestamp NumInteractions

Steps:
1. Remove repeated interactions (keep only first per user-item pair)
2. Compute item features from train portion
3. Apply user/item core filters
4. Output silver format (4 columns)

Usage:
    python to_silver_30m.py
    python to_silver_30m.py --user_core 30 --item_core 150
"""

import os
import argparse
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
    parser = argparse.ArgumentParser(description='Preprocess 30music')
    parser.add_argument('--user_core', type=int, default=20,
                        help='Minimum interactions per user (default: 20)')
    parser.add_argument('--item_core', type=int, default=0,
                        help='Minimum interactions per item (default: 0, no filter)')
    args = parser.parse_args()

    input_path = os.path.join('gold', '30m', '30m-time.txt')
    silver_dir = 'silver'
    silver_path = os.path.join(silver_dir, '30m.txt')
    item_features_path = os.path.join(silver_dir, '30m_item_features.txt')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    os.makedirs(silver_dir, exist_ok=True)

    print(f"Config: user_core={args.user_core}, item_core={args.item_core}")
    print(f"Converting {input_path} → {silver_path}")

    # Step 1: Load and remove repeated interactions
    print("\n[1/5] Loading and removing repeated interactions...")
    data = []
    seen_pairs = set()
    total_count = 0
    repeat_count = 0

    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                total_count += 1
                user_id = parts[0]
                item_id = parts[1]
                timestamp = int(parts[2])

                key = (user_id, item_id)
                if key in seen_pairs:
                    repeat_count += 1
                    continue  # Skip repeated interactions

                seen_pairs.add(key)
                data.append((user_id, item_id, timestamp))

    print(f"  Original: {total_count:,} interactions")
    print(f"  Removed: {repeat_count:,} repeats ({repeat_count/total_count*100:.1f}%)")
    print(f"  Kept: {len(data):,} first interactions")

    # Step 2: Sort by timestamp and compute train cutoff
    print("\n[2/5] Sorting and computing train cutoff...")
    data.sort(key=lambda x: x[2])

    all_ts = [ts for _, _, ts in data]
    t_min, t_max = min(all_ts), max(all_ts)
    t_cutoff = t_min + int((t_max - t_min) * TRAIN_RATIO)

    train_count = sum(1 for _, _, ts in data if ts < t_cutoff)
    print(f"  Time range: {t_min} - {t_max}")
    print(f"  Train cutoff: {t_cutoff}")
    print(f"  Train interactions: {train_count:,} ({train_count/len(data):.1%})")

    # Step 3: Compute item features from TRAIN portion only
    print("\n[3/5] Computing item features from TRAIN only...")
    item_cumcount = defaultdict(int)
    item_first_ts = {}
    item_last_ts = {}
    data_with_count = []

    for user_id, item_id, ts in data:
        if ts < t_cutoff:
            if item_id not in item_first_ts:
                item_first_ts[item_id] = ts
            item_last_ts[item_id] = ts
            item_cumcount[item_id] += 1

        # Record count at this point (from train only)
        count = item_cumcount[item_id]
        data_with_count.append((user_id, item_id, ts, count))

    # Build item features
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

    # Step 4: Apply core filters
    print("\n[4/5] Applying core filters...")

    # Item core filter (if specified)
    if args.item_core > 0:
        valid_items = {i for i, c in item_cumcount.items() if c >= args.item_core}
        print(f"  Items with >={args.item_core} interactions: {len(valid_items):,}")
        data_with_count = [row for row in data_with_count if row[1] in valid_items]

    # User core filter
    user_counts = defaultdict(int)
    for row in data_with_count:
        user_counts[row[0]] += 1

    valid_users = {u for u, c in user_counts.items() if c >= args.user_core}
    print(f"  Users with >={args.user_core} interactions: {len(valid_users):,}")

    filtered = [row for row in data_with_count if row[0] in valid_users]

    # Step 5: Write output
    print("\n[5/5] Writing output...")
    users = set(row[0] for row in filtered)
    items = set(row[1] for row in filtered)

    with open(silver_path, 'w') as f:
        for user_id, item_id, ts, count in filtered:
            f.write(f"{user_id} {item_id} {ts} {count}\n")

    print(f"\n=== Silver Dataset Stats ===")
    print(f"Users: {len(users):,}")
    print(f"Items: {len(items):,}")
    print(f"Interactions: {len(filtered):,}")
    print(f"Avg interactions/user: {len(filtered)/len(users):.1f}")
    print(f"Density: {len(filtered)/(len(users)*len(items))*100:.4f}%")
    print(f"Format: UserID ItemID Timestamp CumulativeCount")
    print(f"\nSaved to: {silver_path}")


if __name__ == '__main__':
    main()
