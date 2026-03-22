#!/usr/bin/env python3
"""
Filter gold data by user k-core while preserving item dynamics.

Input:  gold/{dataset}/{dataset}-time.txt
Output: gold/{dataset}_u{k}/{dataset}_u{k}-time.txt

Preserves pre-computed Count and Age values (computed from full data).
Reindexes users and items to be contiguous (1 to N).

Usage:
    python filter_gold.py aws-health_2020-2023 --user_core 20
"""

import os
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Filter gold data by user k-core')
    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument('--user_core', type=int, required=True,
                        help='Minimum interactions per user')
    args = parser.parse_args()

    input_path = os.path.join('gold', args.dataset, f'{args.dataset}-time.txt')
    output_name = f'{args.dataset}_u{args.user_core}'
    output_dir = os.path.join('gold', output_name)
    output_path = os.path.join(output_dir, f'{output_name}-time.txt')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading {input_path}...")
    data = []
    user_counts = defaultdict(int)
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            user = int(parts[0])
            item = int(parts[1])
            ts = int(parts[2])
            count = int(parts[3])
            age = int(parts[4])
            data.append((user, item, ts, count, age))
            user_counts[user] += 1

    print(f"Loaded {len(data):,} interactions, {len(user_counts):,} users")

    # Filter users
    print(f"Filtering users with >={args.user_core} interactions...")
    valid_users = {u for u, c in user_counts.items() if c >= args.user_core}
    filtered = [(u, i, ts, cnt, age) for u, i, ts, cnt, age in data if u in valid_users]
    print(f"  {len(data):,} → {len(filtered):,} interactions")
    print(f"  {len(user_counts):,} → {len(valid_users):,} users")

    # Find items with interactions
    items_with_interactions = set(i for _, i, _, _, _ in filtered)
    print(f"  Items with interactions: {len(items_with_interactions):,}")

    # Reindex users and items
    print("Reindexing...")
    user_map = {}
    item_map = {}

    # Sort by timestamp to ensure consistent ordering
    filtered.sort(key=lambda x: x[2])

    for user, item, _, _, _ in filtered:
        if user not in user_map:
            user_map[user] = len(user_map) + 1
        if item not in item_map:
            item_map[item] = len(item_map) + 1

    # Save
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for user, item, ts, count, age in filtered:
            new_user = user_map[user]
            new_item = item_map[item]
            f.write(f"{new_user} {new_item} {ts} {count} {age}\n")

    # Stats
    n_users = len(user_map)
    n_items = len(item_map)
    n_inter = len(filtered)
    density = n_inter / (n_users * n_items) * 100

    print(f"\n=== Filtered Gold Stats ===")
    print(f"Users: {n_users:,}")
    print(f"Items: {n_items:,}")
    print(f"Interactions: {n_inter:,}")
    print(f"Density: {density:.4f}%")
    print(f"Avg interactions/user: {n_inter/n_users:.1f}")
    print(f"Format: UserID ItemID Timestamp CumulativeCount Age")


if __name__ == '__main__':
    main()
