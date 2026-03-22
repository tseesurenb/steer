#!/usr/bin/env python3
"""
Preprocessing script for Amazon Beauty and Personal Care dataset.
Generates aws-beauty-time.txt with temporal and popularity features.

Input:  raw/aws-beauty/Beauty_and_Personal_Care.jsonl
Output: prep/aws-beauty/aws-beauty-time.txt

Usage:
    python prep_aws_beauty.py --user_core 50  # User 50-core, no item filtering

Output format (6 columns, space-separated):
    UserID ItemID Timestamp FirstTimestamp ItemPopularitySoFar Decay
"""

import os
import json
import argparse
from collections import defaultdict


def filter_user_core(data, min_user=50):
    """Filter to keep only users with at least min_user interactions.

    NOTE: No item filtering - preserves all items including new/cold ones
    to maintain temporal lifecycle signals.
    """
    print(f"Applying user {min_user}-core filtering (no item filtering)...")

    user_counts = defaultdict(int)
    for user_id, item_id, _ in data:
        user_counts[user_id] += 1

    data = [(u, i, t) for u, i, t in data if user_counts[u] >= min_user]

    # Count remaining
    users = set(u for u, i, t in data)
    items = set(i for u, i, t in data)
    print(f"  Users: {len(users):,}, Items: {len(items):,}, Interactions: {len(data):,}")

    return data


def main():
    parser = argparse.ArgumentParser(description='Preprocess AWS Beauty dataset')
    parser.add_argument('--user_core', type=int, default=50,
                        help='Minimum interactions per user (default: 50)')
    args = parser.parse_args()

    raw_path = os.path.join('raw', 'aws-beauty', 'Beauty_and_Personal_Care.jsonl')
    output_dir = os.path.join('prep', 'aws-beauty')
    output_path = os.path.join(output_dir, 'aws-beauty-time.txt')

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {raw_path}...")

    # Load and parse JSONL
    data = []
    with open(raw_path, 'r') as f:
        for i, line in enumerate(f):
            record = json.loads(line.strip())
            user_id = record['user_id']
            item_id = record['asin']
            timestamp = int(record['timestamp'] // 1000)  # Convert ms to seconds
            data.append((user_id, item_id, timestamp))
            if (i + 1) % 5000000 == 0:
                print(f"  Loaded {(i+1)//1000000}M...")

    print(f"Loaded {len(data):,} interactions")

    # Apply user-core filtering (no item filtering to preserve temporal signals)
    data = filter_user_core(data, args.user_core)

    # Sort by timestamp globally
    print("Sorting by timestamp...")
    data.sort(key=lambda x: x[2])

    # Reindex user and item IDs to be contiguous starting from 1
    print("Reindexing user and item IDs...")
    user_map = {}
    item_map = {}

    for user_id, item_id, _ in data:
        if user_id not in user_map:
            user_map[user_id] = len(user_map) + 1
        if item_id not in item_map:
            item_map[item_id] = len(item_map) + 1

    print(f"Users: {len(user_map):,}, Items: {len(item_map):,}")

    # Compute features and write output
    print("Computing features and writing output...")

    item_first_ts = {}  # First timestamp for each item
    item_pop_count = defaultdict(int)  # Popularity count for each item

    with open(output_path, 'w') as f:
        for user_id, item_id, timestamp in data:
            new_user = user_map[user_id]
            new_item = item_map[item_id]

            # Track first timestamp for this item
            if new_item not in item_first_ts:
                item_first_ts[new_item] = timestamp

            first_ts = item_first_ts[new_item]

            # Popularity so far (incremented after this interaction)
            item_pop_count[new_item] += 1
            popularity = item_pop_count[new_item]

            # Decay = time since item's first appearance
            decay = timestamp - first_ts

            # Write: UserID ItemID Timestamp FirstTimestamp ItemPopularitySoFar Decay
            f.write(f"{new_user} {new_item} {timestamp} {first_ts} {popularity} {decay}\n")

    print(f"Saved to {output_path}")
    print(f"Format: UserID ItemID Timestamp FirstTimestamp ItemPopularitySoFar Decay")


if __name__ == '__main__':
    main()
