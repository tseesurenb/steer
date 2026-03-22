#!/usr/bin/env python3
"""
Preprocessing script for MovieLens 1M dataset.
Generates ml-1m-time.txt with temporal and popularity features.

Input:  raw/ml-1m/ratings.dat
Output: prep/ml-1m/ml-1m-time.txt

Output format (5 columns, space-separated):
    UserID ItemID Timestamp CumulativeCount Age
"""

import os
from collections import defaultdict


def main():
    raw_path = os.path.join('raw', 'ml-1m', 'ratings.dat')
    output_dir = os.path.join('prep', 'ml-1m')
    output_path = os.path.join(output_dir, 'ml-1m-time.txt')

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {raw_path}...")

    # Load and parse ratings.dat (format: UserID::MovieID::Rating::Timestamp)
    data = []
    with open(raw_path, 'r') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 4:
                user_id = int(parts[0])
                item_id = int(parts[1])
                timestamp = int(parts[3])
                data.append((user_id, item_id, timestamp))

    print(f"Loaded {len(data)} interactions")

    # Sort by timestamp globally
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

    print(f"Users: {len(user_map)}, Items: {len(item_map)}")

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

            # Age = time since item's first appearance
            age = timestamp - first_ts

            # Write: UserID ItemID Timestamp CumulativeCount Age
            f.write(f"{new_user} {new_item} {timestamp} {popularity} {age}\n")

    print(f"Saved to {output_path}")
    print(f"Format: UserID ItemID Timestamp CumulativeCount Age")


if __name__ == '__main__':
    main()
