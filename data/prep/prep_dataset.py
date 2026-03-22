#!/usr/bin/env python3
"""
Unified preprocessing script for all datasets.
Supports different filtering modes: ui10, ui20, u20, etc.

Usage:
    python prep_dataset.py --dataset aws-beauty --filter u20
    python prep_dataset.py --dataset yelp --filter ui10
"""

import os
import json
import argparse
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm


def load_data(dataset):
    """Load raw data based on dataset name."""
    data = []

    if dataset == 'aws-beauty':
        path = 'raw/aws-beauty/Beauty_and_Personal_Care.jsonl'
        with open(path, 'r') as f:
            for line in tqdm(f, desc='Loading'):
                record = json.loads(line.strip())
                user_id = record['user_id']
                item_id = record['asin']
                timestamp = int(record['timestamp'] // 1000)
                data.append((user_id, item_id, timestamp))

    elif dataset == 'aws-health':
        path = 'raw/aws-health/Health_and_Household.jsonl'
        with open(path, 'r') as f:
            for line in tqdm(f, desc='Loading'):
                record = json.loads(line.strip())
                user_id = record['user_id']
                item_id = record['asin']
                timestamp = int(record['timestamp'] // 1000)
                data.append((user_id, item_id, timestamp))

    elif dataset == 'yelp':
        path = 'raw/yelp/yelp_dataset/yelp_academic_dataset_review.json'
        with open(path, 'r') as f:
            for line in tqdm(f, desc='Loading'):
                record = json.loads(line.strip())
                user_id = record['user_id']
                item_id = record['business_id']
                dt = datetime.strptime(record['date'], '%Y-%m-%d %H:%M:%S')
                timestamp = int(dt.timestamp())
                data.append((user_id, item_id, timestamp))

    elif dataset == 'thirtymusic':
        path = 'raw/ThirtyMusic/relations/events.idomaar'
        with open(path, 'r') as f:
            for line in tqdm(f, desc='Loading'):
                parts = line.strip().split('\t')
                if len(parts) < 5 or parts[0] != 'event.play':
                    continue
                timestamp = int(parts[2])
                try:
                    obj = json.loads(parts[4])
                    user_id = None
                    item_id = None
                    for s in obj.get('subjects', []):
                        if s.get('type') == 'user':
                            user_id = s.get('id')
                            break
                    for o in obj.get('objects', []):
                        if o.get('type') == 'track':
                            item_id = o.get('id')
                            break
                    if user_id and item_id:
                        data.append((user_id, item_id, timestamp))
                except json.JSONDecodeError:
                    continue

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return data


def filter_recent(data, years):
    """Filter to keep only the last N years of data."""
    if not data:
        return data

    max_ts = max(t for _, _, t in data)
    seconds_per_year = 365.25 * 24 * 3600
    cutoff = max_ts - (years * seconds_per_year)

    filtered = [(u, i, t) for u, i, t in data if t >= cutoff]
    print(f"Recent filter: keeping last {years} years (cutoff: {cutoff:.0f})")
    return filtered


def filter_data(data, filter_mode):
    """Apply filtering based on mode."""
    print(f"Applying {filter_mode} filtering...")

    if filter_mode == 'u20':
        # User-only 20-core
        user_counts = defaultdict(int)
        for u, _, _ in data:
            user_counts[u] += 1
        data = [(u, i, t) for u, i, t in data if user_counts[u] >= 20]

    elif filter_mode == 'ui10':
        # 10-core for both users and items (iterative)
        prev_len = len(data) + 1
        while len(data) < prev_len:
            prev_len = len(data)
            user_counts = defaultdict(int)
            item_counts = defaultdict(int)
            for u, i, _ in data:
                user_counts[u] += 1
                item_counts[i] += 1
            data = [(u, i, t) for u, i, t in data
                    if user_counts[u] >= 10 and item_counts[i] >= 10]

    elif filter_mode == 'ui20':
        # 20-core for both users and items (iterative)
        prev_len = len(data) + 1
        while len(data) < prev_len:
            prev_len = len(data)
            user_counts = defaultdict(int)
            item_counts = defaultdict(int)
            for u, i, _ in data:
                user_counts[u] += 1
                item_counts[i] += 1
            data = [(u, i, t) for u, i, t in data
                    if user_counts[u] >= 20 and item_counts[i] >= 20]

    elif filter_mode == 'ui30':
        # 30-core for both users and items (iterative)
        prev_len = len(data) + 1
        while len(data) < prev_len:
            prev_len = len(data)
            user_counts = defaultdict(int)
            item_counts = defaultdict(int)
            for u, i, _ in data:
                user_counts[u] += 1
                item_counts[i] += 1
            data = [(u, i, t) for u, i, t in data
                    if user_counts[u] >= 30 and item_counts[i] >= 30]

    elif filter_mode == 'ui50':
        # 50-core for both users and items (iterative)
        prev_len = len(data) + 1
        while len(data) < prev_len:
            prev_len = len(data)
            user_counts = defaultdict(int)
            item_counts = defaultdict(int)
            for u, i, _ in data:
                user_counts[u] += 1
                item_counts[i] += 1
            data = [(u, i, t) for u, i, t in data
                    if user_counts[u] >= 50 and item_counts[i] >= 50]

    elif filter_mode == 'ui100':
        # 100-core for both users and items (iterative)
        prev_len = len(data) + 1
        while len(data) < prev_len:
            prev_len = len(data)
            user_counts = defaultdict(int)
            item_counts = defaultdict(int)
            for u, i, _ in data:
                user_counts[u] += 1
                item_counts[i] += 1
            data = [(u, i, t) for u, i, t in data
                    if user_counts[u] >= 100 and item_counts[i] >= 100]

    elif filter_mode == 'ui200':
        # 200-core for both users and items (iterative)
        prev_len = len(data) + 1
        while len(data) < prev_len:
            prev_len = len(data)
            user_counts = defaultdict(int)
            item_counts = defaultdict(int)
            for u, i, _ in data:
                user_counts[u] += 1
                item_counts[i] += 1
            data = [(u, i, t) for u, i, t in data
                    if user_counts[u] >= 200 and item_counts[i] >= 200]

    else:
        raise ValueError(f"Unknown filter mode: {filter_mode}")

    return data


def process_and_save(data, output_path):
    """Sort, reindex, compute features, and save."""
    print("Sorting by timestamp...")
    data.sort(key=lambda x: x[2])

    print("Reindexing IDs...")
    user_map = {}
    item_map = {}
    for user_id, item_id, _ in data:
        if user_id not in user_map:
            user_map[user_id] = len(user_map) + 1
        if item_id not in item_map:
            item_map[item_id] = len(item_map) + 1

    print(f"Users: {len(user_map):,}, Items: {len(item_map):,}")

    print("Computing features and saving...")
    item_first_ts = {}
    item_pop_count = defaultdict(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for user_id, item_id, timestamp in data:
            new_user = user_map[user_id]
            new_item = item_map[item_id]

            if new_item not in item_first_ts:
                item_first_ts[new_item] = timestamp

            first_ts = item_first_ts[new_item]
            item_pop_count[new_item] += 1
            popularity = item_pop_count[new_item]
            decay = timestamp - first_ts

            f.write(f"{new_user} {new_item} {timestamp} {first_ts} {popularity} {decay}\n")

    print(f"Saved: {output_path}")
    print(f"Total: {len(data):,} interactions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['aws-beauty', 'aws-health', 'yelp', 'thirtymusic'])
    parser.add_argument('--filter', default=None, choices=['ui10', 'ui20', 'ui30', 'ui50', 'ui100', 'ui200', 'u20'])
    parser.add_argument('--recent', type=int, default=None, help='Keep only last N years of data')
    args = parser.parse_args()

    # Determine output filename
    suffix = f'_recent{args.recent}' if args.recent else ''
    if args.filter:
        output_path = f"prep/{args.dataset}/{args.filter}/{args.dataset}-time{suffix}.txt"
    else:
        output_path = f"prep/{args.dataset}/{args.dataset}-time{suffix}.txt"

    print(f"Dataset: {args.dataset}")
    print(f"Filter: {args.filter or 'none'}")
    print(f"Recent: {args.recent or 'all'} years")
    print(f"Output: {output_path}")
    print()

    data = load_data(args.dataset)
    print(f"Loaded: {len(data):,} interactions")

    # Apply recent filter first (before core filtering)
    if args.recent:
        data = filter_recent(data, args.recent)
        print(f"After recent filter: {len(data):,} interactions")

    # Apply core filtering
    if args.filter:
        data = filter_data(data, args.filter)
        print(f"After core filtering: {len(data):,} interactions")

    process_and_save(data, output_path)


if __name__ == '__main__':
    main()
