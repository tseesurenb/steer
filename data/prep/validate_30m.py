#!/usr/bin/env python3
"""
Validate 30m dataset by regenerating from raw and comparing with current.

Reads: bronze/ThirtyMusic/relations/events.idomaar
Generates: temp files for comparison
Compares: with current silver/gold files

Following our preprocessing rules:
1. Load raw data (parse events.idomaar)
2. Sort by timestamp
3. Compute train cutoff (90%)
4. Compute item features from TRAIN only (BEFORE filtering)
5. Save item_features
6. Apply user core (20) and item core filters
7. Write silver (4 columns)
8. Reindex and compute age for gold (5 columns)
"""

import os
import json
from collections import defaultdict
from datetime import datetime

TRAIN_RATIO = 0.90
USER_CORE = 20
ITEM_CORE = 150  # Based on preprocessing.md

def main():
    raw_path = os.path.join('bronze', 'ThirtyMusic', 'relations', 'events.idomaar')

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    print("=" * 60)
    print("VALIDATING 30m DATASET")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n[1/7] Loading raw events.idomaar...")
    data = []
    with open(raw_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                ts = int(parts[2])
                subj_obj = json.loads(parts[4])
                user_id = subj_obj['subjects'][0]['id']
                item_id = subj_obj['objects'][0]['id']
                data.append((user_id, item_id, ts))

            if (i + 1) % 5000000 == 0:
                print(f"  Loaded {(i+1) // 1000000}M...")

    print(f"  Total raw events: {len(data):,}")

    # Step 2: Sort by timestamp
    print("\n[2/7] Sorting by timestamp...")
    data.sort(key=lambda x: x[2])

    all_ts = [ts for _, _, ts in data]
    t_min, t_max = min(all_ts), max(all_ts)
    t_cutoff = t_min + int((t_max - t_min) * TRAIN_RATIO)

    print(f"  Time range: {t_min} - {t_max}")
    print(f"  Date range: {datetime.fromtimestamp(t_min)} - {datetime.fromtimestamp(t_max)}")
    print(f"  Train cutoff: {t_cutoff}")

    # Step 3: Remove duplicate user-item pairs (keep first)
    print("\n[3/7] Removing duplicate user-item pairs...")
    seen_pairs = set()
    unique_data = []
    for user_id, item_id, ts in data:
        key = (user_id, item_id)
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_data.append((user_id, item_id, ts))

    print(f"  Before dedup: {len(data):,}")
    print(f"  After dedup: {len(unique_data):,}")
    print(f"  Removed: {len(data) - len(unique_data):,} duplicates")

    data = unique_data

    # Step 4: Compute item features from TRAIN only (BEFORE filtering)
    print("\n[4/7] Computing item features from TRAIN only...")
    item_cumcount = defaultdict(int)
    item_first_ts = {}
    item_last_ts = {}
    data_with_count = []

    train_count = 0
    for user_id, item_id, ts in data:
        if ts < t_cutoff:
            train_count += 1
            if item_id not in item_first_ts:
                item_first_ts[item_id] = ts
            item_last_ts[item_id] = ts
            item_cumcount[item_id] += 1

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

    print(f"  Train interactions: {train_count:,} ({train_count/len(data):.1%})")
    print(f"  Items in train: {len(item_features):,}")

    # Step 5: Apply item core filter
    print(f"\n[5/7] Applying item core filter (>={ITEM_CORE})...")
    valid_items = {i for i, c in item_cumcount.items() if c >= ITEM_CORE}
    print(f"  Items passing filter: {len(valid_items):,}")

    data_with_count = [row for row in data_with_count if row[1] in valid_items]
    print(f"  Interactions after item filter: {len(data_with_count):,}")

    # Step 6: Apply user core filter
    print(f"\n[6/7] Applying user core filter (>={USER_CORE})...")
    user_counts = defaultdict(int)
    for row in data_with_count:
        user_counts[row[0]] += 1

    valid_users = {u for u, c in user_counts.items() if c >= USER_CORE}
    print(f"  Users passing filter: {len(valid_users):,}")

    filtered = [row for row in data_with_count if row[0] in valid_users]
    print(f"  Interactions after user filter: {len(filtered):,}")

    # Step 7: Compute final stats (silver equivalent)
    print("\n[7/7] Computing final stats...")
    final_users = set(row[0] for row in filtered)
    final_items = set(row[1] for row in filtered)

    print(f"\n{'=' * 60}")
    print("FRESH GENERATION STATS (Silver equivalent)")
    print(f"{'=' * 60}")
    print(f"Users: {len(final_users):,}")
    print(f"Items: {len(final_items):,}")
    print(f"Interactions: {len(filtered):,}")
    print(f"Avg interactions/user: {len(filtered)/len(final_users):.1f}")

    # Now compare with current files
    print(f"\n{'=' * 60}")
    print("COMPARING WITH CURRENT DATA")
    print(f"{'=' * 60}")

    # Load current silver
    current_silver_path = os.path.join('silver', '30m.txt')
    if os.path.exists(current_silver_path):
        current_users = set()
        current_items = set()
        current_count = 0
        with open(current_silver_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                current_users.add(int(parts[0]))
                current_items.add(int(parts[1]))
                current_count += 1

        print(f"\nCurrent Silver:")
        print(f"  Users: {len(current_users):,}")
        print(f"  Items: {len(current_items):,}")
        print(f"  Interactions: {current_count:,}")

        print(f"\nFresh Generated:")
        print(f"  Users: {len(final_users):,}")
        print(f"  Items: {len(final_items):,}")
        print(f"  Interactions: {len(filtered):,}")

        print(f"\nDifference:")
        print(f"  Users: {len(final_users) - len(current_users):+,}")
        print(f"  Items: {len(final_items) - len(current_items):+,}")
        print(f"  Interactions: {len(filtered) - current_count:+,}")

        if (len(final_users) == len(current_users) and
            len(final_items) == len(current_items) and
            len(filtered) == current_count):
            print(f"\n✅ VALIDATION PASSED - Counts match!")
        else:
            print(f"\n❌ VALIDATION FAILED - Counts differ!")
    else:
        print(f"Current silver file not found: {current_silver_path}")

    # Load current gold
    current_gold_path = os.path.join('gold', '30m', '30m-time.txt')
    if os.path.exists(current_gold_path):
        gold_users = set()
        gold_items = set()
        gold_count = 0
        with open(current_gold_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                gold_users.add(int(parts[0]))
                gold_items.add(int(parts[1]))
                gold_count += 1

        print(f"\nCurrent Gold:")
        print(f"  Users: {len(gold_users):,}")
        print(f"  Items: {len(gold_items):,}")
        print(f"  Interactions: {gold_count:,}")


if __name__ == '__main__':
    main()
