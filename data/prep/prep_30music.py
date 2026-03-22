"""
Preprocess 30Music dataset for sequential recommendation.

Raw data format (events.idomaar):
event.play  ID  TIMESTAMP  {"playtime":X}  {"subjects":[{"type":"user","id":X}], "objects":[{"type":"track","id":X}]}
"""
import os
import json
import argparse
from collections import defaultdict
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_core', type=int, default=200, help='Minimum interactions per user')
    parser.add_argument('--item_core', type=int, default=200, help='Minimum interactions per item')
    parser.add_argument('--max_len', type=int, default=None, help='Max sequence length per user (truncate oldest)')
    return parser.parse_args()

def main():
    args = parse_args()

    bronze_path = 'data/bronze/30music/relations/events.idomaar'

    # Output naming based on filtering
    suffix = f"u{args.user_core}_i{args.item_core}"
    output_dir = f'data/gold/30music_{suffix}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== 30Music Preprocessing ===")
    print(f"User core: {args.user_core}, Item core: {args.item_core}")
    print(f"Output: {output_dir}")

    # Step 1: Parse raw events
    print("\n[1/4] Parsing raw events...")
    interactions = []  # (user_id, item_id, timestamp)

    with open(bronze_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 5_000_000 == 0:
                print(f"  Processed {i:,} lines...")

            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            timestamp = int(parts[2])

            # Parse JSON for user and track
            try:
                meta = json.loads(parts[4])
                user_id = meta['subjects'][0]['id']
                item_id = meta['objects'][0]['id']
                interactions.append((user_id, item_id, timestamp))
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    print(f"  Total raw interactions: {len(interactions):,}")

    # Step 2: Apply iterative core filtering
    print(f"\n[2/4] Applying {args.user_core}/{args.item_core}-core filtering...")

    prev_count = -1
    iteration = 0
    while len(interactions) != prev_count:
        prev_count = len(interactions)
        iteration += 1

        # Count per user and item
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        for u, i, t in interactions:
            user_counts[u] += 1
            item_counts[i] += 1

        # Filter
        interactions = [
            (u, i, t) for u, i, t in interactions
            if user_counts[u] >= args.user_core and item_counts[i] >= args.item_core
        ]

        print(f"  Iteration {iteration}: {len(interactions):,} interactions")

    if len(interactions) == 0:
        print("ERROR: No interactions left after filtering. Try lower core values.")
        return

    # Step 3: Remap IDs and compute features
    print("\n[3/4] Remapping IDs and computing features...")

    # Sort by timestamp
    interactions.sort(key=lambda x: x[2])

    # Remap user and item IDs to contiguous integers
    user_map = {}
    item_map = {}

    for u, i, t in interactions:
        if u not in user_map:
            user_map[u] = len(user_map) + 1  # 1-indexed
        if i not in item_map:
            item_map[i] = len(item_map) + 1  # 1-indexed

    # Compute item features
    item_first_ts = {}  # First timestamp for each item
    item_counts = defaultdict(int)  # Cumulative count

    processed = []
    for u, i, t in interactions:
        new_u = user_map[u]
        new_i = item_map[i]

        # Track first timestamp
        if new_i not in item_first_ts:
            item_first_ts[new_i] = t

        # Increment count before recording (so first interaction has count=1)
        item_counts[new_i] += 1

        processed.append((new_u, new_i, t, item_first_ts[new_i], item_counts[new_i]))

    # Optional: truncate per user
    if args.max_len:
        print(f"  Truncating to max {args.max_len} per user...")
        user_interactions = defaultdict(list)
        for row in processed:
            user_interactions[row[0]].append(row)

        processed = []
        for u in user_interactions:
            # Keep most recent max_len interactions
            user_ints = sorted(user_interactions[u], key=lambda x: x[2])
            processed.extend(user_ints[-args.max_len:])

        # Re-sort globally
        processed.sort(key=lambda x: x[2])

    # Step 4: Write output
    print("\n[4/4] Writing output...")

    output_file = os.path.join(output_dir, f'30music_{suffix}.txt')
    with open(output_file, 'w') as f:
        for u, i, t, t_first, count in processed:
            f.write(f"{u} {i} {t} {t_first} {count}\n")

    # Statistics
    users = set(row[0] for row in processed)
    items = set(row[1] for row in processed)

    # Per-user stats
    user_lens = defaultdict(int)
    for row in processed:
        user_lens[row[0]] += 1
    avg_len = sum(user_lens.values()) / len(user_lens)

    # Time range
    min_ts = min(row[2] for row in processed)
    max_ts = max(row[2] for row in processed)
    time_span_days = (max_ts - min_ts) / 86400

    print(f"\n=== Final Statistics ===")
    print(f"Users: {len(users):,}")
    print(f"Items: {len(items):,}")
    print(f"Interactions: {len(processed):,}")
    print(f"Avg interactions/user: {avg_len:.1f}")
    print(f"Density: {len(processed) / (len(users) * len(items)) * 100:.4f}%")
    print(f"Time span: {time_span_days:.0f} days ({time_span_days/365:.1f} years)")
    print(f"Date range: {datetime.fromtimestamp(min_ts)} to {datetime.fromtimestamp(max_ts)}")
    print(f"\nOutput: {output_file}")

if __name__ == '__main__':
    main()
