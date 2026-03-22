"""Generate item_feature.txt from training data for each dataset.

File format:
- Line 1: T_min T_max (global timestamp range of training data)
- Line 2+: ItemID FirstTimestamp LastTimestamp NumUsers
"""
import os
from collections import defaultdict
from pathlib import Path

# Datasets and their filters
DATASETS = [
    ('ml-1m', 'ui10'),
    ('aws-beauty', 'ui10'),
    ('aws-beauty', 'u20'),
    ('aws-health', 'ui10'),
    ('yelp', 'ui10'),
    ('yelp', 'u20'),
    ('thirtymusic', 'ui10'),
    ('thirtymusic', 'ui200'),
]

SPLIT_RATIO = 0.1  # Same as default in config.py


def load_data_with_timestamps(fname):
    """Load data file with format: user item timestamp ..."""
    data = []  # List of (user, item, timestamp)

    with open(fname, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            user, item, ts = int(parts[0]), int(parts[1]), int(parts[2])
            data.append((user, item, ts))

    return data


def temporal_split_cutoff(data, split_ratio):
    """Get temporal cutoff for training data."""
    all_ts = [ts for _, _, ts in data]
    min_time, max_time = min(all_ts), max(all_ts)
    cutoff = min_time + (1 - split_ratio) * (max_time - min_time)
    return cutoff, min_time, max_time


def compute_item_features(data, cutoff):
    """Compute item features from training data (before cutoff).

    Returns:
        features: dict item_id -> (first_ts, last_ts, num_users)
        t_min: minimum timestamp in training data
        t_max: maximum timestamp in training data (before cutoff)
    """
    item_timestamps = defaultdict(list)
    item_users = defaultdict(set)
    train_timestamps = []

    for user, item, ts in data:
        if ts < cutoff:  # Training data only
            item_timestamps[item].append(ts)
            item_users[item].add(user)
            train_timestamps.append(ts)

    features = {}
    for item in item_timestamps:
        first_ts = min(item_timestamps[item])
        last_ts = max(item_timestamps[item])
        num_users = len(item_users[item])
        features[item] = (first_ts, last_ts, num_users)

    t_min = min(train_timestamps)
    t_max = max(train_timestamps)

    return features, t_min, t_max


def save_item_features(features, t_min, t_max, output_path):
    """Save item features to file with global time range header."""
    with open(output_path, 'w') as f:
        # First line: global time range
        f.write(f"{t_min} {t_max}\n")
        # Remaining lines: item features
        for item in sorted(features.keys()):
            first_ts, last_ts, num_users = features[item]
            f.write(f"{item} {first_ts} {last_ts} {num_users}\n")


def main():
    base_dir = Path(__file__).parent / 'prep'

    for dataset, filter_type in DATASETS:
        data_path = base_dir / dataset / filter_type / f'{dataset}-time.txt'

        if not data_path.exists():
            print(f"Skipping {dataset}/{filter_type}: file not found")
            continue

        print(f"Processing {dataset}/{filter_type}...")

        # Load data
        data = load_data_with_timestamps(data_path)
        print(f"  Loaded {len(data):,} interactions")

        # Get temporal cutoff
        cutoff, _, _ = temporal_split_cutoff(data, SPLIT_RATIO)
        train_count = sum(1 for _, _, ts in data if ts < cutoff)
        print(f"  Training interactions: {train_count:,}")

        # Compute item features
        features, t_min, t_max = compute_item_features(data, cutoff)
        print(f"  Items with features: {len(features):,}")
        print(f"  Time range: {t_min} - {t_max}")

        # Save
        output_path = base_dir / dataset / filter_type / 'item_feature.txt'
        save_item_features(features, t_min, t_max, output_path)
        print(f"  Saved to {output_path}")
        print()


if __name__ == '__main__':
    main()
