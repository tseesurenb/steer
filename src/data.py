import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from utils import print_once


def load_data(fname):
    """Load data file with format: UserID ItemID Timestamp CumulativeCount Age"""
    user_items = defaultdict(list)
    user_ts = defaultdict(list)
    user_count = defaultdict(list)  # Dynamic C: cumulative count at interaction time
    user_age = defaultdict(list)    # Dynamic A: item age at interaction time
    num_users, num_items = 0, 0

    with open(fname, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            user, item = int(parts[0]), int(parts[1])
            user_items[user].append(item)
            num_users = max(num_users, user)
            num_items = max(num_items, item)

            if len(parts) >= 3:
                user_ts[user].append(int(parts[2]))

            if len(parts) >= 4:
                user_count[user].append(int(parts[3]))  # Column 4: cumulative count (C)

            if len(parts) >= 5:
                user_age[user].append(int(parts[4]))    # Column 5: age (A)

    return (user_items, num_users, num_items, user_ts, user_count, user_age)


def _load_item_features_file(item_features_path, num_items):
    """Load item features from pre-computed file.

    File format: ItemID t_first t_last n_users
    Generated during preprocessing from ALL data before user filtering.
    """
    t_first = torch.zeros(num_items + 1, dtype=torch.float32)
    t_last = torch.zeros(num_items + 1, dtype=torch.float32)
    n_users = torch.zeros(num_items + 1, dtype=torch.float32)

    with open(item_features_path, 'r') as f:
        for line in f:
            if line.startswith('#'):  # Skip header
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            item_id = int(parts[0])
            if item_id <= num_items:
                t_first[item_id] = float(parts[1])
                t_last[item_id] = float(parts[2])
                n_users[item_id] = float(parts[3])  # total_pop as n_users

    # Compute normalization values
    all_t_first = t_first[t_first > 0]
    all_t_last = t_last[t_last > 0]

    t_min = float(all_t_first.min()) if len(all_t_first) > 0 else 0
    t_max = float(all_t_last.max()) if len(all_t_last) > 0 else 0
    max_n_users = float(n_users.max())
    max_age = t_max - t_min
    max_stale = t_max - float(all_t_last.min()) if len(all_t_last) > 0 else 0

    print_once(f"Item features: LOADED from pre-computed file ({item_features_path})")
    return {
        't_first': t_first,
        't_last': t_last,
        'n_users': n_users,
        't_min': t_min,
        't_max': t_max,
        'max_n_users': max_n_users,
        'max_age': max_age,
        'max_stale': max_stale,
    }


def get_item_features(data_path, num_items):
    """Load item features from pre-computed file.

    ALL datasets require item_features.txt generated during preprocessing.
    Item features must be computed BEFORE user filtering to reflect true item dynamics.
    """
    dataset_dir = os.path.dirname(data_path)
    item_features_path = os.path.join(dataset_dir, 'item_features.txt')

    if not os.path.exists(item_features_path):
        raise FileNotFoundError(
            f"Missing item_features.txt for dataset.\n"
            f"Expected: {item_features_path}\n"
            f"Run preprocessing to generate this file."
        )

    return _load_item_features_file(item_features_path, num_items)


class SeqDataset(Dataset):
    def __init__(self, user_train, maxlen, num_items, user_train_ts=None, user_train_count=None, user_train_age=None):
        self.user_train = user_train
        self.maxlen = maxlen
        self.num_items = num_items
        self.user_list = sorted(user_train.keys())
        self.user_pos_set = {u: set(items) for u, items in user_train.items()}
        self.user_train_ts = user_train_ts
        self.user_train_count = user_train_count
        self.user_train_age = user_train_age

    def _pad(self, seq):
        """Left-pad sequence to maxlen."""
        seq = seq[-self.maxlen:] if len(seq) > self.maxlen else seq
        pad_len = self.maxlen - len(seq)
        return np.array([0] * pad_len + list(seq), dtype=np.int32)

    def _pad_float(self, seq):
        """Left-pad float sequence (timestamps/popularity) to maxlen."""
        seq = seq[-self.maxlen:] if len(seq) > self.maxlen else seq
        pad_len = self.maxlen - len(seq)
        return np.array([0.0] * pad_len + list(seq), dtype=np.float32)

    def _sample_neg(self, user_id):
        """Sample negative items for user."""
        pos_set = self.user_pos_set[user_id]
        neg = np.zeros(self.maxlen, dtype=np.int32)
        for i in range(self.maxlen):
            t = np.random.randint(1, self.num_items + 1)
            while t in pos_set:
                t = np.random.randint(1, self.num_items + 1)
            neg[i] = t
        return neg

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        user_id = self.user_list[index]
        items = self.user_train[user_id]

        seq = self._pad(items[:-1])
        pos = self._pad(items[1:])
        neg = self._sample_neg(user_id)

        # Position 4: timestamps
        if self.user_train_ts is not None:
            ts = self.user_train_ts[user_id]
            seq_ts = torch.from_numpy(self._pad_float(ts[:-1]))
        else:
            seq_ts = torch.zeros(self.maxlen, dtype=torch.float32)

        # Position 5: cumulative count (Dynamic C)
        if self.user_train_count is not None:
            count = self.user_train_count[user_id]
            seq_count = torch.from_numpy(self._pad_float(count[:-1]))
        else:
            seq_count = torch.zeros(self.maxlen, dtype=torch.float32)

        # Position 6: pre-computed age (Dynamic A)
        if self.user_train_age is not None:
            age = self.user_train_age[user_id]
            seq_age = torch.from_numpy(self._pad_float(age[:-1]))
        else:
            seq_age = torch.zeros(self.maxlen, dtype=torch.float32)

        return (torch.LongTensor([user_id]), torch.from_numpy(seq),
                torch.from_numpy(pos), torch.from_numpy(neg),
                seq_ts, seq_count, seq_age)


def temporal_split(user_items, user_ts, user_count, user_age, train_ratio, val_ratio):
    """Split data by temporal cutoff into train/val/test.

    Two cutoffs divide the global time range based on train/val/test ratios.
    """
    all_ts = [ts for ts_list in user_ts.values() for ts in ts_list]
    if not all_ts:
        raise ValueError("No timestamp data found")

    min_time, max_time = min(all_ts), max(all_ts)
    time_range = max_time - min_time
    val_cutoff = min_time + train_ratio * time_range
    test_cutoff = min_time + (train_ratio + val_ratio) * time_range

    user_train, user_val, user_test = {}, {}, {}
    user_train_ts = {}
    user_train_count = {}
    user_train_age = {}

    for user_id in user_items:
        if user_id not in user_ts:
            continue

        items = user_items[user_id]
        timestamps = user_ts[user_id]
        counts = user_count.get(user_id, [0] * len(items))
        ages = user_age.get(user_id, [0] * len(items))

        train_mask = [ts < val_cutoff for ts in timestamps]
        train_items = [item for item, m in zip(items, train_mask) if m]
        train_ts = [ts for ts, m in zip(timestamps, train_mask) if m]
        train_counts = [c for c, m in zip(counts, train_mask) if m]
        train_ages = [a for a, m in zip(ages, train_mask) if m]

        val_items = [item for item, ts in zip(items, timestamps) if val_cutoff <= ts < test_cutoff]
        test_items = [item for item, ts in zip(items, timestamps) if ts >= test_cutoff]

        if train_items:
            user_train[user_id] = train_items
            user_train_ts[user_id] = train_ts
            user_train_count[user_id] = train_counts
            user_train_age[user_id] = train_ages

            user_val[user_id] = val_items
            user_test[user_id] = test_items

    # Filter cold items from val and test
    train_items_set = {item for seq in user_train.values() for item in seq}
    for user_id in user_val:
        user_val[user_id] = [i for i in user_val[user_id] if i in train_items_set]
    for user_id in user_test:
        user_test[user_id] = [i for i in user_test[user_id] if i in train_items_set]

    return (user_train, user_val, user_test, user_train_ts, user_train_count, user_train_age)


def get_data_loaders(data_path, args):
    """Load data and create data loaders."""
    (user_items, num_users, num_items, user_ts, user_count, user_age) = load_data(data_path)

    (user_train, user_val, user_test, user_train_ts, user_train_count, user_train_age) = temporal_split(
        user_items, user_ts, user_count, user_age,
        args.train_ratio, args.val_ratio)

    # Print summary only once
    train_n = sum(len(s) for s in user_train.values())
    val_n = sum(len(s) for s in user_val.values())
    test_n = sum(len(s) for s in user_test.values())
    print_once(f"Data: {num_users:,} users, {num_items:,} items | Train: {train_n:,}, Val: {val_n:,}, Test: {test_n:,}")

    # Check which dynamic features are needed
    t_mode = getattr(args, 't_mode', 'lnAC_WHYM')
    need_age = 'A' in t_mode   # A=age (pre-computed)
    need_count = 'C' in t_mode  # C=count (pre-computed)

    # Create dataset and loader
    train_dataset = SeqDataset(user_train, args.maxlen, num_items,
                               user_train_ts=user_train_ts,
                               user_train_count=user_train_count if need_count else None,
                               user_train_age=user_train_age if need_age else None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load pre-computed item temporal features
    item_features = get_item_features(data_path, num_items)

    return train_loader, {
        'user_train': user_train,
        'user_val': user_val,
        'user_test': user_test,
        'user_train_ts': user_train_ts,
        'user_train_count': user_train_count,  # Dynamic C: cumulative count per interaction
        'user_train_age': user_train_age,  # Dynamic A: item age per interaction
        'num_users': num_users,
        'num_items': num_items,
        'item_features': item_features,
    }
