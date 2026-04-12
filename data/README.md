# Data Preprocessing Guide

## Overview

Three-stage pipeline: **Bronze → Silver → Gold**

```
bronze/              Raw dataset files (original format)
    ↓
silver/              Intermediate format (UserID ItemID Timestamp CumulativeCount)
    ↓
gold/                Final format (UserID ItemID Timestamp CumulativeCount Age)
```

## Folder Structure

```
data/
├── bronze/                    # Raw dataset files
│   ├── ml-1m/                 # MovieLens 1M
│   ├── aws-beauty/            # Amazon Beauty
│   ├── aws-health/            # Amazon Health
│   ├── steam/                 # Steam Reviews
│   └── yelp/                  # Yelp Reviews
├── silver/                    # Intermediate processed
│   ├── {dataset}.txt          # Main data file
│   └── {dataset}_item_features.txt  # Item temporal features
├── gold/                      # Final output
│   └── {dataset}/
│       └── {dataset}-time.txt
└── prep/                      # Preprocessing scripts
    ├── to_silver_ml1m.py
    ├── to_silver_aws_beauty.py
    ├── to_silver_aws_health.py
    ├── to_silver_steam.py
    ├── to_silver_yelp.py
    └── to_gold.py
```

---

## Stage 1: Bronze → Silver

### Processing Steps (in order)

| Pass | Operation | Description |
|------|-----------|-------------|
| 1 | Year Filter | Keep only interactions within specified year range |
| 2 | Sort + Train Cutoff | Sort by timestamp, compute 90% train cutoff |
| 3 | Item Dynamics | Compute `first_ts`, `last_ts`, `cumulative_count` from TRAIN only |
| 4 | User Core Filter | Keep users with ≥N interactions |
| 5 | Write Output | Save filtered data |

### Important Design Decisions

**Item dynamics computed BEFORE user core filter:**
```
Pass 3: Item dynamics ← Uses ALL users (before filtering)
Pass 4: User core     ← Filters users AFTER item dynamics
```

This means `cumulative_count` reflects true item popularity across all users, not just core users.

### Pass 3: Item Dynamics (Exact Code)

```python
for user_id, item_id, ts in data:  # data = ALL users after year filter
    if ts < t_cutoff:              # Only from TRAIN portion
        if item_id not in item_first_ts:
            item_first_ts[item_id] = ts
        item_last_ts[item_id] = ts
        item_cumcount[item_id] += 1

    count = item_cumcount[item_id]
    data_with_counts.append((user_id, item_id, ts, count))
```

### Pass 4: User Core Filter (Exact Code)

```python
user_counts = defaultdict(int)
for user_id, _, _, _ in data_with_counts:
    user_counts[user_id] += 1

valid_users = {u for u, c in user_counts.items() if c >= args.user_core}
filtered = [(u, i, ts, cnt) for u, i, ts, cnt in data_with_counts if u in valid_users]
```

### Silver Output Format

**Main file:** `silver/{dataset}.txt`
```
UserID ItemID Timestamp CumulativeCount
```

**Item features:** `silver/{dataset}_item_features.txt`
```
Line 1: T_min T_max T_train_cutoff
Line 2+: ItemID FirstTimestamp LastTimestamp NumInteractions
```

---

## Stage 2: Silver → Gold

### Processing Steps

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | Load item features | Read `first_ts` from item_features file |
| 2 | Load silver data | Read all interactions |
| 3 | Sort by timestamp | Chronological order |
| 4 | Reindex IDs | Map to 1...N (0 reserved for padding) |
| 5 | Compute Age | `age = timestamp - first_ts` (cold items get 0) |
| 6 | Write output | Save to gold format |

### Gold Output Format

**File:** `gold/{dataset}/{dataset}-time.txt`
```
UserID ItemID Timestamp CumulativeCount Age
```

| Column | Description |
|--------|-------------|
| UserID | Reindexed user ID (1 to N) |
| ItemID | Reindexed item ID (1 to N) |
| Timestamp | Unix timestamp (seconds) |
| CumulativeCount | Item interaction count at this time (from train) |
| Age | `timestamp - first_ts` in seconds (0 for cold items) |

---

## Dataset-Specific Commands

### ML-1M (No user filtering)

```bash
cd data
python prep/to_silver_ml1m.py
python prep/to_gold.py ml-1m
```
- No year filter (uses all data)
- No user core filter (keeps all users)

### AWS-Beauty

```bash
cd data
python prep/to_silver_aws_beauty.py --start_year 2020 --end_year 2022 --user_core 20
python prep/to_gold.py aws-beauty
```

### AWS-Health

```bash
cd data
python prep/to_silver_aws_health.py --start_year 2020 --end_year 2022 --user_core 20
python prep/to_gold.py aws-health
```

### Steam

```bash
cd data
# Standard (20-core, 1 year)
python prep/to_silver_steam.py --start_year 2023 --end_year 2023 --user_core 20
python prep/to_gold.py steam

# Dense version (30-core)
python prep/to_silver_steam.py --start_year 2023 --end_year 2023 --user_core 30 --output steam_u30
python prep/to_gold.py steam_u30
```

### Yelp

```bash
cd data
python prep/to_silver_yelp.py --start_year 2019 --end_year 2019 --user_core 20
python prep/to_gold.py yelp
```

---

## Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--start_year` | Start year (inclusive) | Dataset-specific |
| `--end_year` | End year (inclusive) | Dataset-specific |
| `--user_core` | Min interactions per user | 20 |
| `--output` | Output name (for variants) | Dataset name |

---

## Data Leakage Prevention

### What we do:
1. **Train/Val/Test split by time** (90/5/5 by default)
2. **Item dynamics from TRAIN only** - `cumulative_count` and `first_ts` computed only from train portion
3. **Cold items get age=0** - Items not seen in train have no temporal features

### What this means:
- Val/Test items that weren't in train get `cumulative_count=0` and `age=0`
- No future information leaks into training

---

## Example Data Flow

**Raw (Bronze):**
```json
{"user_id": "A1B2C3", "asin": "B00XYZ", "timestamp": 1640000000000, ...}
```

**Silver:**
```
A1B2C3 B00XYZ 1640000000 5
```
(UserID, ItemID, Timestamp in seconds, CumulativeCount=5)

**Gold:**
```
1 1 1640000000 5 86400
```
(Reindexed to 1, 1, same timestamp, count=5, age=1 day)

---

## Dataset Statistics

| Dataset | Users | Items | Interactions | Density | Avg/User |
|---------|-------|-------|--------------|---------|----------|
| ML-1M | 6,040 | 3,706 | 1,000,209 | 4.47% | 165.6 |
| AWS-Beauty | ~50K | ~50K | ~1M | ~0.04% | ~20 |
| AWS-Health | ~40K | ~40K | ~800K | ~0.05% | ~20 |
| Steam | 20,816 | 16,876 | 1,016,492 | 0.29% | 48.8 |
| Steam_u30 | 11,150 | 16,403 | 788,200 | 0.43% | 70.7 |
| Yelp | ~10K | ~15K | ~300K | ~0.2% | ~30 |

---

## Verification

After preprocessing, verify with:

```bash
# Check file format
head -5 gold/{dataset}/{dataset}-time.txt

# Check statistics
awk '{users[$1]++; items[$2]++} END {
  print "Users:", length(users)
  print "Items:", length(items)
  print "Interactions:", NR
}' gold/{dataset}/{dataset}-time.txt
```
