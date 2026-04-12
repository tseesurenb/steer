# Dataset Preprocessing Principles

This document defines the preprocessing pipeline for sequential recommendation datasets. Use this as a reference when preprocessing new datasets.

## Pipeline Overview

```
Bronze (raw) → Silver (cleaned) → Gold (ready for training)
```

## Dataset-Specific Approaches

Not all datasets require the same preprocessing:

| Dataset | Time Filter | User Core | Item Core | Notes |
|---------|-------------|-----------|-----------|-------|
| ML-1M | No | No | No | Used as-is |
| AWS-Beauty | Yes (2020-2022) | Yes (20) | No | Time window + user filter |
| AWS-Health | Yes (2020-2022) | Yes (20) | No | Time window + user filter |
| 30music | No | Yes (20) | Yes (150) | User + item filter |

**Choose preprocessing based on dataset characteristics:**
- Clean datasets (ML-1M): Use as-is
- Large time span: Apply time window
- Many inactive users: Apply user core filter
- Too sparse: May need item core filter (use cautiously)

---

## Stage 1: Bronze → Silver

### Step 1: Time Filtering (OPTIONAL)
- Select a time window (e.g., 2020-2022)
- This is normal data selection
- Skip if dataset time span is reasonable

### Step 2: Sort and Compute Train Cutoff
- Sort all interactions by timestamp (chronological order)
- Compute train cutoff: `t_cutoff = t_min + (t_max - t_min) * 0.90`

### Step 3: Compute Item Dynamics from Train Portion ⭐ CRUCIAL
**This step must happen BEFORE any user/item filtering.**

For interactions where `ts < t_cutoff` (train portion only):
- `first_ts`: First timestamp of item
- `last_ts`: Last timestamp of item in train
- `cumulative_count`: Running count at each interaction

For ALL interactions:
- Record current `cumulative_count` (0 if item not seen in train yet)

**Save item features file immediately after this step.**

### Step 4: Apply User Core Filter (OPTIONAL)
- Count interactions per user (from ALL data, not just train)
- Keep users with ≥ N interactions (e.g., N=20)
- **This does NOT recompute item stats** - preserves item dynamics
- Skip if dataset already has active users

### Step 5: Write Silver Output
- Filter interactions to valid users only (if user core applied)
- Format: `UserID ItemID Timestamp CumulativeCount`
- Also save: `{dataset}_item_features.txt`

---

## Stage 2: Silver → Gold

### Step 1: Load Item Features
- Load `first_ts` per item from item_features file

### Step 2: Load Silver Data
- Load all interactions (already user-filtered)

### Step 3: Sort by Timestamp
- Ensure chronological order

### Step 4: Reindex IDs
- Map original IDs to contiguous integers (1 to N)
- IDs assigned in order of first appearance

### Step 5: Compute Age
- `age = timestamp - first_ts`
- Cold items (not in train): `age = 0`

### Step 6: Write Gold Output
- Format: `UserID ItemID Timestamp CumulativeCount Age`

---

## Critical Design Principles

### 1. Order of Operations Matters

When preprocessing steps are applied, the ORDER is crucial:

| Order | Step | Rationale |
|-------|------|-----------|
| 1 | Time filtering (if needed) | Normal - select observation window |
| 2 | Compute item stats | Item dynamics reflect true temporal patterns |
| 3 | User/item core filter (if needed) | Does NOT distort already-computed item dynamics |

### 2. Item Stats BEFORE Any Filtering

**Why this order is crucial:**
- Item dynamics (`first_ts`, `count`) capture real-world item lifecycle
- User filtering removes users but should NOT change item statistics
- If we filter users first, item stats would be computed on biased subset

### 3. Item Core Filtering (Use Cautiously)

- Prefer NO item filtering when possible - preserves item temporal dynamics
- If dataset is too sparse, item core may be necessary
- When applied: still compute item stats BEFORE filtering
- Distortion impact: item filtering > user filtering

### 4. Train-Only for Features

- Item dynamics computed from train portion only (first 90% by time)
- Prevents data leakage to validation/test sets
- Val/test items not seen in train get neutral values (age=0, count=0)

---

## Output Formats

### Silver Main File (`silver/{dataset}.txt`)
```
UserID ItemID Timestamp CumulativeCount
user123 itemABC 1609459200 5
user123 itemXYZ 1609545600 12
```

### Silver Item Features (`silver/{dataset}_item_features.txt`)
```
T_min T_max T_cutoff
1609459200 1640995199 1637875199
ItemID FirstTimestamp LastTimestamp NumInteractions
itemABC 1609459200 1635724800 127
itemXYZ 1612137600 1638316800 45
```

### Gold File (`gold/{dataset}/{dataset}-time.txt`)
```
UserID ItemID Timestamp CumulativeCount Age
1 42 1609459200 5 0
1 17 1609545600 12 86400
```

---

## Checklist for New Datasets

**Required steps:**
- [ ] Data sorted by timestamp
- [ ] Train cutoff computed (90% by time range)
- [ ] Item dynamics computed from TRAIN portion only
- [ ] Item features saved BEFORE any filtering
- [ ] IDs reindexed to contiguous integers
- [ ] Age computed using first_ts from train
- [ ] Cold items handled (age=0, count=0)

**Optional steps (apply if needed):**
- [ ] Time filter applied first (if dataset spans too long)
- [ ] User core filter applied AFTER item stats (if many inactive users)
- [ ] Item core filter applied AFTER item stats (if too sparse, use cautiously)

**Order verification:**
- [ ] Item stats computed BEFORE any user/item filtering
