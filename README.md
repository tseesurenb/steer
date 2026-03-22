# STEER: Sequential Temporal Encoding with Feature Routing

STEER integrates item lifecycle dynamics into sequential recommendation via selective QKV-level routing. Each temporal feature is assigned to exactly one attention component (Query, Key, or Value) through a learnable top-1 routing mechanism, achieving feature-role disentanglement that prevents temporal interference in attention computation.

## Setup

```bash
pip install torch numpy tqdm wandb
```

## Quick Start

```bash
# Train STEER on ML-1M
python src/main.py --model steer --dataset ml-1m --epochs 45

# Train SASRec+Temporal baseline
python src/main.py --model sasrec --dataset ml-1m --epochs 100

# Benchmark with multiple seeds (mean +/- std)
python src/benchmark.py --model steer --dataset ml-1m --runs 5
```

## Best Configurations

### STEER

```bash
# ML-1M (dense, 165 interactions/user)
python src/benchmark.py --model steer --dataset ml-1m \
  --lr 0.0003 --l2_decay 0.001 --dropout 0.7 \
  --maxlen 35 --blocks 2 --heads 2 --epochs 45 \
  --runs 5 --t_mode lnAC_WHYM

# Amazon Beauty (sparse, 50 interactions/user)
python src/benchmark.py --model steer --dataset beauty \
  --lr 0.0001 --l2_decay 0.001 --dropout 0.1 \
  --maxlen 15 --blocks 2 --heads 1 --epochs 200 \
  --runs 5 --t_mode lnAC_WHYM

# Amazon Health (sparse, 44 interactions/user)
python src/benchmark.py --model steer --dataset health \
  --lr 0.0001 --l2_decay 0.01 --dropout 0.7 \
  --maxlen 10 --blocks 2 --heads 1 --epochs 70 \
  --runs 5 --t_mode lnAC_WHYM

# 30Music (dense, 185 interactions/user)
python src/benchmark.py --model steer --dataset 30m \
  --lr 0.001 --l2_decay 0.0001 --dropout 0.65 \
  --maxlen 200 --blocks 2 --heads 1 --epochs 200 \
  --runs 5 --t_mode lnAC_WHYM
```

### SASRec+Temporal (input-level temporal features, no routing)

```bash
python src/benchmark.py --model sasrec --dataset ml-1m \
  --lr 0.0001 --l2_decay 0.001 --dropout 0.5 \
  --maxlen 35 --blocks 2 --heads 2 --epochs 100 \
  --runs 5 --t_mode lnAC_WHYM
```

## Temporal Features (`--t_mode`)

Features are specified as a string. Default: `lnAC_WHYM`

**Static features** (computed once per item from training data):

| Code | Description | Encoding |
|------|-------------|----------|
| `f` | First interaction timestamp | Sinusoidal |
| `l` | Last interaction timestamp | Sinusoidal |
| `n` | Total interaction count (popularity) | Log-linear |

**Dynamic features** (computed per interaction):

| Code | Description | Encoding |
|------|-------------|----------|
| `A` | Item age at interaction time | Log-linear |
| `C` | Cumulative count at interaction time | Log-linear |

**Cyclic features** (specified after `_`):

| Code | Description | Period |
|------|-------------|--------|
| `H` | Hour of day | 24 |
| `W` | Day of week | 7 |
| `M` | Month of year | 12 |
| `Y` | Day of year | 365 |

Examples: `lnAC_WHYM` (all features), `ln` (static only), `lnAC` (no cyclic)

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | `steer` or `sasrec` | `steer` |
| `--dataset` | `ml-1m`, `beauty`, `health`, `30m` | `ml-1m` |
| `--t_mode` | Temporal feature string | `lnAC_WHYM` |
| `--dim` | Embedding dimension | `64` |
| `--blocks` | Transformer blocks | `2` |
| `--heads` | Attention heads | `1` |
| `--maxlen` | Max sequence length | `35` |
| `--lr` | Learning rate | `0.0001` |
| `--dropout` | Dropout rate | `0.7` |
| `--l2_decay` | L2 regularization | `0.001` |
| `--epochs` | Training epochs | `100` |
| `--batch_size` | Batch size | `512` |
| `--runs` | Number of runs for benchmarking | `3` |
| `--uniform_routing` | Use uniform 1/3 routing (ablation) | `false` |
| `--no_pos` | Disable positional encoding | `false` |
| `--wandb` | Enable W&B logging | `false` |

## Data Format

Processed data in `data/gold/{dataset}/{dataset}-time.txt` (space-separated):

```
UserID ItemID Timestamp CumulativeCount Age
```

Item features are precomputed in `data/gold/{dataset}/item_features.txt`.

Preprocessing scripts are in `data/prep/`.

## Project Structure

```
src/
  main.py          # Training with validation monitoring
  benchmark.py     # Multi-run benchmarking (mean +/- std)
  config.py        # All hyperparameters
  data.py          # Data loading and temporal splitting
  trainer.py       # Training loop
  evaluator.py     # NDCG@k, HR@k, Recall@k
  search.py        # Hyperparameter search
  utils.py         # Utilities
  models/
    steer.py       # STEER (top-1 routing to Q/K/V)
    sasrec.py      # SASRec + temporal features
    base.py        # Base model and loss mixins
data/
  gold/            # Processed datasets
  prep/            # Preprocessing scripts
```

## Evaluation

- Temporal split: train (90%) / val (5%) / test (5%) by timestamp
- Cold items filtered from val/test
- Full ranking over all items (no sampling)
- Metrics: HR@k, NDCG@k, Recall@k for k in {5, 10, 20}

## Citation

```
[TODO]
```
