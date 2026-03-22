"""Standardized Hyperparameter Search for STEER/SASRec.

Fixed: dim=64, batch_size=512, t_mode=lnAC_WHYM

Default values: lr=0.0001, l2_decay=0, dropout=0.1

Search Stages (4 stages):
1. maxlen: adaptive [10, 20, 30, 50, 100, 200] - stop after 2 non-improvements + midpoint refinement
2. lr: learning rate + midpoint refinement
3. l2_decay: weight decay
4. dropout: regularization (with extension if 0.5 best) + midpoint refinement

Fixed: blocks=2, heads=1

Early Stopping (within each training run):
- Monitor: Validation NDCG@10
- Warmup: 20 epochs (dips ignored - attention heads may "re-align")
- Min Delta: 1e-4 (ignore tiny fluctuations after warmup)
- Patience: 100 epochs (ml-1m, 30m) or 50 epochs (beauty, health)
- Max Epochs: 200 (hard cap)

Features:
- File-based JSON cache (no repeat configs)
- Early stopping based on validation NDCG@10
- Dropout extension (try 0.6, 0.7... if 0.5 best)
- Progress indicators with colors

Usage:
    python src/search.py --model steer --dataset ml-1m
    python src/search.py --model sasrec --dataset ml-1m --runs 3
    python src/search.py --model steer --dataset ml-1m --lr 0.0001  # skip lr search
"""
import os
import sys
import json
import copy
import argparse
import numpy as np
from datetime import datetime

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data import get_data_loaders
from evaluator import evaluate
from trainer import Trainer
from utils import set_seed, get_device, get_model, setup_model

_CACHE_DIR = os.path.join(_SRC_DIR, '..', 'runs', 'search_cache')
_RUNS_DIR = os.path.join(_SRC_DIR, '..', 'runs', 'search')

# Colors (dark colors for white terminal background)
BOLD = '\033[1m'
GREEN = '\033[32m'   # dark green
BLUE = '\033[34m'    # dark blue
MAGENTA = '\033[35m' # dark magenta
RED = '\033[31m'     # dark red
DIM = '\033[2m'
RESET = '\033[0m'

# =============================================================================
# Search Spaces
# =============================================================================

# Standard search space (non-extended)
SEARCH_SPACE = {
    'lr': [0.001, 0.0005, 0.0001],
    'l2_decay': [0.001, 0.0001, 0],
    'dropout': [0.1, 0.3, 0.5],
    'maxlen': [10, 20, 30, 50, 100, 200],
    'epochs': [50, 100, 200],
    'blocks': [1, 2],
    'heads': [1, 2],  # must divide dim=64
}

# Extended search space (--ext flag)
SEARCH_SPACE_EXT = {
    'lr': [0.001, 0.0005, 0.0001],
    'l2_decay': [0.001, 0.0001, 0],
    'dropout': [0.1, 0.3, 0.5],
    'maxlen': [10, 20, 30, 50, 100, 200],
    'epochs': [50, 100, 200, 400],
    'blocks': [1, 2, 3],
    'heads': [1, 2],
}

# Model-specific overrides (for extended mode)
SEARCH_SPACE_STEER = {
    'lr': [0.001, 0.0005, 0.0001],
    'maxlen': [10, 20, 30, 50, 100, 200],
}

# Adaptive search parameters (applies to all datasets)
ADAPTIVE_MAXLEN = {'start': 25, 'step': 5, 'max_val': 200}
ADAPTIVE_EPOCHS = {'start': 50, 'step': 50, 'max_val': 500}

# Early stopping parameters
EARLY_STOPPING = {
    'max_epochs': 200,      # Hard cap
    'warmup': 20,           # No early stopping during warmup (attention re-aligning)
    'patience': {           # Dataset-specific patience (in epochs)
        'ml-1m': 100,
        '30m': 100,
        'beauty': 50,
        'health': 50,
    },
    'default_patience': 50,
    'min_delta': {          # Dataset-specific min_delta (sparse datasets have lower NDCG)
        'ml-1m': 1e-4,      # NDCG ~0.07, so 1e-4 = ~0.14% relative
        '30m': 1e-4,        # NDCG ~0.06, so 1e-4 = ~0.17% relative
        'beauty': 1e-5,     # NDCG ~0.004, so 1e-5 = ~0.25% relative
        'health': 1e-5,     # NDCG ~0.004, so 1e-5 = ~0.25% relative
    },
    'default_min_delta': 1e-4,
}

DEFAULTS = {
    'lr': 0.0001,
    'l2_decay': 1e-5,       # Small baseline regularization (not 0)
    'dropout': 0.1,
    'blocks': 2,
    'heads': 1,
}

# Note: maxlen is always searched first (Stage 1), epochs use early stopping



def get_search_space(param, model='sasrec', extended=False, dataset=None):
    """Get search space for a parameter, with model-specific overrides.

    Note: maxlen and epochs use adaptive search, not fixed grids.
    """
    # Use extended space if flag is set
    base_space = SEARCH_SPACE_EXT if extended else SEARCH_SPACE

    # Model-specific overrides (only in extended mode for STEER)
    if extended and model == 'steer' and param in SEARCH_SPACE_STEER:
        return SEARCH_SPACE_STEER[param]
    return base_space.get(param, None)


# =============================================================================
# Cache Management
# =============================================================================

def get_cache_path(model, dataset):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f'{model}_{dataset}.json')


def load_cache(model, dataset):
    path = get_cache_path(model, dataset)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def save_cache(model, dataset, cache):
    path = get_cache_path(model, dataset)
    with open(path, 'w') as f:
        json.dump(cache, f, indent=2)


def config_to_key(config, num_runs=1):
    """Include num_runs in key so runs=1 and runs=3 are cached separately."""
    config_with_runs = {**config, '_runs': num_runs}
    items = sorted(config_with_runs.items())
    return json.dumps(items)


# =============================================================================
# Logger
# =============================================================================

class Logger:
    def __init__(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_and_eval(args, device, cache, num_runs=1):
    """Train model and return validation metrics with early stopping."""
    config = {
        'lr': args.lr,
        'l2_decay': args.l2_decay,
        'dropout': args.dropout,
        'maxlen': args.maxlen,
        'epochs': args.epochs,
        'blocks': args.blocks,
        'heads': args.heads,
        't_mode': args.t_mode,
    }
    cache_key = config_to_key(config, num_runs)

    # Check cache
    if cache_key in cache:
        result = cache[cache_key]
        print(f" {DIM}[CACHED]{RESET} NDCG@10={RED}{result['ndcg']:.4f}{RESET}  HR@10={RED}{result['hr']:.4f}{RESET}", end="")
        return result

    # Early stopping parameters (dataset-specific)
    patience = EARLY_STOPPING['patience'].get(args.dataset, EARLY_STOPPING['default_patience'])
    min_delta = EARLY_STOPPING['min_delta'].get(args.dataset, EARLY_STOPPING['default_min_delta'])
    max_epochs = EARLY_STOPPING['max_epochs']
    warmup = EARLY_STOPPING['warmup']

    all_ndcg = []
    all_hr = []
    all_best_epochs = []

    for run_idx in range(num_runs):
        seed = args.seed + run_idx
        set_seed(seed)

        # Show run progress
        if num_runs > 1:
            print(f"\n      Run {run_idx+1}/{num_runs}: ", end="", flush=True)
        else:
            print(" ", end="", flush=True)

        # Load data
        data_path = os.path.join('data', 'gold', args.dataset_folder, f'{args.dataset_folder}-time.txt')
        train_loader, dataset_info = get_data_loaders(data_path, args)
        args.num_users = dataset_info['num_users']

        # Build model
        model = get_model(args.model, dataset_info['num_items'], args).to(device)
        setup_model(model, args, dataset_info, device)

        trainer = Trainer(model, train_loader, args, device)
        user_val = dataset_info['user_val']

        best_ndcg = 0.0
        best_hr = 0.0
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            trainer.train_epoch()

            if epoch % args.eval == 0:
                ndcg, hr, _ = evaluate(model, dataset_info, args.maxlen, device, user_eval=user_val)

                # During warmup: track best but ignore dips (attention heads may "re-align")
                # After warmup: start counting patience for early stopping
                if epoch <= warmup:
                    # Warmup phase: only track improvements, ignore dips
                    if ndcg[10] > best_ndcg:
                        best_ndcg = ndcg[10]
                        best_hr = hr[10]
                        best_epoch = epoch
                        print(f"{GREEN}●{RESET}", end="", flush=True)
                    else:
                        print(f"{DIM}w{RESET}", end="", flush=True)  # 'w' = warmup, dip ignored
                else:
                    # Post-warmup: apply early stopping logic with min_delta
                    if ndcg[10] > best_ndcg + min_delta:
                        best_ndcg = ndcg[10]
                        best_hr = hr[10]
                        best_epoch = epoch
                        epochs_without_improvement = 0
                        print(f"{GREEN}●{RESET}", end="", flush=True)
                    else:
                        epochs_without_improvement += args.eval
                        print("·", end="", flush=True)

                    # Early stopping check
                    if epochs_without_improvement >= patience:
                        print(f"{DIM}[ES@{epoch}]{RESET}", end="", flush=True)
                        break

        all_ndcg.append(best_ndcg)
        all_hr.append(best_hr)
        all_best_epochs.append(best_epoch)

        if num_runs > 1:
            print(f" NDCG@10={RED}{best_ndcg:.4f}{RESET} HR@10={RED}{best_hr:.4f}{RESET} (ep={best_epoch})", end="", flush=True)

    avg_ndcg = np.mean(all_ndcg)
    avg_hr = np.mean(all_hr)
    avg_epoch = np.mean(all_best_epochs)

    result = {'ndcg': avg_ndcg, 'hr': avg_hr, 'best_epoch': avg_epoch}
    cache[cache_key] = result

    if num_runs > 1:
        print(f"\n      Avg: NDCG@10={RED}{avg_ndcg:.4f}{RESET}  HR@10={RED}{avg_hr:.4f}{RESET} (ep={avg_epoch:.0f})", end="")
    else:
        print(f"NDCG@10={RED}{avg_ndcg:.4f}{RESET}  HR@10={RED}{avg_hr:.4f}{RESET}", end="")

    return result


# =============================================================================
# Search Functions
# =============================================================================

def search_grid(args, device, cache, param_names, values_list, num_runs, prefer=None):
    """Search multiple parameters jointly.

    Args:
        prefer: dict of preferred values for tie-breaking (e.g., {'blocks': 2, 'heads': 1})
    """
    from itertools import product
    all_combos = list(product(*values_list))

    print(f"\n  Searching {param_names}: {len(all_combos)} configs")
    print("  " + "-" * 50)

    results = []
    best_ndcg_so_far = 0.0

    for combo in all_combos:
        for name, val in zip(param_names, combo):
            setattr(args, name, val)

        param_str = ", ".join(f"{n}={v}" for n, v in zip(param_names, combo))
        print(f"    {param_str}:", end="")
        res = train_and_eval(args, device, cache, num_runs)
        results.append({**dict(zip(param_names, combo)), 'ndcg': res['ndcg'], 'hr': res['hr']})

        if res['ndcg'] > best_ndcg_so_far:
            best_ndcg_so_far = res['ndcg']
            print(f" {GREEN}★ NEW BEST{RESET}")
        else:
            print()

    # Find best, with tie-breaking using preferred values
    best = max(results, key=lambda x: x['ndcg'])
    best_ndcg = best['ndcg']

    # Check for ties and apply preference
    if prefer:
        tied = [r for r in results if abs(r['ndcg'] - best_ndcg) < 1e-6]
        if len(tied) > 1:
            # Sort by preference (prefer values closer to preferred)
            def preference_score(r):
                score = 0
                for name, pref_val in prefer.items():
                    if name in r and r[name] == pref_val:
                        score += 1
                return score
            tied.sort(key=preference_score, reverse=True)
            best = tied[0]
            print(f"  {DIM}(Tie-break: preferring {prefer}){RESET}")

    best_str = ", ".join(f"{n}={best[n]}" for n in param_names)
    print(f"\n  {GREEN}{BOLD}▶ Best: {best_str}{RESET}")
    print(f"    {MAGENTA}NDCG@10={best['ndcg']:.4f}  HR@10={best['hr']:.4f}{RESET}")

    return {n: best[n] for n in param_names}, best


def search_param(args, device, cache, param_name, values, num_runs, extend=False):
    """Search single parameter with optional extension."""
    print(f"\n  Searching {param_name}: {values}")
    print("  " + "-" * 50)

    results = []
    best_ndcg_so_far = 0.0

    for val in values:
        setattr(args, param_name, val)
        print(f"    {param_name}={val}:", end="")
        res = train_and_eval(args, device, cache, num_runs)
        results.append({'value': val, 'ndcg': res['ndcg'], 'hr': res['hr']})

        if res['ndcg'] > best_ndcg_so_far:
            best_ndcg_so_far = res['ndcg']
            print(f" {GREEN}★ NEW BEST{RESET}")
        else:
            print()

    best_idx = max(range(len(results)), key=lambda i: results[i]['ndcg'])
    best = results[best_idx]

    # Extension: if best is at max, try higher (for dropout)
    if extend and best_idx == len(results) - 1:
        current_val = best['value']
        while current_val < 0.9:
            new_val = round(current_val + 0.1, 1)
            setattr(args, param_name, new_val)
            print(f"    {param_name}={new_val} (extended):", end="")
            res = train_and_eval(args, device, cache, num_runs)

            if res['ndcg'] > best['ndcg']:
                best = {'value': new_val, 'ndcg': res['ndcg'], 'hr': res['hr']}
                current_val = new_val
                print(f" {GREEN}★ NEW BEST{RESET}")
            else:
                print()
                break

    print(f"\n  {GREEN}{BOLD}▶ Best {param_name}: {best['value']}{RESET}")
    print(f"    {MAGENTA}NDCG@10={best['ndcg']:.4f}  HR@10={best['hr']:.4f}{RESET}")
    return best['value'], best


def search_maxlen_adaptive(args, device, cache, num_runs, values=[10, 20, 30, 50, 100, 200]):
    """Search maxlen adaptively: stop after 2 consecutive non-improvements.

    Only checks larger values if trend is improving. For example:
    - If 20 is best and 30, 50 are worse → stops, doesn't check 100, 200
    - If 10, 20, 30 keep improving → continues to 50, 100, 200

    After finding best, refines with midpoint search:
    - If best=20 and next=30 → try (20+30)/2 = 25
    - If best=200 (max) → try 250, 300, ... until no improvement
    """
    print(f"\n  Searching maxlen (adaptive): {values}")
    print(f"  Stop after 2 consecutive non-improvements")
    print("  " + "-" * 50)

    results = []
    best_ndcg = 0.0
    best_result = None
    best_idx = 0
    consecutive_non_improvements = 0

    for i, val in enumerate(values):
        args.maxlen = val
        print(f"    maxlen={val}:", end="")
        res = train_and_eval(args, device, cache, num_runs)
        results.append({'value': val, 'ndcg': res['ndcg'], 'hr': res['hr']})

        if res['ndcg'] > best_ndcg:
            best_ndcg = res['ndcg']
            best_result = {'value': val, 'ndcg': res['ndcg'], 'hr': res['hr']}
            best_idx = i
            consecutive_non_improvements = 0
            print(f" {GREEN}★ NEW BEST{RESET}")
        else:
            consecutive_non_improvements += 1
            if consecutive_non_improvements >= 2:
                print(f" {RED}[STOP] 2 consecutive non-improvements{RESET}")
                break
            else:
                print(f" (1 non-improvement)")

    if best_result is None:
        best_result = results[0]

    # Refinement step
    print(f"\n  {BLUE}Refinement:{RESET}")
    best_val = best_result['value']
    max_val = values[-1]

    if best_val == max_val:
        # Best is at max → try extending by 50
        print(f"  Best is at max ({max_val}), trying larger values...")
        refine_val = best_val + 50
        while True:
            args.maxlen = refine_val
            print(f"    maxlen={refine_val} (extend):", end="")
            res = train_and_eval(args, device, cache, num_runs)

            if res['ndcg'] > best_result['ndcg']:
                best_result = {'value': refine_val, 'ndcg': res['ndcg'], 'hr': res['hr']}
                print(f" {GREEN}★ NEW BEST{RESET}")
                refine_val += 50
            else:
                print(f" {RED}[STOP]{RESET}")
                break
    else:
        # Try midpoint between best and next value
        next_idx = best_idx + 1
        if next_idx < len(results):
            next_val = results[next_idx]['value']
            mid_val = (best_val + next_val) // 2
            if mid_val != best_val and mid_val != next_val:
                args.maxlen = mid_val
                print(f"    maxlen={mid_val} (midpoint of {best_val} and {next_val}):", end="")
                res = train_and_eval(args, device, cache, num_runs)

                if res['ndcg'] > best_result['ndcg']:
                    best_result = {'value': mid_val, 'ndcg': res['ndcg'], 'hr': res['hr']}
                    print(f" {GREEN}★ NEW BEST{RESET}")
                else:
                    print(f" (no improvement)")
            else:
                print(f"  No midpoint to try (best={best_val}, next={next_val})")
        else:
            print(f"  No next value for midpoint (best is last tested)")

    print(f"\n  {GREEN}{BOLD}▶ Best maxlen: {best_result['value']}{RESET}")
    print(f"    {MAGENTA}NDCG@10={best_result['ndcg']:.4f}  HR@10={best_result['hr']:.4f}{RESET}")
    return best_result['value'], best_result


def search_epochs_adaptive(args, device, cache, num_runs, start=30, step=10, max_val=300):
    """Search epochs adaptively: stop after two consecutive decreases."""
    print(f"\n  Searching epochs (adaptive): start={start}, step={step}, stop on 2 consecutive decreases")
    print("  " + "-" * 50)

    results = []
    best_ndcg = 0.0
    best_result = None
    consecutive_decreases = 0
    prev_ndcg = 0.0
    val = start

    while val <= max_val:
        args.epochs = val
        print(f"    epochs={val}:", end="")
        res = train_and_eval(args, device, cache, num_runs)
        results.append({'value': val, 'ndcg': res['ndcg'], 'hr': res['hr']})

        if res['ndcg'] > best_ndcg:
            best_ndcg = res['ndcg']
            best_result = {'value': val, 'ndcg': res['ndcg'], 'hr': res['hr']}
            consecutive_decreases = 0
            print(f" {GREEN}★ NEW BEST{RESET}")
        else:
            print()
            if res['ndcg'] < prev_ndcg:
                consecutive_decreases += 1
                if consecutive_decreases >= 2:
                    print(f"    {RED}[STOP] Two consecutive decreases{RESET}")
                    break
            else:
                consecutive_decreases = 0

        prev_ndcg = res['ndcg']
        val += step

    if best_result is None:
        best_result = results[0]

    print(f"\n  {GREEN}{BOLD}▶ Best epochs: {best_result['value']}{RESET}")
    print(f"    {MAGENTA}NDCG@10={best_result['ndcg']:.4f}  HR@10={best_result['hr']:.4f}{RESET}")
    return best_result['value'], best_result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(prog="STEER/SASRec Search")
    parser.add_argument('--model', type=str, required=True, choices=['sasrec', 'steer'])
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--runs', type=int, default=1, help='Runs per config')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)

    # Skip stages
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--l2_decay', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--maxlen', type=int, default=None)
    parser.add_argument('--blocks', type=int, default=None)
    parser.add_argument('--heads', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--t_mode', type=str, default='lnAC_WHYM',
                        help='Temporal features (e.g., lnAC_WHYM, flnAC)')
    parser.add_argument('--ext', action='store_true', help='Use extended search space')

    cmd_args = parser.parse_args()

    # Setup logging: runs/search/{model}/{dataset}/
    log_dir = os.path.join(_RUNS_DIR, cmd_args.model, cmd_args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = Logger(log_file)

    print(f"{'=' * 60}")
    print(f"Search: {cmd_args.model.upper()} on {cmd_args.dataset}")
    print(f"{'=' * 60}")
    print(f"Log: {log_file}")

    device = get_device(cmd_args.device)
    print(f"Device: {device}")
    print(f"Runs per config: {cmd_args.runs}")

    # Load cache
    cache = load_cache(cmd_args.model, cmd_args.dataset)
    print(f"Cached configs: {len(cache)}")

    # Print search space
    ext = cmd_args.ext
    print(f"\n{MAGENTA}Search Space {'(EXTENDED)' if ext else '(STANDARD)'}:{RESET}")
    print(f"  Stage 1 - maxlen:   {get_search_space('maxlen', cmd_args.model, ext)} (adaptive + midpoint refinement)")
    print(f"  Stage 2 - lr:       {get_search_space('lr', cmd_args.model, ext)} + refinement (default: {DEFAULTS['lr']})")
    print(f"  Stage 3 - l2_decay: {get_search_space('l2_decay', cmd_args.model, ext)} (default: {DEFAULTS['l2_decay']})")
    print(f"  Stage 4 - dropout:  {get_search_space('dropout', cmd_args.model, ext)} + extension + refinement (default: {DEFAULTS['dropout']})")
    print(f"  {DIM}Fixed: blocks={DEFAULTS['blocks']}, heads={DEFAULTS['heads']}{RESET}")
    print(f"\n{MAGENTA}Fixed:{RESET}")
    print(f"  dim: 64, batch_size: 512, t_mode: {cmd_args.t_mode}")

    # Early stopping info
    patience = EARLY_STOPPING['patience'].get(cmd_args.dataset, EARLY_STOPPING['default_patience'])
    min_delta = EARLY_STOPPING['min_delta'].get(cmd_args.dataset, EARLY_STOPPING['default_min_delta'])
    print(f"\n{MAGENTA}Early Stopping:{RESET}")
    print(f"  Monitor: Validation NDCG@10")
    print(f"  Warmup: {EARLY_STOPPING['warmup']} epochs (dips ignored)")
    print(f"  Min Delta: {min_delta} (dataset: {cmd_args.dataset})")
    print(f"  Patience: {patience} epochs")
    print(f"  Max Epochs: {EARLY_STOPPING['max_epochs']}")

    # Dataset folder mapping
    dataset_map = {
        'ml-1m': 'ml-1m',
        'beauty': 'aws-beauty',
        'health': 'aws-health',
        '30m': '30m',
    }

    # Build args
    class Args:
        pass

    args = Args()
    args.model = cmd_args.model
    args.dataset = cmd_args.dataset
    args.dataset_folder = dataset_map.get(cmd_args.dataset, cmd_args.dataset)
    args.seed = cmd_args.seed
    args.eval = 5
    args.warmup = 10
    args.split = [90, 5, 5]
    args.train_ratio = 0.9
    args.val_ratio = 0.05

    # Fixed params
    args.dim = 64
    args.batch_size = 512
    args.t_mode = cmd_args.t_mode
    args.t_tau = 86400.0
    args.blocks = 2
    args.heads = 1

    # Defaults (will be updated as search progresses)
    args.lr = DEFAULTS['lr']
    args.l2_decay = DEFAULTS['l2_decay']
    args.dropout = DEFAULTS['dropout']
    args.maxlen = 50  # Temporary, will be set in Stage 1
    args.epochs = EARLY_STOPPING['max_epochs']  # Early stopping handles actual stopping

    # Stage 1: maxlen (architecture - most impactful)
    print(f"\n{'=' * 60}")
    print("STAGE 1: maxlen (adaptive)")
    print('=' * 60)

    if cmd_args.maxlen is not None:
        print(f"  SKIPPED - using maxlen={cmd_args.maxlen}")
        args.maxlen = cmd_args.maxlen
    else:
        args.maxlen, _ = search_maxlen_adaptive(args, device, cache, cmd_args.runs,
                                                 values=get_search_space('maxlen', cmd_args.model, ext))

    save_cache(cmd_args.model, cmd_args.dataset, cache)

    # Stage 2: lr (optimization with refinement)
    print(f"\n{'=' * 60}")
    print("STAGE 2: lr (with refinement)")
    print('=' * 60)

    if cmd_args.lr is not None:
        print(f"  SKIPPED - using lr={cmd_args.lr}")
        args.lr = cmd_args.lr
    else:
        lr_values = get_search_space('lr', cmd_args.model, ext)
        args.lr, best_lr_result = search_param(args, device, cache, 'lr', lr_values, cmd_args.runs)

        # Refinement: try midpoint with neighbors
        print(f"\n  {BLUE}Refinement:{RESET}")
        best_idx = lr_values.index(args.lr)
        refined = False

        # Try midpoint with left neighbor (larger lr)
        if best_idx > 0:
            left_val = lr_values[best_idx - 1]
            mid_val = (args.lr + left_val) / 2
            setattr(args, 'lr', mid_val)
            print(f"    lr={mid_val:.6f} (midpoint with {left_val}):", end="")
            res = train_and_eval(args, device, cache, cmd_args.runs)
            if res['ndcg'] > best_lr_result['ndcg']:
                best_lr_result = res
                args.lr = mid_val
                refined = True
                print(f" {GREEN}★ NEW BEST{RESET}")
            else:
                args.lr = lr_values[best_idx]  # restore
                print(f" (no improvement)")

        # Try midpoint with right neighbor (smaller lr)
        if best_idx < len(lr_values) - 1:
            right_val = lr_values[best_idx + 1]
            mid_val = (lr_values[best_idx] + right_val) / 2
            setattr(args, 'lr', mid_val)
            print(f"    lr={mid_val:.6f} (midpoint with {right_val}):", end="")
            res = train_and_eval(args, device, cache, cmd_args.runs)
            if res['ndcg'] > best_lr_result['ndcg']:
                best_lr_result = res
                args.lr = mid_val
                refined = True
                print(f" {GREEN}★ NEW BEST{RESET}")
            else:
                if not refined:
                    args.lr = lr_values[best_idx]  # restore only if no refinement worked
                print(f" (no improvement)")

        if not refined:
            print(f"  No improvement from refinement")
        print(f"\n  {GREEN}{BOLD}▶ Best lr: {args.lr}{RESET}")

    save_cache(cmd_args.model, cmd_args.dataset, cache)

    # Stage 3: l2_decay (regularization)
    print(f"\n{'=' * 60}")
    print("STAGE 3: l2_decay")
    print('=' * 60)

    if cmd_args.l2_decay is not None:
        print(f"  SKIPPED - using l2_decay={cmd_args.l2_decay}")
        args.l2_decay = cmd_args.l2_decay
    else:
        args.l2_decay, _ = search_param(args, device, cache, 'l2_decay', get_search_space('l2_decay', cmd_args.model, ext), cmd_args.runs)

    save_cache(cmd_args.model, cmd_args.dataset, cache)

    # Stage 4: dropout (regularization, with extension and refinement)
    print(f"\n{'=' * 60}")
    print("STAGE 4: dropout (with refinement)")
    print('=' * 60)

    if cmd_args.dropout is not None:
        print(f"  SKIPPED - using dropout={cmd_args.dropout}")
        args.dropout = cmd_args.dropout
        best_result = train_and_eval(args, device, cache, cmd_args.runs)
    else:
        dropout_values = get_search_space('dropout', cmd_args.model, ext)
        args.dropout, best_result = search_param(args, device, cache, 'dropout', dropout_values, cmd_args.runs, extend=True)

        # Refinement: try midpoint with neighbors
        print(f"\n  {BLUE}Refinement:{RESET}")
        # Find where best dropout is (could be extended value)
        best_dropout = args.dropout
        refined = False

        # Determine neighbors based on whether it's in original list or extended
        if best_dropout in dropout_values:
            best_idx = dropout_values.index(best_dropout)
            left_neighbor = dropout_values[best_idx - 1] if best_idx > 0 else None
            right_neighbor = dropout_values[best_idx + 1] if best_idx < len(dropout_values) - 1 else None
        else:
            # Extended value - left neighbor is 0.5 or the previous extended value
            left_neighbor = round(best_dropout - 0.1, 1)
            right_neighbor = None  # Don't go higher than extended best

        # Try midpoint with left neighbor
        if left_neighbor is not None:
            mid_val = round((best_dropout + left_neighbor) / 2, 2)
            args.dropout = mid_val
            print(f"    dropout={mid_val} (midpoint with {left_neighbor}):", end="")
            res = train_and_eval(args, device, cache, cmd_args.runs)
            if res['ndcg'] > best_result['ndcg']:
                best_result = res
                refined = True
                print(f" {GREEN}★ NEW BEST{RESET}")
            else:
                args.dropout = best_dropout  # restore
                print(f" (no improvement)")

        # Try midpoint with right neighbor
        if right_neighbor is not None:
            mid_val = round((best_dropout + right_neighbor) / 2, 2)
            args.dropout = mid_val
            print(f"    dropout={mid_val} (midpoint with {right_neighbor}):", end="")
            res = train_and_eval(args, device, cache, cmd_args.runs)
            if res['ndcg'] > best_result['ndcg']:
                best_result = res
                refined = True
                print(f" {GREEN}★ NEW BEST{RESET}")
            else:
                if not refined:
                    args.dropout = best_dropout  # restore only if no refinement worked
                print(f" (no improvement)")

        if not refined:
            print(f"  No improvement from refinement")
        print(f"\n  {GREEN}{BOLD}▶ Best dropout: {args.dropout}{RESET}")

    save_cache(cmd_args.model, cmd_args.dataset, cache)

    # Final Summary
    print(f"\n{'=' * 60}")
    print(f"{GREEN}{BOLD}BEST CONFIGURATION{RESET}")
    print('=' * 60)
    print(f"  model: {cmd_args.model}")
    print(f"  dataset: {cmd_args.dataset}")
    print(f"  {DIM}dim: 64 (fixed){RESET}")
    print(f"  {DIM}batch_size: 512 (fixed){RESET}")
    print(f"  t_mode: {MAGENTA}{args.t_mode}{RESET}")
    print(f"  lr: {MAGENTA}{args.lr}{RESET}")
    print(f"  l2_decay: {MAGENTA}{args.l2_decay}{RESET}")
    print(f"  dropout: {MAGENTA}{args.dropout}{RESET}")
    print(f"  maxlen: {MAGENTA}{args.maxlen}{RESET}")
    print(f"  blocks: {MAGENTA}{args.blocks}{RESET}")
    print(f"  heads: {MAGENTA}{args.heads}{RESET}")
    print()
    print(f"  {GREEN}{BOLD}NDCG@10: {best_result['ndcg']:.4f}{RESET}")
    print(f"  {GREEN}{BOLD}HR@10:   {best_result['hr']:.4f}{RESET}")
    if 'best_epoch' in best_result:
        print(f"  {DIM}Best epoch (avg): {best_result['best_epoch']:.0f}{RESET}")

    # Print search space again at end
    print(f"\n{'=' * 60}")
    print(f"{MAGENTA}Search Space {'(EXTENDED)' if ext else '(STANDARD)'} (for reference):{RESET}")
    print('=' * 60)
    print(f"  Stage 1 - maxlen:   {get_search_space('maxlen', cmd_args.model, ext)} (adaptive)")
    print(f"  Stage 2 - lr:       {get_search_space('lr', cmd_args.model, ext)} + refinement (default: {DEFAULTS['lr']})")
    print(f"  Stage 3 - l2_decay: {get_search_space('l2_decay', cmd_args.model, ext)} (default: {DEFAULTS['l2_decay']})")
    print(f"  Stage 4 - dropout:  {get_search_space('dropout', cmd_args.model, ext)} + extension + refinement (default: {DEFAULTS['dropout']})")
    print(f"  {DIM}Fixed: blocks={DEFAULTS['blocks']}, heads={DEFAULTS['heads']}{RESET}")
    print(f"  {DIM}Fixed: dim=64, batch_size=512{RESET}")
    print(f"  t_mode: {args.t_mode}")
    print(f"\n{MAGENTA}Early Stopping:{RESET}")
    print(f"  Warmup: {EARLY_STOPPING['warmup']}, Min Delta: {min_delta}, Patience: {patience}, Max: {EARLY_STOPPING['max_epochs']}")

    print(f"\n{'=' * 60}")
    print("COMMAND")
    print('=' * 60)
    print(f"python src/benchmark.py --model {cmd_args.model} --dataset {cmd_args.dataset} "
          f"--t_mode {args.t_mode} --lr {args.lr} --l2_decay {args.l2_decay} --dropout {args.dropout} "
          f"--maxlen {args.maxlen} --blocks {args.blocks} --heads {args.heads} "
          f"--epochs {EARLY_STOPPING['max_epochs']} --patience {patience} --runs 5")


if __name__ == '__main__':
    main()
