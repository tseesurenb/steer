"""
Benchmark script for running experiments multiple times with different seeds.
Reports mean ± std for publishing.

Usage:
    python src/benchmark.py --dataset ml-1m --runs 3
    python src/benchmark.py --dataset ml-1m --runs 5 --t_mode fnAC_HWYM
    python src/benchmark.py --model sasrec --dataset beauty --runs 5 --t_mode lnAC_WHYM
"""
import os
import sys

# Ensure src/ is in path for imports (works from any directory)
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import copy
import torch
import numpy as np
from config import get_config
from data import get_data_loaders
from evaluator import evaluate
from trainer import Trainer
from utils import set_seed, get_device, get_model, setup_model, get_experiment_summary, setup_logging


def run_single(args, run_idx, device):
    """Run a single experiment and return test metrics."""
    seed = args.seed + run_idx
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Run {run_idx + 1}/{args.runs} (seed={seed})")
    print('='*60)

    data_path = os.path.join('data', 'gold', args.dataset_folder, f'{args.dataset_folder}-time.txt')
    train_loader, dataset_info = get_data_loaders(data_path, args)

    args.num_users = dataset_info['num_users']

    model = get_model(args.model, dataset_info['num_items'], args).to(device)
    setup_model(model, args, dataset_info, device)

    trainer = Trainer(model, train_loader, args, device)
    user_val = dataset_info['user_val']

    best_val_score = (0.0, 0.0)  # (ndcg, hr) for tiebreaking
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        loss = trainer.train_epoch()

        if epoch % args.eval == 0:
            ndcg, hr, _ = evaluate(model, dataset_info, args.maxlen, device, user_eval=user_val)

            # Only consider checkpoints after warmup period (prevents early false peaks)
            # Compare (ndcg rounded to 4 decimals, hr) - HR breaks ties when NDCG ~equal
            current_score = (round(ndcg[10], 4), round(hr[10], 5))
            is_best = epoch >= args.warmup and current_score > best_val_score
            marker = " *" if is_best else ""
            print(f"Epoch {epoch}: Loss={loss:.4f}, Val NDCG@10={ndcg[10]:.4f}, Val HR@10={hr[10]:.4f}{marker}")

            if is_best:
                best_val_score = current_score
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            elif epoch >= args.warmup:
                patience_counter += 1

            # Early stopping
            if args.early is not None and patience_counter >= args.early:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.early} eval cycles)")
                break

    # Test with best model
    if best_state is not None:
        model.load_state_dict(best_state)
    ndcg, hr, recall = evaluate(model, dataset_info, args.maxlen, device)
    B, E = "\033[1m", "\033[0m"  # Bold
    print(f"{B}Test (best epoch {best_epoch}): NDCG@10={ndcg[10]:.4f}, HR@10={hr[10]:.4f}{E}")

    # Cleanup
    del model, trainer, best_state
    torch.cuda.empty_cache()

    return {'ndcg': ndcg, 'hr': hr, 'recall': recall, 'best_epoch': best_epoch}


def aggregate_results(all_results):
    """Compute mean and std from multiple runs."""
    metrics = {}
    ks = [5, 10, 20]

    for metric_name in ['ndcg', 'hr', 'recall']:
        for k in ks:
            values = [r[metric_name][k] for r in all_results]
            metrics[f'{metric_name}@{k}'] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }

    # Collect best epochs
    epochs = [r['best_epoch'] for r in all_results]
    metrics['epochs'] = {
        'mean': np.mean(epochs),
        'values': epochs
    }

    return metrics


def print_benchmark_results(metrics, args):
    """Print formatted results table."""
    exp_summary = get_experiment_summary(args)
    epochs = metrics['epochs']
    epochs_str = ', '.join([str(int(e)) for e in epochs['values']])

    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS ({args.runs} runs): {exp_summary}")
    print(f"Best epochs: [{epochs_str}] (avg: {epochs['mean']:.1f})")
    print('='*70)

    print(f"\n{'Metric':<12} {'Mean':>10} {'± Std':>10} {'Values'}")
    print('-'*60)

    for k in [5, 10, 20]:
        for metric in ['hr', 'ndcg', 'recall']:
            name = f'{metric}@{k}'
            m = metrics[name]
            values_str = ', '.join([f'{v:.4f}' for v in m['values']])
            print(f"{name:<12} {m['mean']:>10.4f} {m['std']:>10.4f}   [{values_str}]")
        if k != 20:
            print('-'*60)

    # LaTeX format for paper
    print(f"\n--- LaTeX format (HR@10 & NDCG@10) ---")
    hr10 = metrics['hr@10']
    ndcg10 = metrics['ndcg@10']
    print(f"{hr10['mean']:.4f}$\\pm${hr10['std']:.4f} & {ndcg10['mean']:.4f}$\\pm${ndcg10['std']:.4f}")

    # Compact format
    print(f"\n--- Compact format ---")
    print(f"HR@10:   {hr10['mean']:.4f} ± {hr10['std']:.4f}")
    print(f"NDCG@10: {ndcg10['mean']:.4f} ± {ndcg10['std']:.4f}")
    print(f"Epochs:  {epochs['mean']:.1f} avg [{epochs_str}]")

    # Print running command
    print(f"\n--- Command ---")
    print(f"python {' '.join(sys.argv)}")


def main():
    args = get_config()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    # Setup logging
    log_path, logger = setup_logging(args)

    try:
        device = get_device(args.device)
        print(f"Using device: {device}")
        print(f"Running {args.runs} experiments with seeds {args.seed} to {args.seed + args.runs - 1}")

        all_results = []
        for run_idx in range(args.runs):
            result = run_single(args, run_idx, device)
            all_results.append(result)


        metrics = aggregate_results(all_results)
        print_benchmark_results(metrics, args)

    finally:
        logger.close()
        torch.cuda.empty_cache()
        # Print after logger closed (goes to terminal only)
        print(f"Log saved to: {log_path}")


if __name__ == '__main__':
    main()
