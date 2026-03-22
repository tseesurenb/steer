"""
STEER: Selective Temporal Expert Routing for Sequential Recommendation.

Usage:
    python src/main.py --dataset ml-1m --epochs 200
    python src/main.py --dataset ml-1m --t_mode fnAC_HWYM
    python src/main.py --model sasrec --dataset beauty --t_mode lnAC_WHYM
"""
import os
import sys

# Ensure src/ is in path for imports (works from any directory)
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import copy
import torch
from config import get_config
from data import get_data_loaders
from evaluator import evaluate
from trainer import Trainer
from utils import set_seed, get_device, get_model, init_wandb, setup_model, get_experiment_summary, print_results, setup_logging


def main():
    args = get_config()

    # Setup logging
    log_path, logger = setup_logging(args)

    try:
        set_seed(args.seed)

        device = get_device(args.device)
        print(f"Using device: {device}")

        if args.wandb:
            os.environ['WANDB_SILENT'] = 'true'
            import wandb
            init_wandb(args)

        data_path = os.path.join('data', 'gold', args.dataset_folder, f'{args.dataset_folder}-time.txt')
        train_loader, dataset_info = get_data_loaders(data_path, args)
        args.num_users = dataset_info['num_users']

        model = get_model(args.model, dataset_info['num_items'], args).to(device)
        setup_model(model, args, dataset_info, device)

        trainer = Trainer(model, train_loader, args, device)
        user_val = dataset_info['user_val']

        # Track best model by NDCG@10 (standard practice)
        best_val_ndcg = 0.0
        best_state = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            loss = trainer.train_epoch()

            if epoch % args.eval == 0:
                ndcg, hr, _ = evaluate(model, dataset_info, args.maxlen, device, user_eval=user_val)
                print(f"Epoch {epoch}: Loss={loss:.4f}, Val NDCG@10={ndcg[10]:.4f}, Val HR@10={hr[10]:.4f}")

                if args.wandb:
                    wandb.log({"epoch": epoch, "loss": loss, "val/ndcg10": ndcg[10], "val/hr10": hr[10]})

                # Only consider checkpoints after warmup period (prevents early false peaks)
                if epoch >= args.warmup and ndcg[10] > best_val_ndcg:
                    best_val_ndcg = ndcg[10]
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    patience_counter = 0
                elif epoch >= args.warmup:
                    patience_counter += 1

                # Early stopping
                if args.early is not None and patience_counter >= args.early:
                    print(f"Early stopping at epoch {epoch} (no improvement for {args.early} eval cycles)")
                    break

        # Test evaluation
        exp_summary = get_experiment_summary(args)
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {exp_summary}")
        print('='*60)

        if best_state is not None:
            model.load_state_dict(best_state)
            ndcg, hr, recall = evaluate(model, dataset_info, args.maxlen, device)
            print(f"Test (best epoch {best_epoch}):")
            print_results(ndcg, hr, recall)
            if args.wandb:
                wandb.log({"test/ndcg10": ndcg[10], "test/hr10": hr[10]})

        if args.wandb:
            run_url = wandb.run.get_url() if wandb.run else None
            wandb.finish(quiet=True)
            if run_url:
                print(f"\nwandb: {run_url}")

    finally:
        logger.close()
        torch.cuda.empty_cache()
        # Print after logger closed (goes to terminal only)
        print(f"Log saved to: {log_path}")


if __name__ == '__main__':
    main()
