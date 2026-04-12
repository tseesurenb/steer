import argparse

# Map short dataset names to folder names
DATASET_MAP = {
    'ml-1m': 'ml-1m',
    'beauty': 'aws-beauty',
    'health': 'aws-health',
    '30m': '30m',
}

def get_config():
    parser = argparse.ArgumentParser(prog="Sequential Recommendation Framework")

    # Data
    parser.add_argument('--dataset', type=str, default='ml-1m', choices=['ml-1m', 'beauty', 'health', '30m'])
    parser.add_argument('--split', type=int, nargs=3, default=[90, 5, 5], metavar=('TRAIN', 'VAL', 'TEST'), help='Train/val/test split percentages (must sum to 100)')

    # Model (standardized for fair comparison)
    parser.add_argument('--model', type=str, default='steer', choices=['steer', 'sasrec'])
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--maxlen', type=int, default=35)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--blocks', type=int, default=2, help='Transformer blocks')
    parser.add_argument('--heads', type=int, default=1, help='Attention heads')
    # Training (standardized defaults)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2_decay', type=float, default=0.001)
    parser.add_argument('--eval', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--warmup', type=int, default=20, help='Min epochs before checkpoint selection (prevents early false peaks)')
    parser.add_argument('--early', type=int, default=None, help='Early stopping patience (stop after N epochs without improvement). If not set, run all epochs.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=3, help='Number of runs with different seeds (for mean±std)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    # Temporal encoding
    parser.add_argument('--no_pos', action='store_true', help='Disable learnable positional encoding')
    parser.add_argument('--add_pos_all', action='store_true', help='STEER: Add position embedding to Q, K, V (default: Q only)')
    parser.add_argument('--add_pos_q', action='store_true', help='STEER: Add position embedding to Q (same as default, for explicit control)')
    parser.add_argument('--add_pos_k', action='store_true', help='STEER: Add position embedding to K')
    parser.add_argument('--add_pos_v', action='store_true', help='STEER: Add position embedding to V')
    parser.add_argument('--uniform_routing', action='store_true', help='STEER: Use uniform routing weights (1/3 each) instead of learned')
    parser.add_argument('--t_mode', type=str, default='lnAC_WHYM',
                        help='Features: flnAC_HWYM. Static: f(first),l(last),n(users) | '
                             'Dynamic: A(age),C(count) | Cyclic: H(hour),W(week),M(month),Y(year)')
    # Logging

    args = parser.parse_args()

    # Resolve split percentages to ratios
    if sum(args.split) != 100:
        parser.error(f'--split must sum to 100, got {sum(args.split)}')
    args.train_ratio = args.split[0] / 100.0
    args.val_ratio = args.split[1] / 100.0
    args.test_ratio = args.split[2] / 100.0

    # Temporal base period (1 day in seconds)
    args.t_tau = 86400.0

    # Resolve dataset name to folder name
    args.dataset_folder = DATASET_MAP[args.dataset]

    return args
