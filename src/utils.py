import os
import re
import sys
from datetime import datetime
import torch
import numpy as np
import random


# Track what has been printed to avoid repeated messages during search
_printed_messages = set()

def print_once(msg, key=None):
    """Print a message only once per session. Uses msg as key if key not provided."""
    k = key if key is not None else msg
    if k not in _printed_messages:
        _printed_messages.add(k)
        print(msg)

def reset_print_once():
    """Reset printed messages (call at start of new experiment)."""
    global _printed_messages
    _printed_messages.clear()


class Logger:
    """Logger that writes to both console and file."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'w')
        self.log_path = log_path
        self._closed = False

    def write(self, message):
        self.terminal.write(message)
        if not self._closed:
            # Strip ANSI color codes for file
            clean_message = re.sub(r'\x1b\[[0-9;]*m', '', message)
            self.log_file.write(clean_message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if not self._closed:
            self.log_file.flush()

    def close(self):
        if not self._closed:
            self._closed = True
            self.log_file.close()
            sys.stdout = self.terminal


def get_log_filename(args):
    """Generate run name for logging.

    Returns: (dataset, filename) tuple where filename starts with timestamp.
    Model name is excluded since it's in the directory path.
    """
    # Timestamp first for easy sorting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [timestamp]

    # Both models use temporal
    parts.append(getattr(args, 't_mode', 'lnAC_WHYM'))
    # Learning rate (format: 1e-04 -> lr1e04)
    lr = getattr(args, 'lr', None)
    if lr is not None:
        lr_str = f"{lr:.0e}".replace('-', '')  # 1e-04 -> 1e04
        parts.append(f"lr{lr_str}")

    # Dropout (format: 0.5 -> d05)
    dropout = getattr(args, 'dropout', None)
    if dropout is not None:
        parts.append(f"d{int(dropout*10)}")

    # L2 decay (only if non-zero)
    l2 = getattr(args, 'l2_decay', 0)
    if l2 > 0:
        l2_str = f"{l2:.0e}".replace('-', '')
        parts.append(f"l2{l2_str}")

    return args.dataset, "_".join(parts)


def setup_logging(args):
    """Setup logging to file and console. Returns log path.

    Logs are saved to: runs/{model}/{dataset}/{timestamp}_....log
    """
    dataset, run_name = get_log_filename(args)
    log_path = os.path.join('runs', args.model, dataset, f'{run_name}.log')
    logger = Logger(log_path)
    sys.stdout = logger
    return log_path, logger


MODEL_IMPORTS = {
    'steer': ('models.steer', 'STEER'),
    'sasrec': ('models.sasrec', 'SASRec'),
}
MODEL_REGISTRY = {}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(device_arg):
    """Get torch device based on argument and availability."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print("Warning: CUDA not available, falling back to CPU")
        return torch.device('cpu')
    elif device_arg == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        print("Warning: MPS not available, falling back to CPU")
        return torch.device('cpu')
    return torch.device('cpu')

def print_results(ndcg, hr, recall):
    """Print evaluation results with highlighting."""
    R = "\033[1;91m"  # Bold red
    E = "\033[0m"     # Reset
    print(f"  NDCG@5={ndcg[5]:.4f}, {R}NDCG@10={ndcg[10]:.4f}{E}, NDCG@20={ndcg[20]:.4f}")
    print(f"  HR@5={hr[5]:.4f}, {R}HR@10={hr[10]:.4f}{E}, HR@20={hr[20]:.4f}")
    print(f"  Recall@5={recall[5]:.4f}, Recall@10={recall[10]:.4f}, Recall@20={recall[20]:.4f}")


def get_experiment_summary(args):
    """Generate a concise summary of experiment configuration."""
    parts = [f"{args.model.upper()}", f"{args.dataset}"]

    # Both models use temporal
    t_mode = getattr(args, 't_mode', 'lnAC_WHYM')
    parts.append(f"Temp({t_mode})")

    parts.append(f"lr={args.lr}")
    parts.append(f"drop={args.dropout}")
    if args.l2_decay > 0:
        parts.append(f"l2={args.l2_decay}")

    return " | ".join(parts)


def get_model(model_name, num_items, args):
    if model_name not in MODEL_REGISTRY:
        if model_name not in MODEL_IMPORTS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_IMPORTS.keys())}")
        module_path, class_name = MODEL_IMPORTS[model_name]
        import importlib
        module = importlib.import_module(module_path)
        MODEL_REGISTRY[model_name] = getattr(module, class_name)
    return MODEL_REGISTRY[model_name](num_items, args)

def setup_model(model, args, dataset_info, device):
    """Set up temporal features for models."""
    # Set up temporal encoding
    item_features = dataset_info.get('item_features')
    if item_features is not None and hasattr(model, 'set_item_features'):
        model.set_item_features(item_features, device)
        t_mode = getattr(args, 't_mode', 'lnAC_WHYM')
        print_once(f"Temporal: mode={t_mode}")
