import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, args, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_decay)
        self.epoch = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}', ncols=80, leave=False)

        for batch in pbar:
            tensors = [t.to(self.device) for t in batch]
            user_ids, seqs, pos, neg = tensors[:4]
            # Position 4: timestamps (unused, kept for data compatibility)
            # Position 5: cumulative count (Dynamic C)
            seq_count = tensors[5] if len(tensors) > 5 else None
            # Position 6: pre-computed age (Dynamic A)
            seq_age = tensors[6] if len(tensors) > 6 else None

            loss = self.model.compute_loss(
                user_ids, seqs, pos, neg,
                seq_count=seq_count, seq_age=seq_age
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        self.epoch += 1
        return total_loss / n_batches if n_batches > 0 else 0.0
