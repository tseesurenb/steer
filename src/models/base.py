import torch
from torch import nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, num_items, args):
        super().__init__()
        self.num_items = num_items
        self.args = args

    def forward(self, seq, seq_count=None, seq_age=None):
        raise NotImplementedError

    def predict(self, user_ids, seq, item_indices, seq_count=None, seq_age=None):
        raise NotImplementedError

    def compute_loss(self, user_ids, seq, pos, neg, seq_count=None, seq_age=None):
        raise NotImplementedError

    def get_item_embeddings(self):
        """Return item embeddings. Override in subclass if needed."""
        return self.item_emb.weight


class BPRLossMixin:
    def compute_bpr_loss(self, seq, pos, neg, seq_count=None, seq_age=None):
        seq_emb = self.forward(seq, seq_count=seq_count, seq_age=seq_age)

        item_embeddings = self.get_item_embeddings()
        pos_emb = F.embedding(pos, item_embeddings)
        neg_emb = F.embedding(neg, item_embeddings)

        pos_logits = torch.sum(seq_emb * pos_emb, dim=-1)
        neg_logits = torch.sum(seq_emb * neg_emb, dim=-1)

        mask = (pos != 0).float()
        loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-24) * mask
        return loss.sum() / mask.sum()
