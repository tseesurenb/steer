import torch
from torch import nn
import torch.nn.functional as F
import math
from .base import BaseModel, BPRLossMixin


class ItemTemporalEncoder(nn.Module):
    """Temporal encoder for item features.

    Encoding methods:
    - Sinusoidal: Multi-scale representation via sin/cos at different frequencies
    - Log-linear: For counts/durations (magnitude signals)
    - Cyclic calendar: For periodic calendar patterns (sin/cos)

    Static features (lowercase, computed once per item):
    - 'f': t_first - first timestamp (sinusoidal)
    - 'l': t_last - last timestamp (sinusoidal)
    - 'n': n_users - total user count (log-linear)

    Dynamic features (UPPERCASE, computed per interaction):
    - 'A': age - t - t_first, item age at interaction (log-linear)
    - 'C': count - n_users(t), cumulative user count (log-linear)

    Cyclic calendar features (after underscore in t_mode, e.g. flnAC_HWYM):
    - 'H': hour of day (0-23, period=24)
    - 'W': day of week (0-6, period=7)
    - 'M': month of year (0-11, period=12)
    - 'Y': day of year (0-364, period=365)
    """

    def __init__(self, d_model, tau=86400.0, t_mode='lnAC', base=10000, cyclic='WHYM'):
        super().__init__()
        self.d_model = d_model
        self.t_mode = t_mode
        self.base = base
        self.cyclic = cyclic

        # Static feature flags (lowercase)
        self.use_t_first = 'f' in t_mode
        self.use_t_last = 'l' in t_mode
        self.use_n_users = 'n' in t_mode

        # Dynamic feature flags (UPPERCASE)
        self.use_age = 'A' in t_mode
        self.use_count = 'C' in t_mode

        # Cyclic calendar feature flags
        self.use_hour = 'H' in cyclic
        self.use_week = 'W' in cyclic
        self.use_month = 'M' in cyclic
        self.use_year = 'Y' in cyclic
        self.has_cyclic = self.use_hour or self.use_week or self.use_month or self.use_year

        # n_users projection (log-scaled -> d_model)
        if self.use_n_users:
            self.n_users_proj = nn.Linear(1, d_model)

        # count projection (log-scaled -> d_model)
        if self.use_count:
            self.count_proj = nn.Linear(1, d_model)

        # age projection (log-scaled -> d_model)
        if self.use_age:
            self.age_proj = nn.Linear(1, d_model)

        # Fixed sinusoidal frequency bands (for absolute timestamps f, l)
        d_half = d_model // 2
        i = torch.arange(d_half, dtype=torch.float32)
        freqs = tau * (float(base) ** (2 * i / d_model))
        self.register_buffer('freqs', freqs)

        # Cyclic frequency bands (for calendar features H, W, M, Y)
        cyclic_base = 100.0
        cyclic_freqs = cyclic_base ** (2 * i / d_model)
        self.register_buffer('cyclic_freqs', cyclic_freqs)

    def _log_encode(self, x):
        """Log-linear encoding."""
        return torch.log1p(x.float()).unsqueeze(-1)

    def _sinusoidal(self, t):
        angles = t.unsqueeze(-1) / self.freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def _cyclic_encode(self, value, period):
        """Multi-frequency sinusoidal encoding for cyclic features."""
        angle = 2 * math.pi * value.float() / period
        angles = angle.unsqueeze(-1) / self.cyclic_freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, item_emb, t_first, t_last, n_users, seq_age=None, seq_count=None):
        """Apply temporal encoding to item embeddings.

        Args:
            item_emb: (..., d_model) item embeddings
            t_first: (...) first timestamp per item
            t_last: (...) last timestamp per item
            n_users: (...) total user count per item
            seq_age: (...) pre-computed item age at interaction (Dynamic A)
            seq_count: (...) cumulative count at interaction (Dynamic C)

        Returns:
            (..., d_model) temporally-encoded embeddings
        """
        result = item_emb

        # Static features
        if self.use_t_first:
            result = result + self._sinusoidal(t_first.float())
        if self.use_t_last:
            result = result + self._sinusoidal(t_last.float())
        if self.use_n_users:
            result = result + self.n_users_proj(self._log_encode(n_users))

        # Dynamic features
        if self.use_age and seq_age is not None:
            result = result + self.age_proj(self._log_encode(seq_age))
        if self.use_count and seq_count is not None:
            result = result + self.count_proj(self._log_encode(seq_count))

        # Cyclic calendar features
        if self.has_cyclic and seq_age is not None:
            # Reconstruct interaction time: t = t_first + seq_age
            t = t_first.float() + seq_age.float()
            t_seconds = t.long()

            if self.use_hour:
                hour = (t_seconds // 3600) % 24
                result = result + self._cyclic_encode(hour, 24)
            if self.use_week:
                week = (t_seconds // 86400 + 4) % 7
                result = result + self._cyclic_encode(week, 7)
            if self.use_month:
                day_of_year = (t_seconds // 86400) % 365
                month = (day_of_year * 12) // 365
                result = result + self._cyclic_encode(month, 12)
            if self.use_year:
                year = (t_seconds // 86400) % 365
                result = result + self._cyclic_encode(year, 365)

        return result


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking (original SASRec architecture)."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._causal_mask_cache = {}

    def _get_causal_mask(self, seq_len, device):
        key = (seq_len, device)
        if key not in self._causal_mask_cache:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self._causal_mask_cache[key] = mask
        return self._causal_mask_cache[key]

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask
        causal_mask = self._get_causal_mask(seq_len, x.device)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # Padding mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return output


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward (original SASRec architecture)."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        # Original SASRec: 1x expansion with ReLU (not 4x with GELU)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with pre-norm
        attn_out = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        # Feed-forward with pre-norm
        ff_out = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_out)
        return x


class SASRec(BaseModel, BPRLossMixin):
    """SASRec with temporal encoding.

    Temporal features are added to the embeddings before passing through
    the transformer blocks.
    """

    def __init__(self, num_items, args):
        super().__init__(num_items, args)
        self.maxlen = args.maxlen
        self.d_model = args.dim

        # Item embeddings
        self.item_emb = nn.Embedding(num_items + 1, self.d_model, padding_idx=0)

        # Positional encoding
        self.pos_emb = nn.Embedding(args.maxlen + 1, self.d_model, padding_idx=0)

        # Temporal settings (always enabled)
        t_mode_full = getattr(args, 't_mode', 'lnAC_WHYM')
        t_tau = getattr(args, 't_tau', 86400.0)

        # Parse t_mode: "flnsAC_HWYM" -> t_mode="flnsAC", t_cyclic="HWYM"
        if '_' in t_mode_full:
            self.t_mode, self.t_cyclic = t_mode_full.split('_', 1)
        else:
            self.t_mode = t_mode_full
            self.t_cyclic = ''

        self.temporal_encoder = ItemTemporalEncoder(
            self.d_model,
            tau=t_tau,
            t_mode=self.t_mode,
            cyclic=self.t_cyclic
        )

        # Item feature buffers
        self.register_buffer('item_t_first', torch.zeros(num_items + 1))
        self.register_buffer('item_t_last', torch.zeros(num_items + 1))
        self.register_buffer('item_n_users', torch.zeros(num_items + 1))

        self.emb_scale = math.sqrt(self.d_model)
        self.emb_dropout = nn.Dropout(args.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, args.heads, args.dropout)
            for _ in range(args.blocks)
        ])

        self.ln_final = nn.LayerNorm(self.d_model)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, seq, seq_count=None, seq_age=None):
        """Forward pass through SASRec.

        Args:
            seq: [batch, seq_len] item sequence
            seq_count: [batch, seq_len] cumulative count (Dynamic C)
            seq_age: [batch, seq_len] item age at interaction (Dynamic A)
        """
        batch_size, seq_len = seq.shape
        mask = (seq != 0).float()

        # Item embeddings
        seq_emb = self.item_emb(seq) * self.emb_scale

        # Position embeddings
        positions = torch.arange(1, seq_len + 1, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        positions = positions * mask.long()
        seq_emb = seq_emb + self.pos_emb(positions)

        # Apply temporal encoding
        t_first = self.item_t_first[seq]
        t_last = self.item_t_last[seq]
        n_users = self.item_n_users[seq]

        seq_emb = self.temporal_encoder(
            seq_emb, t_first, t_last, n_users,
            seq_age=seq_age, seq_count=seq_count
        )

        seq_emb = self.emb_dropout(seq_emb)
        seq_emb = seq_emb * mask.unsqueeze(-1)

        # Transformer blocks
        for block in self.blocks:
            seq_emb = block(seq_emb, mask=mask)

        seq_emb = self.ln_final(seq_emb)
        return seq_emb

    def predict(self, user_ids, seq, item_indices, seq_count=None, seq_age=None):
        """Predict scores for candidate items."""
        seq_emb = self.forward(seq, seq_count=seq_count, seq_age=seq_age)
        seq_emb = seq_emb[:, -1, :]  # Last position

        if item_indices.dim() == 1:
            item_emb = self.item_emb.weight[item_indices]
            scores = torch.matmul(seq_emb, item_emb.transpose(0, 1))
        else:
            item_emb = self.item_emb(item_indices)
            scores = torch.sum(seq_emb.unsqueeze(1) * item_emb, dim=-1)

        return scores

    def compute_loss(self, user_ids, seq, pos, neg, seq_count=None, seq_age=None):
        """Compute BPR training loss."""
        seq_emb = self.forward(seq, seq_count=seq_count, seq_age=seq_age)

        pos_emb = self.item_emb(pos)
        neg_emb = self.item_emb(neg)

        pos_logits = torch.sum(seq_emb * pos_emb, dim=-1)
        neg_logits = torch.sum(seq_emb * neg_emb, dim=-1)

        mask = (pos != 0).float()
        loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-24) * mask
        return loss.sum() / mask.sum()

    def set_item_features(self, item_features, device):
        """Set item temporal features."""
        if item_features is not None:
            self.item_t_first = item_features['t_first'].to(device)
            self.item_t_last = item_features['t_last'].to(device)
            self.item_n_users = item_features['n_users'].to(device)

    def get_item_embeddings(self):
        """Get item embeddings for candidate scoring."""
        return self.item_emb.weight
