"""STEER: Selective Temporal Expert Routing for Sequential Recommendation.

Key Innovation: Top-1 routing of temporal features to Q, K, V paths.
Each feature learns to route to exactly one of Q, K, or V using hard routing
with straight-through estimator for gradient flow.

Features (9 total):
- Static: f (t_first), l (t_last), n (n_users)
- Dynamic: A (age), C (count)
- Cyclic: H (hour), W (week), M (month), Y (year)
"""
import torch
from torch import nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import print_once
from .base import BaseModel, BPRLossMixin


class TemporalFeatureEncoder(nn.Module):
    """Unified encoder for all temporal features.

    Encodes each feature independently, then router decides Q/K/V distribution.
    """

    def __init__(self, d_model, t_mode='lnAC', tau=86400.0, base=10000):
        super().__init__()
        self.d_model = d_model
        self.t_mode = t_mode

        # Feature flags
        self.use_n_users = 'n' in t_mode
        self.use_age = 'A' in t_mode
        self.use_count = 'C' in t_mode

        # Projections for log-linear features
        if self.use_n_users:
            self.n_users_proj = nn.Linear(1, d_model)
        if self.use_age:
            self.age_proj = nn.Linear(1, d_model)
        if self.use_count:
            self.count_proj = nn.Linear(1, d_model)

        # Sinusoidal frequency bands for timestamps
        d_half = d_model // 2
        i = torch.arange(d_half, dtype=torch.float32)
        freqs = tau * (float(base) ** (2 * i / d_model))
        self.register_buffer('freqs', freqs)

        # Cyclic frequency bands
        cyclic_base = 100.0
        cyclic_freqs = cyclic_base ** (2 * i / d_model)
        self.register_buffer('cyclic_freqs', cyclic_freqs)

    def _log_encode(self, x):
        """Log-linear encoding."""
        return torch.log1p(x.float()).unsqueeze(-1)

    def _sinusoidal(self, t):
        """Sinusoidal encoding for timestamps."""
        angles = t.unsqueeze(-1) / self.freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def _cyclic_encode(self, value, period):
        """Cyclic encoding for calendar features."""
        angle = 2 * math.pi * value.float() / period
        angles = angle.unsqueeze(-1) / self.cyclic_freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def encode_static(self, t_first, t_last, n_users, features='fln'):
        """Encode static features.

        Args:
            t_first, t_last, n_users: [...] tensors
            features: which features to encode (string of 'fln')

        Returns:
            dict of feature_name -> [..., d_model] encoded tensors
        """
        encodings = {}
        if 'f' in features:
            encodings['f'] = self._sinusoidal(t_first.float())
        if 'l' in features:
            encodings['l'] = self._sinusoidal(t_last.float())
        if 'n' in features:
            encodings['n'] = self.n_users_proj(self._log_encode(n_users))
        return encodings

    def encode_dynamic(self, seq_age, seq_count, features='AC'):
        """Encode dynamic features.

        Args:
            seq_age, seq_count: [batch, seq_len] tensors
            features: which features to encode (string of 'AC')

        Returns:
            dict of feature_name -> [batch, seq_len, d_model] encoded tensors
        """
        encodings = {}
        if 'A' in features and seq_age is not None:
            encodings['A'] = self.age_proj(self._log_encode(seq_age))
        if 'C' in features and seq_count is not None:
            encodings['C'] = self.count_proj(self._log_encode(seq_count))

        return encodings

    def encode_cyclic(self, t, features=''):
        """Encode cyclic calendar features.

        Args:
            t: [...] timestamps in seconds
            features: which features to encode (string of 'HWMY')

        Returns:
            dict of feature_name -> [..., d_model] encoded tensors
        """
        encodings = {}
        if not features or t is None:
            return encodings

        t_seconds = t.long()

        if 'H' in features:
            hour = (t_seconds // 3600) % 24
            encodings['H'] = self._cyclic_encode(hour, 24)
        if 'W' in features:
            week = (t_seconds // 86400 + 4) % 7
            encodings['W'] = self._cyclic_encode(week, 7)
        if 'M' in features:
            day_of_year = (t_seconds // 86400) % 365
            month = (day_of_year * 12) // 365
            encodings['M'] = self._cyclic_encode(month, 12)
        if 'Y' in features:
            year = (t_seconds // 86400) % 365
            encodings['Y'] = self._cyclic_encode(year, 365)
        return encodings


class TemporalRouter(nn.Module):
    """Top-1 Router for temporal features to Q, K, V paths.

    Each feature learns to route to exactly one of Q, K, or V using hard routing
    with straight-through estimator for gradient flow.
    """

    FEATURE_NAMES = ['f', 'l', 'n', 'A', 'C', 'H', 'W', 'M', 'Y']

    def __init__(self, d_model, features='flnsAC', cyclic='', uniform=False):
        super().__init__()
        self.d_model = d_model
        self.features = features + cyclic
        self.uniform = uniform

        # Active features
        self.active_features = [f for f in self.FEATURE_NAMES if f in self.features]
        self.n_features = len(self.active_features)

        # Learnable routing logits: [n_features, 3]
        self.routing_logits = nn.Parameter(torch.zeros(self.n_features, 3))

    def get_routing_weights(self):
        """Get top-1 routing weights [n_features, 3] for Q, K, V."""
        if self.uniform:
            # Uniform routing: equal weights to Q, K, V
            return torch.ones(self.n_features, 3, device=self.routing_logits.device) / 3.0
        probs = F.softmax(self.routing_logits, dim=-1)
        # Hard routing: argmax with straight-through
        indices = probs.argmax(dim=-1)
        one_hot = F.one_hot(indices, num_classes=3).float()
        weights = one_hot - probs.detach() + probs
        return weights

    def route_features(self, encodings):
        """Route encoded features to Q, K, V paths.

        Args:
            encodings: dict of feature_name -> [..., d_model] tensors

        Returns:
            q_emb, k_emb, v_emb: [..., d_model] tensors
        """
        weights = self.get_routing_weights()

        # Initialize outputs
        first_enc = next(iter(encodings.values()))
        shape = first_enc.shape
        device = first_enc.device

        q_emb = torch.zeros(shape, device=device)
        k_emb = torch.zeros(shape, device=device)
        v_emb = torch.zeros(shape, device=device)

        for i, feat_name in enumerate(self.active_features):
            if feat_name not in encodings:
                continue

            enc = encodings[feat_name]
            w_q, w_k, w_v = weights[i, 0], weights[i, 1], weights[i, 2]
            q_emb = q_emb + w_q * enc
            k_emb = k_emb + w_k * enc
            v_emb = v_emb + w_v * enc

        return q_emb, k_emb, v_emb


class CrossAttentionSeparateKV(nn.Module):
    """Cross-attention with separate K and V embeddings (original SASRec architecture)."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # Note: Original SASRec has no output projection (W_O)

        self.dropout = nn.Dropout(dropout)
        self._causal_mask_cache = {}

    def _get_causal_mask(self, seq_len, device):
        key = (seq_len, device)
        if key not in self._causal_mask_cache:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self._causal_mask_cache[key] = mask
        return self._causal_mask_cache[key]

    def forward(self, q_emb, k_emb, v_emb, mask=None, causality=False):
        """
        Args:
            q_emb: [batch, seq_len, d_model] - query embeddings
            k_emb: [batch, seq_len, d_model] - key embeddings (separate from V)
            v_emb: [batch, seq_len, d_model] - value embeddings (separate from K)
            mask: [batch, seq_len] padding mask
            causality: whether to apply causal masking
        """
        batch_size, seq_len, _ = q_emb.shape

        Q = self.W_Q(q_emb).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_K(k_emb).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_V(v_emb).view(batch_size, seq_len, self.num_heads, self.d_k)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        if causality:
            causal_mask = self._get_causal_mask(seq_len, scores.device)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return output  # No output projection, same as original SASRec


class CrossAttentionBlockSeparateKV(nn.Module):
    """Cross-attention block with separate K and V (original SASRec architecture)."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttentionSeparateKV(d_model, num_heads, dropout)
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

    def forward(self, q_emb, k_emb, v_emb, mask=None, causality=False):
        # Cross-attention with pre-norm
        normed_q = self.ln1(q_emb)
        attn_out = self.cross_attn(normed_q, k_emb, v_emb, mask, causality)
        q_emb = q_emb + self.dropout(attn_out)

        # Feed-forward with pre-norm
        q_emb = q_emb + self.dropout(self.feed_forward(self.ln2(q_emb)))

        if mask is not None:
            q_emb = q_emb * mask.unsqueeze(-1)

        return q_emb


class STEER(BaseModel, BPRLossMixin):
    """STEER: Selective Temporal Expert Routing.

    All temporal features are encoded, then routed to Q, K, V via top-1 routing.
    """

    def __init__(self, num_items, args):
        super().__init__(num_items, args)

        self.maxlen = args.maxlen
        self.d_model = args.dim

        # Item embeddings
        self.item_emb = nn.Embedding(num_items + 1, self.d_model, padding_idx=0)

        # Positional encoding (added to Q only)
        self.pos_emb = nn.Embedding(args.maxlen + 1, self.d_model, padding_idx=0)

        # Temporal settings (always enabled for STEER)
        t_tau = getattr(args, 't_tau', 86400.0)
        t_mode_full = getattr(args, 't_mode', 'lnAC_WHYM')

        # Parse t_mode: "flnsAC_HWYM" -> features="flnsAC", cyclic="HWYM"
        if '_' in t_mode_full:
            self.t_mode, self.t_cyclic = t_mode_full.split('_', 1)
        else:
            self.t_mode = t_mode_full
            self.t_cyclic = ''

        # Parse features
        self.static_features = ''.join([f for f in 'fln' if f in self.t_mode])
        self.dynamic_features = ''.join([f for f in 'AC' if f in self.t_mode])

        # Unified temporal encoder
        self.temporal_encoder = TemporalFeatureEncoder(
            self.d_model, t_mode=self.t_mode, tau=t_tau, base=10000
        )

        # Top-1 Router
        self.temporal_router = TemporalRouter(
            self.d_model,
            features=self.t_mode,
            cyclic=self.t_cyclic,
            uniform=getattr(args, 'uniform_routing', False)
        )

        # Item feature buffers
        self.register_buffer('item_t_first', torch.zeros(num_items + 1))
        self.register_buffer('item_t_last', torch.zeros(num_items + 1))
        self.register_buffer('item_n_users', torch.zeros(num_items + 1))

        self.emb_scale = math.sqrt(self.d_model)
        self.emb_dropout = nn.Dropout(args.dropout)

        # Cross-attention blocks with separate K, V
        self.blocks = nn.ModuleList([
            CrossAttentionBlockSeparateKV(self.d_model, args.heads, args.dropout)
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

    def get_base_embeddings(self):
        """Get base item embeddings."""
        return self.item_emb.weight

    def forward(self, seq, seq_count=None, seq_age=None):
        """Forward pass through STEER.

        Args:
            seq: [batch, seq_len] item sequence
            seq_count: [batch, seq_len] cumulative count (Dynamic C)
            seq_age: [batch, seq_len] item age at interaction (Dynamic A)
        """
        batch_size, seq_len = seq.shape
        mask = (seq != 0).float()

        # Base embeddings
        all_emb = self.get_base_embeddings()
        base_emb = F.embedding(seq, all_emb) * self.emb_scale  # [batch, seq_len, d_model]

        # Position embeddings
        if not getattr(self.args, 'no_pos', False):
            positions = torch.arange(1, seq_len + 1, device=seq.device).unsqueeze(0).expand(batch_size, -1)
            positions = positions * mask.long()
            pos_enc = self.pos_emb(positions)

            # Determine which components get position
            add_all = getattr(self.args, 'add_pos_all', False)
            pos_q = add_all or getattr(self.args, 'add_pos_q', False)
            pos_k = add_all or getattr(self.args, 'add_pos_k', False)
            pos_v = add_all or getattr(self.args, 'add_pos_v', False)

            # Default: Q only (when no flags set)
            if not (pos_q or pos_k or pos_v):
                pos_q = True

            q_emb = base_emb + pos_enc if pos_q else base_emb.clone()
            k_emb = base_emb + pos_enc if pos_k else base_emb.clone()
            v_emb = base_emb + pos_enc if pos_v else base_emb.clone()
        else:
            q_emb = base_emb
            k_emb = base_emb.clone()
            v_emb = base_emb.clone()

        # Encode all temporal features
        all_encodings = {}

        # Static features (per item in sequence)
        t_first = self.item_t_first[seq]
        t_last = self.item_t_last[seq]
        n_users = self.item_n_users[seq]

        static_enc = self.temporal_encoder.encode_static(
            t_first, t_last, n_users, self.static_features
        )
        all_encodings.update(static_enc)

        # Dynamic features
        dynamic_enc = self.temporal_encoder.encode_dynamic(
            seq_age, seq_count, self.dynamic_features
        )
        all_encodings.update(dynamic_enc)

        # Cyclic features (need interaction time = t_first + age)
        if self.t_cyclic and seq_age is not None:
            t_interaction = t_first + seq_age.float()
            cyclic_enc = self.temporal_encoder.encode_cyclic(t_interaction, self.t_cyclic)
            all_encodings.update(cyclic_enc)

        # Route features to Q, K, V (learnable scale to match item embedding magnitude)
        if all_encodings:
            q_temp, k_temp, v_temp = self.temporal_router.route_features(all_encodings)
            q_emb = q_emb + q_temp
            k_emb = k_emb + k_temp
            v_emb = v_emb + v_temp

        # Apply dropout
        q_emb = self.emb_dropout(q_emb) * mask.unsqueeze(-1)
        k_emb = k_emb * mask.unsqueeze(-1)
        v_emb = v_emb * mask.unsqueeze(-1)

        # Cross-attention blocks
        for block in self.blocks:
            q_emb = block(q_emb, k_emb, v_emb, mask=mask, causality=True)

        return self.ln_final(q_emb)

    def predict(self, user_ids, seq, item_indices, seq_count=None, seq_age=None):
        """Predict scores for candidate items."""
        seq_emb = self.forward(seq, seq_count=seq_count, seq_age=seq_age)
        seq_emb = seq_emb[:, -1, :]

        item_emb_all = self.get_base_embeddings()

        if item_indices.dim() == 1:
            item_emb = item_emb_all[item_indices]
            scores = torch.matmul(seq_emb, item_emb.transpose(0, 1))
        else:
            item_emb = F.embedding(item_indices, item_emb_all)
            scores = torch.sum(seq_emb.unsqueeze(1) * item_emb, dim=-1)

        return scores

    def compute_loss(self, user_ids, seq, pos, neg, seq_count=None, seq_age=None):
        """Compute BPR training loss."""
        seq_emb = self.forward(seq, seq_count=seq_count, seq_age=seq_age)

        item_emb = self.get_base_embeddings()
        pos_emb = F.embedding(pos, item_emb)
        neg_emb = F.embedding(neg, item_emb)

        pos_logits = torch.sum(seq_emb * pos_emb, dim=-1)
        neg_logits = torch.sum(seq_emb * neg_emb, dim=-1)

        mask = (pos != 0).float()
        bpr_loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-24) * mask
        return bpr_loss.sum() / mask.sum()

    def set_item_features(self, item_features, device):
        """Set item temporal features."""
        if item_features is not None:
            self.item_t_first = item_features['t_first'].to(device)
            self.item_t_last = item_features['t_last'].to(device)
            self.item_n_users = item_features['n_users'].to(device)

    def get_item_embeddings(self):
        """Get item embeddings for candidate scoring."""
        return self.get_base_embeddings()

    def get_routing_weights(self):
        """Get current routing weights for visualization."""
        return self.temporal_router.get_routing_weights()
