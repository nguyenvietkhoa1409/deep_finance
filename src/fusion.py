import torch
import torch.nn as nn


class StableGatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention (GCA) fusion stage (report Section 3.4, Eq. 9-14).

    Used twice in sequence (Section 3.4 "Sequential fusion", Eq. 15-16):
      Stage 1: primary=price,       auxiliary=news  -> price-news representation
      Stage 2: primary=price-news,  auxiliary=macro -> final joint representation
    """

    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # Eq.9-11: multi-head cross-attention (query=primary, key/value=auxiliary)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_head,
            batch_first=True,
            dropout=dropout
        )

        # Eq.12-14: gated feature selection
        self.W_a = nn.Linear(dim, dim)
        self.W_b = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, stable, unstable, mask=None, is_causal=False):
        """
        stable: primary/anchor modality representation (B, T, dim)
        unstable: auxiliary modality representation (B, T, dim)
        """
        # Eq.9-11: cross-attention, query from primary, key/value from auxiliary
        H_unstable, _ = self.cross_attn(
            query=stable,
            key=unstable,
            value=unstable,
            key_padding_mask=mask,
            is_causal=is_causal,
            need_weights=False
        )

        # Eq.12: candidate features from the cross-attended output
        H_a = self.W_a(self.dropout(H_unstable))

        # Eq.13: soft selection gate derived strictly from the stable primary anchor
        H_b = torch.sigmoid(self.W_b(stable))

        # Eq.14: element-wise gating -- suppresses auxiliary signal where it
        # conflicts with the primary context. No residual add to `stable`
        # here: the report's gated fusion keeps the gated auxiliary signal as
        # the stage output, which is then carried forward as the next
        # stage's primary input (Eq.15-16).
        H_gated = H_a * H_b
        output = self.norm(self.dropout(H_gated))

        return output
