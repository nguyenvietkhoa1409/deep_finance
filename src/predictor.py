# FILE: src/predictor.py
import torch
import torch.nn as nn

from src.modules.layers import ThreeLayerMLP


class FinegrainedMovementPrediction(nn.Module):
    """
    Prediction layer (report Section 3.5, Eq. 17-20).

    g(.): a three-layer MLP that progressively collapses the time dimension,
    applied independently (separate weights) to the fused representation and
    to the original price representation (parallel stream, preserving the
    unmodified price signal as a direct reference).

    Eq.17/18: g(fused_seq), g(orig_seq)  -> (B, dim) each
    Eq.19: concatenate along the feature dimension -> (B, 2*dim)
    Eq.20: linear projection (W_c, b_c) onto the 3-class logit space
    """

    def __init__(self, dim, window_size, num_classes=3, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        d_in = window_size * dim
        d_h1 = dim * 4
        d_h2 = dim * 2

        # Eq.17: time-dim compression for the fused representation
        self.g_fused = ThreeLayerMLP(
            d_in=d_in, d_out=dim, d_h1=d_h1, d_h2=d_h2,
            final_activation=True, dropout=dropout
        )
        # Eq.18: time-dim compression for the original price representation
        self.g_orig = ThreeLayerMLP(
            d_in=d_in, d_out=dim, d_h1=d_h1, d_h2=d_h2,
            final_activation=True, dropout=dropout
        )

        # Eq.20: (W_c, b_c) linear projection to 3-class logits
        self.classifier = nn.Linear(2 * dim, num_classes)

    def forward(self, fused_seq, orig_seq):
        """
        fused_seq, orig_seq: (B, T, dim)
        """
        B = fused_seq.shape[0]

        h_fused = self.g_fused(fused_seq.reshape(B, -1))  # Eq.17 -> (B, dim)
        h_orig = self.g_orig(orig_seq.reshape(B, -1))     # Eq.18 -> (B, dim)

        m = torch.cat([h_fused, h_orig], dim=-1)  # Eq.19 -> (B, 2*dim)
        logits = self.classifier(m)               # Eq.20 -> (B, num_classes)
        return logits
