import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from configs.config import TrainConfig


class NewsEncoder(nn.Module):
    """
    News modality encoder (report Section 3.3, Eq. 5-8).

    Input s_n is the precomputed daily news embedding sequence (produced
    upstream by the news preprocessing pipeline: LLM S-R-O triple extraction
    -> quality filtering -> text embedding -> confidence-weighted daily
    aggregation per ticker).

    Eq.5: temporal-decay weighting over the observation window, maximized at
    the forecast date T (i.e. news closer to the forecast horizon carries a
    stronger predictive signal).
    Eq.6-7: two-stage linear projection (W1, W2) with GELU activations,
    LayerNorm applied before each subsequent transformation.
    """

    def __init__(self, input_dim, dim, decay_rate=None):
        super().__init__()
        self.decay_rate = decay_rate if decay_rate is not None else getattr(TrainConfig, "news_temporal_decay", 0.1)

        mid_dim = max(dim, 256)

        self.norm_in = nn.LayerNorm(input_dim)
        self.stage1 = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, mid_dim)),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.norm_mid = nn.LayerNorm(mid_dim)
        self.stage2 = nn.Sequential(
            spectral_norm(nn.Linear(mid_dim, dim)),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def _temporal_decay_weights(self, T, device, dtype):
        """
        Eq.5: weight_t = exp(-decay_rate * (T - t)), t = 0 (earliest day in
        the window) .. T-1 (forecast date). Weight is maximized at t = T-1.
        """
        t = torch.arange(T, device=device, dtype=dtype)
        weights = torch.exp(-self.decay_rate * (T - 1 - t))
        return weights.view(1, T, 1)  # broadcast over (B, T, 1)

    def forward(self, s_n):
        """
        s_n: (B, T, input_dim) - precomputed daily news embeddings.
        output: (B, T, dim)
        """
        B, T, _ = s_n.shape

        decay_w = self._temporal_decay_weights(T, s_n.device, s_n.dtype)
        s_n = s_n * decay_w  # Eq.5

        x = self.stage1(self.norm_in(s_n))       # Eq.6
        output = self.stage2(self.norm_mid(x))   # Eq.7
        return output
