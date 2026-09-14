import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class IndicatorSequenceEncoder(nn.Module):
    """
    Price indicator encoder (report Section 3.3, Eq. 1-4).

    Eq.1: per-indicator linear projection (open/high/close independently).
    Eq.2: concatenation + linear projection -> aggregated representation.
    Eq.3: Bidirectional GRU over the aggregated sequence.
    Eq.4: residual connection (aggregated + BiGRU output) followed by LayerNorm.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim

        self.proj_o = nn.Sequential(
            spectral_norm(nn.Linear(1, dim)),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_h = nn.Sequential(
            spectral_norm(nn.Linear(1, dim)),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_c = nn.Sequential(
            spectral_norm(nn.Linear(1, dim)),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.combine = nn.Sequential(
            spectral_norm(nn.Linear(3 * dim, dim)),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        # Eq.3: Bidirectional GRU, hidden size = dim // 2 so that the
        # concatenated forward/backward states land back in `dim`.
        assert dim % 2 == 0, "IndicatorSequenceEncoder dim must be even for BiGRU concat."
        self.bigru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Eq.4: residual connection + LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, s_o, s_h, s_c):
        """
        s_o, s_h, s_c: (B, T, 1)
        """
        v_o = self.proj_o(s_o)
        v_h = self.proj_h(s_h)
        v_c = self.proj_c(s_c)

        v_agg = self.combine(torch.cat([v_o, v_h, v_c], dim=-1))  # Eq.2

        v_gru, _ = self.bigru(v_agg)  # Eq.3: (B, T, dim//2 * 2) = (B, T, dim)

        v_i = self.norm(v_agg + self.dropout(v_gru))  # Eq.4
        return v_i
