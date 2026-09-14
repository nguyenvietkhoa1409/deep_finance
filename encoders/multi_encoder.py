import torch.nn as nn

from .indicator_encoder import IndicatorSequenceEncoder
from .macro_encoder import MacroIndicatorEncoder
from .news_encoder import NewsEncoder


class MultimodalSourceEncoding(nn.Module):
    """
    Multimodal encoding layer (report Section 3.3): projects price, macro,
    and news modalities into the shared d-dimensional latent space.
    """

    def __init__(self, price_dim, macro_dim, news_dim, dim):
        super().__init__()

        self.indicator_encoder = IndicatorSequenceEncoder(dim)
        self.macro_encoder = MacroIndicatorEncoder(in_dim=macro_dim, dim=dim)
        self.news_encoder = NewsEncoder(input_dim=news_dim, dim=dim)

    def forward(self, s_o, s_h, s_c, s_m, s_n):
        """
        Input: (B, T, Features) tensors per modality.
        Output: 3 tensors (B, T, dim).
        """
        v_i = self.indicator_encoder(s_o, s_h, s_c)
        v_m = self.macro_encoder(s_m)
        v_n = self.news_encoder(s_n)

        return v_m, v_i, v_n
