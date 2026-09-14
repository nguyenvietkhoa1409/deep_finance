from torch import nn

class MacroIndicatorEncoder(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim),
            # nn.GELU(),
            # nn.Linear(dim, dim)
        )

    def forward(self, s_m):
        """
        s_m: (T, num_macro) hoặc (batch, T, num_macro)
        """
        v_m = self.mlp(s_m)
        return v_m
