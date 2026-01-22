import torch
import torch.nn as nn

class NewsEncoder(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        # Project từ 1024 (Voyage) về 128 (Model Dim)
        self.projector = nn.Linear(input_dim, dim)
        self.act = nn.GELU()

    def forward(self, s_n):
        """
        s_n: (B, T, 1024)
        output: (B, T, dim)
        """
        return self.act(self.projector(s_n))

# import torch
# import torch.nn as nn

# class NewsEncoder(nn.Module):
#     def __init__(self, input_dim, dim, dropout=0.1):
#         super().__init__()
        
#         # News vector (1024) rất thưa (sparse) và nhiễu.
#         # Cần nén nó lại (Bottleneck) trước khi đưa vào model.
        
#         hidden_dim = dim * 2 # Ví dụ: 64 * 2 = 128
        
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), # 1024 -> 128
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim)     # 128 -> 64
#         )

#     def forward(self, x):
#         return self.mlp(x)