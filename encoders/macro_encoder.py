from torch import nn

class MacroIndicatorEncoder(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.projector = nn.Linear(in_dim, dim)

    def forward(self, s_m):
        """
        s_m: (T, num_macro) hoặc (batch, T, num_macro)
        """
        v_m = self.projector(s_m)
        return v_m

# class MacroIndicatorEncoder(nn.Module):
#     def __init__(self, in_dim, dim, dropout=0.1):
#         super().__init__()
        
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, dim), # Project lên không gian lớn
#             nn.GELU(),              # Non-linearity (Quan trọng nhất)
#             nn.Linear(dim, dim)     # Refine features
#         )
#         # Lưu ý: Không dùng LayerNorm ở đây nữa vì data quá ít, 
#         # LayerNorm sẽ làm "phẳng" hết tín hiệu macro yếu ớt.

#     def forward(self, x):
#         return self.mlp(x)