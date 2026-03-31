import torch
import torch.nn as nn

class StableGatedCrossAttention(nn.Module):
    """
    MSGCA Gated Cross-Attention Mechanism
    Paper Reference: MSGCA Equations 10-14 and 15-19
    """
    
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()
        
        # ===== STEP 1: Multi-Head Cross-Attention =====
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_head,
            batch_first=True,
            dropout=dropout
        )
        
        # ===== STEP 2: Gating Mechanism =====
        # nn.Linear đã tự động có sẵn tham số bias bên trong.
        self.W_a = nn.Linear(dim, dim)
        self.W_b = nn.Linear(dim, dim)

        # Initialize gate bias for safer start (Gate ≈ sigmoid(1) ≈ 0.73)
        # Giúp model không bị mù tín hiệu auxiliary ở những epoch đầu tiên
        nn.init.constant_(self.W_b.bias, 1.0)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, stable, unstable, mask=None, is_causal=False):
        """
        Đã đổi tên tham số thành stable, unstable để đồng bộ với gọi hàm ở model.py
        """
        
        # ========================================
        # STEP 1: UNSTABLE FUSION (Eq. 10 / Eq. 15)
        # ========================================
        H_unstable, _ = self.cross_attn(
            query=stable,
            key=unstable,
            value=unstable,
            key_padding_mask=mask,
            is_causal=is_causal,
            need_weights=False
        )
        
        # ========================================
        # STEP 2: STABLE GATING (Eq. 13-14 / Eq. 18-19)
        # ========================================
        # Transform unstable features (Eq. 13)
        H_a = self.W_a(self.dropout(H_unstable))
        
        # Generate gate from stable modality (Eq. 14)
        H_b = torch.sigmoid(self.W_b(stable))
        
        # ========================================
        # STEP 3: ELEMENT-WISE SELECTION (Eq. 12 / Eq. 17)
        # ========================================
        # Tuân thủ tuyệt đối paper: Không cộng Residual `stable + ...` ở đây.
        output = H_a * H_b
        
        return output