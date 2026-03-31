
# FILE: src/predictor.py
import torch
import torch.nn as nn

class FinegrainedMovementPrediction(nn.Module):
    """
    [OPTIMIZED] Hybrid: Learnable Query + Price Context
    Sử dụng Pooling thay vì Flatten để tránh Overfitting trên dataset nhỏ
    """
    def __init__(self, dim, window_size, num_classes=3, dropout=0.1):
        super().__init__()
        
        # Learnable query (global pattern seeker)
        self.query_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Attention pooling
        self.attn_fused = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, batch_first=True, dropout=dropout
        )
        self.attn_orig = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, batch_first=True, dropout=dropout
        )
        
        # Price context encoder (Đã sửa: input chỉ là dim, không phải dim * window_size)
        self.price_context = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classifier (3*dim input)
        self.classifier = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.3), # [FIXED] Tăng Dropout ở lớp cuối lên 0.4 để ép model generalize
            nn.Linear(dim, num_classes)
        )

    def forward(self, fused_seq, orig_seq):
        B, T, D = fused_seq.shape
        
        # 1. Learnable query attention (fused)
        query = self.query_token.expand(B, -1, -1)
        h_fused, _ = self.attn_fused(query, fused_seq, fused_seq)
        h_fused = h_fused.squeeze(1)
        
        # 2. Learnable query attention (original)
        h_orig, _ = self.attn_orig(query, orig_seq, orig_seq)
        h_orig = h_orig.squeeze(1)
        
        # 3. [OPTIMIZED] Price context from ALL timesteps
        # Thay vì dàn phẳng (Flatten) sinh ra hàng trăm ngàn tham số thừa, 
        # dùng Global Average Pooling để lấy "ngữ cảnh giá trung bình" của cả cửa sổ.
        # Hoặc dùng orig_seq[:, -1, :] nếu bạn muốn nhấn mạnh vào ngày gần nhất.
        price_pooled = orig_seq.mean(dim=1)  # Shape: (B, D)
        price_context = self.price_context(price_pooled)  # (B, D)
        
        # 4. Combine all three
        combined = torch.cat([h_fused, h_orig, price_context], dim=-1)
        
        # 5. Classify
        logits = self.classifier(combined)
        
        return logits