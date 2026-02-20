# import torch
# import torch.nn as nn
# from .modules.layers import ThreeLayerMLP

# class FinegrainedMovementPrediction(nn.Module):
#     """
#     Decoder module for Fine-grained Movement Prediction (Proposal Section 3.5).
#     Replaces RNNs with MLPs for dimension aggregation.
#     """
#     def __init__(self, dim, window_size, num_classes=3, dropout=0.0): # [NEW] Nhận tham số dropout
#         super().__init__()
        
#         # --- 1. Time Dimension Aggregation (Reduce T -> 1) ---
#         self.time_agg_fused = ThreeLayerMLP(
#             d_in=window_size, 
#             d_out=1, 
#             d_h1=window_size // 2, 
#             d_h2=window_size // 4, 
#             final_activation=True,
#             dropout=dropout # [NEW]
#         )
        
#         self.time_agg_orig = ThreeLayerMLP(
#             d_in=window_size, 
#             d_out=1, 
#             d_h1=window_size // 2, 
#             d_h2=window_size // 4, 
#             final_activation=True,
#             dropout=dropout # [NEW]
#         )
        
#         # --- 2. Feature Dimension Aggregation (Reduce 2D -> Classes) ---
#         self.feat_agg = ThreeLayerMLP(
#             d_in=2 * dim, 
#             d_out=num_classes, 
#             d_h1=dim, 
#             d_h2=dim // 2, 
#             final_activation=False, # No activation -> Raw Logits
#             dropout=dropout # [NEW]
#         )

#     def forward(self, fused_seq, orig_seq):
#         """
#         Args:
#             fused_seq: Multimodal features after fusion (B, T, D)
#             orig_seq: Original Indicator/Price features (B, T, D)
#         """
#         # Transpose to (B, D, T) so Linear layers operate on T dimension
#         fused_trans = fused_seq.transpose(1, 2)
#         orig_trans = orig_seq.transpose(1, 2)
        
#         # --- Step 1: Time Aggregation ---
#         h_fused = self.time_agg_fused(fused_trans).squeeze(-1)
#         h_orig = self.time_agg_orig(orig_trans).squeeze(-1)
        
#         # --- Step 2: Concatenation ---
#         h_final = torch.cat([h_fused, h_orig], dim=-1) # Shape: (B, 2D)
        
#         # --- Step 3: Feature Aggregation (Prediction) ---
#         logits = self.feat_agg(h_final) # Shape: (B, 3)
        
#         return logits
    

# import torch
# import torch.nn as nn

# class FinegrainedMovementPrediction(nn.Module):
#     """
#     Module dự đoán xu hướng giá (Predictor).
    
#     [MAJOR UPDATE]: Thay thế Time-Aggregation MLP bằng Attention Pooling.
#     - Cơ chế cũ: MLP(Time) -> Quá cứng nhắc, overfitting vào vị trí.
#     - Cơ chế mới: Attention(Query, Key=Seq, Val=Seq) -> Model tự học cách "nhặt"
#       thông tin quan trọng từ bất kỳ đâu trong chuỗi thời gian.
#     """
#     def __init__(self, dim, window_size, num_classes=3, dropout=0.1):
#         super().__init__()
        
#         # 1. Learnable Query Token
#         # Đây là một tham số học được, đóng vai trò như một "câu hỏi" trừu tượng:
#         # "Tôi cần tìm kiếm tín hiệu gì trong chuỗi quá khứ để dự đoán tương lai?"
#         # Shape: (1, 1, Dim)
#         self.query_token = nn.Parameter(torch.randn(1, 1, dim))
        
#         # 2. Temporal Attention Pooling Layers
#         # Dùng riêng 2 lớp Attention cho 2 nhánh (Fused và Original)
#         # để chúng có thể học các patterns thời gian khác nhau.
#         self.attn_fused = nn.MultiheadAttention(
#             embed_dim=dim, 
#             num_heads=4, 
#             batch_first=True, 
#             dropout=dropout
#         )
        
#         self.attn_orig = nn.MultiheadAttention(
#             embed_dim=dim, 
#             num_heads=4, 
#             batch_first=True, 
#             dropout=dropout
#         )
        
#         # 3. Final Classifier
#         # Input: 2 * dim (Do nối output của 2 nhánh lại)
#         self.classifier = nn.Sequential(
#             nn.Linear(2 * dim, dim),
#             nn.LayerNorm(dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim, num_classes)
#         )

#     def forward(self, fused_seq, orig_seq):
#         """
#         Args:
#             fused_seq (Tensor): (Batch, Time, Dim) - Chuỗi đã qua Fusion (chứa News/Macro)
#             orig_seq (Tensor): (Batch, Time, Dim) - Chuỗi Price gốc (Indicator)
#         Returns:
#             logits (Tensor): (Batch, Num_Classes)
#         """
#         B, T, D = fused_seq.shape
        
#         # Expand Query Token cho cả Batch
#         # Shape: (Batch, 1, Dim)
#         query = self.query_token.expand(B, -1, -1)
        
#         # --- 1. Attention Pooling: Fused Branch ---
#         # Query đi quét qua toàn bộ chuỗi fused_seq để tổng hợp thông tin
#         # Output: (Batch, 1, Dim)
#         h_fused, _ = self.attn_fused(query=query, key=fused_seq, value=fused_seq)
        
#         # Squeeze dimension thời gian (vì chỉ có 1 query token nên ra 1 vector)
#         h_fused = h_fused.squeeze(1)  # (Batch, Dim)
        
#         # --- 2. Attention Pooling: Original Branch ---
#         # Tương tự cho nhánh Price gốc
#         h_orig, _ = self.attn_orig(query=query, key=orig_seq, value=orig_seq)
#         h_orig = h_orig.squeeze(1)    # (Batch, Dim)
        
#         # --- 3. Concatenation & Classification ---
#         # Kết hợp thông tin từ cả 2 nhánh
#         h_final = torch.cat([h_fused, h_orig], dim=-1)  # (Batch, 2*Dim)
        
#         # Đưa vào Classifier để ra Logits
#         logits = self.classifier(h_final)
        
#         return logits

# FILE: src/predictor.py
# import torch
# import torch.nn as nn

# class FinegrainedMovementPrediction(nn.Module):
#     """
#     Predictor v2.0: Dynamic Attention Pooling.
#     Sử dụng trạng thái cuối cùng của chuỗi giá (Last Price State) làm Query 
#     để tổng hợp thông tin từ toàn bộ chuỗi Fused Features.
#     """
#     def __init__(self, dim, window_size, num_classes=3, dropout=0.1):
#         super().__init__()
        
#         # Attention Pooling cho nhánh Fused Features
#         self.attn_pool = nn.MultiheadAttention(
#             embed_dim=dim, 
#             num_heads=4, 
#             batch_first=True, 
#             dropout=dropout
#         )
        
#         # Final Classifier
#         # Input dim = dim * 2 (Vì ta nối Fused Context + Last Price State)
#         self.classifier = nn.Sequential(
#             nn.Linear(dim * 2, dim),
#             nn.LayerNorm(dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim, num_classes)
#         )

#     def forward(self, fused_seq, orig_seq):
#         """
#         Args:
#             fused_seq: [B, T, D] - Kết quả cuối cùng của chuỗi Fusion
#             orig_seq:  [B, T, D] - Chuỗi Price gốc (để lấy Dynamic Query)
#         """
#         # 1. Lấy Dynamic Query: Trạng thái của ngày cuối cùng (t)
#         # Đây là thông tin quan trọng nhất: "Vị thế thị trường hôm nay"
#         last_step_price = orig_seq[:, -1:, :] # [B, 1, D]
        
#         # 2. Attention Pooling
#         # Hỏi: "Dựa trên giá hôm nay, hãy quét quá khứ (fused_seq) để tìm tín hiệu?"
#         context, _ = self.attn_pool(
#             query=last_step_price,
#             key=fused_seq,
#             value=fused_seq
#         )
#         # context: [B, 1, D] -> squeeze -> [B, D]
#         context = context.squeeze(1)
        
#         # 3. Feature Combination
#         # Kết hợp Context tìm được với chính trạng thái hiện tại (Residual-like)
#         last_step_flat = last_step_price.squeeze(1) # [B, D]
        
#         combined = torch.cat([context, last_step_flat], dim=-1) # [B, 2D]
        
#         # 4. Classify
#         logits = self.classifier(combined)
        
#         return logits

# FILE: src/predictor.py
import torch
import torch.nn as nn

class FinegrainedMovementPrediction(nn.Module):
    """
    [ALTERNATIVE] Hybrid: Learnable Query + Price Context
    Simpler than V4 but still uses ALL timesteps
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
        
        # Price context encoder (uses ALL timesteps)
        self.price_context = nn.Sequential(
            nn.Linear(dim * window_size, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classifier (now 3*dim input)
        self.classifier = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
        
        # 3. [NEW] Price context from ALL timesteps
        price_flat = orig_seq.reshape(B, -1)  # (B, T*D)
        price_context = self.price_context(price_flat)  # (B, D)
        
        # 4. Combine all three
        combined = torch.cat([h_fused, h_orig, price_context], dim=-1)
        
        # 5. Classify
        logits = self.classifier(combined)
        
        return logits