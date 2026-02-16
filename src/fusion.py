# import torch
# import torch.nn as nn

# class StableGatedCrossAttention(nn.Module):
#     """
#     Stable Gated Cross-Attention Mechanism
#     Uses Indicator/Price features (Primary) to guide the selection of 
#     Auxiliary features via a gating mechanism.
#     """
#     def __init__(self, dim, num_head, dropout=0.1):
#         super().__init__()
        
#         # 1. Unstable Fusion (Standard Multi-Head Attention)
#         # Note: In PyTorch MHA, batch_first=True expects (Batch, Seq, Feature)
#         # Output: H^l_{i,d}
#         self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_head, batch_first=True, dropout=dropout)
        
#         # 2. Gating Mechanism 
#         # Gate is generated from the Primary modality
#         # Input: H_i (Primary/Indicator) -> Output: H_b (Gate)
#         self.gate_linear = nn.Linear(dim, dim)
#         self.sigmoid = nn.Sigmoid()
        
#         #3. Feature transformation weights 
#         # Input: H^l_{i,d} (Unstable) -> Output: H_a (Transformed for gating)
#         self.transform_linear = nn.Linear(dim, dim)
        
#         # 4. Residual & Norm
#         self.norm = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, primary, aux):
#         """
#         Args:
#             primary (Tensor): The guiding modality (Price/Indicator). Shape: (B, T, D)
#             aux (Tensor): The auxiliary - unstable fusion modality (e.g., News, Macro). Shape: (B, T, D)
#         """
#         # --- Step 1: Unstable Fusion ---
#         # Query = Primary, Key = Aux, Value = Aux
#         # Get H^l_{i,d} from Cross-Attention
#         # unstable_fused shape: (B, T, D)
#         unstable_fused, _ = self.mha(query=primary, key=aux, value=aux)
        
        
#         # # --- Step 2: Calculate H_a (Eq. 13) ---
#         # # Ha = H^l_{i,d} * Wa + b
#         # H_a = self.transform_linear(unstable_fused)
        
#         # # --- Step 3: Calculate H_b (Eq. 14) ---
#         # # Hb = Sigmoid(Hi * Wb + b')
#         # H_b = self.sigmoid(self.gate_linear(primary))
        
#         # # --- Step 4: Stable Feature Selection (Eq. 12) ---
#         # # Output = Gate * New_Info + (1 - Gate) * Old_Info
#         # # Ý nghĩa: Gate quyết định bao nhiêu % thông tin mới được nạp vào, 
#         # # và bao nhiêu % thông tin cũ được giữ lại.
#         # H_id = H_a * H_b 
        
#         # # --- Step 5: Residual Connection & Norm ---
#         # # Note: The paper implies Hi,d is the stable feature. 
#         # # Usually, Transformer blocks add Residual connection with the Query (Primary).
#         # output = self.norm(primary + self.dropout(H_id))
        
#         # 2. GATING MECHANISM (SỬA LẠI ĐÚNG CHUẨN HIGHWAY)
#         # ---------------------------------------------------------
        
#         # A. Tính Gate (H_b): Quyết định dựa trên Primary (Giá)
#         # Gate chạy từ 0 đến 1. 
#         # Gate -> 1: Tin vào News/Macro (Aux)
#         # Gate -> 0: Tin vào Price (Primary)
#         gate = self.sigmoid(self.gate_linear(primary)) 

#         # B. Transform New Info (H_a)
#         H_a = self.transform_linear(unstable_fused)
        
#         # C. Áp dụng Gating (Highway Connection)
#         # Đây là dòng quan trọng nhất sửa lỗi "Leaky Residual"
        
#         # Nhánh 1: Thông tin Mới (News) được Gate cho phép đi qua
#         gated_new = H_a * gate
        
#         # Nhánh 2: Thông tin Cũ (Price) bị Gate chặn lại (Complementary Gate)
#         # Nếu Gate=1 (News tốt), thì (1-Gate)=0 -> Price nhiễu bị loại bỏ hoàn toàn.
#         gated_primary = primary * (1.0 - gate)
        
#         # 3. KẾT HỢP & NORM
#         # ---------------------------------------------------------
#         # Áp dụng Dropout lên phần New Info (nơi chứa nhiều tham số học được)
#         output = self.norm(gated_primary + self.dropout(gated_new))
        
#         return output




# import torch.nn as nn

# class StableGatedCrossAttention(nn.Module):
#     """
#     Stable Gated Cross-Attention Mechanism (STRICT PAPER IMPLEMENTATION VARIANT)
    
#     Trong phiên bản này, ta loại bỏ Residual Connection của Primary Modality ở đầu ra.
#     Mục tiêu: Buộc mô hình phải sử dụng thông tin đã được Gated (chọn lọc), 
#     thay vì dựa dẫm vào tín hiệu gốc của Price.
#     """
#     def __init__(self, dim, num_head, dropout=0.1):
#         super().__init__()
        
#         # 1. Unstable Fusion (Standard Multi-Head Attention)
#         # Query = Primary (Price)
#         # Key/Value = Aux (News/Macro)
#         self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_head, batch_first=True, dropout=dropout)
        
#         # 2. Gating Mechanism 
#         # H_b = Sigmoid(Linear(Primary))
#         self.gate_linear = nn.Linear(dim, dim)
#         self.sigmoid = nn.Sigmoid()
        
#         # 3. Feature Transformation
#         # H_a = Linear(Unstable_Fused)
#         self.transform_linear = nn.Linear(dim, dim)
        
#         # 4. Norm & Dropout
#         self.norm = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, primary, aux):
#         """
#         Args:
#             primary (Tensor): (B, T, D) - Price/Indicator
#             aux (Tensor): (B, T, D) - News/Macro
#         """
#         # --- Step 1: Unstable Fusion (Cross-Attention) ---
#         # Lấy thông tin từ Aux dựa trên sự chú ý của Primary
#         # unstable_fused: (B, T, D)
#         unstable_fused, _ = self.mha(query=primary, key=aux, value=aux)
        
#         # --- Step 2: Tính toán Gate (H_b) ---
#         # Gate dựa hoàn toàn vào Primary (Giá hiện tại)
#         # Gate ~ 1: Cho phép thông tin sau Attention đi qua
#         # Gate ~ 0: Chặn thông tin
#         gate = self.sigmoid(self.gate_linear(primary)) 
        
#         # --- Step 3: Transform Unstable Features (H_a) ---
#         H_a = self.transform_linear(unstable_fused)
        
#         # --- Step 4: Apply Gating (H_id) ---
#         # Công thức gốc (Eq. 12 trong MSGCA paper): H_id = H_a * H_b
#         # Lưu ý: Ở đây ta KHÔNG cộng lại Primary (No Residual with Primary)
#         # Ta chỉ lấy kết quả của phép nhân Gate.
#         H_id = H_a * gate
        
#         # --- Step 5: Output ---
#         # Normalize kết quả đã Gated
#         # Nếu Gate ~ 0, output sẽ xấp xỉ 0 (Blank signal), buộc các tầng sau phải tự xoay sở
#         # hoặc dựa vào nhánh Fusion khác (ví dụ Fusion Macro).
#         output = self.norm(self.dropout(H_id))
        
#         return output

# FILE: src/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StableGatedCrossAttention(nn.Module):
    """
    Cài đặt chính xác cơ chế 'Gated Cross-Feature Fusion' từ paper MSGCA.
    
    Logic:
    1. Cross-Attention: Dùng 'stable' (Query) để khai thác thông tin từ 'unstable' (Key/Value).
       -> Tạo ra 'unstable_fused' features (H_l_id).
    2. Gating Mechanism: Dùng chính 'stable' feature để tạo ra Gate.
       -> Gate quyết định giữ lại bao nhiêu thông tin từ 'unstable_fused'.
    
    Paper Equations (12-14) & (17-19):
        Ha = Linear(Attention_Out) + b
        Hb = Sigmoid(Linear(Stable_Feature) + c)
        Output = Ha * Hb
    """
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()
        
        # 1. Multi-Head Cross Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_head, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 2. Linear layers cho Gating mechanism
        # Wa trong Eq(13) / (18) - Biến đổi output của Attention
        self.linear_a = nn.Linear(dim, dim)
        
        # Wb trong Eq(14) / (19) - Biến đổi Stable feature để tạo Gate
        self.linear_b = nn.Linear(dim, dim)
        nn.init.constant_(self.linear_b.bias, -2.0)
        
        # Layer Norm để ổn định training (Thực tế nên có sau MHA)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stable, unstable, mask=None):
        """
        Args:
            stable (Query): Vector đặc trưng ổn định (VD: Price hoặc Fused_Level_1) [B, T, D]
            unstable (Key/Value): Vector đặc trưng bổ sung (VD: News, KG) [B, T, D]
        """
        # --- Bước 1: Unstable Cross-Attention Fusion (Eq 10-11) ---
        # Query = Stable, Key/Value = Unstable
        attn_out, _ = self.mha(query=stable, key=unstable, value=unstable, key_padding_mask=mask)
        attn_out = self.dropout(attn_out)
        
        # Residual connection & Norm (Chuẩn hóa Transformer)
        # Lưu ý: Paper không nói rõ Residual ở đây, nhưng đây là best practice. 
        # Tuy nhiên để bám sát logic Gating của paper, ta dùng output thuần của MHA cho vào Gate.
        h_unstable_fused = attn_out 

        # --- Bước 2: Stable Gated Feature Selection (Eq 12-14) ---
        
        # Tính Ha (Eq 13): Biến đổi đặc trưng vừa fuse
        h_a = self.linear_a(h_unstable_fused) # [B, T, D]
        h_a = self.dropout(h_a)
        # Tính Hb (Eq 14): Tính Gate dựa trên Stable Feature gốc
        # Đây là mấu chốt: Stable feature quyết định cái gì được đi qua
        gate = torch.sigmoid(self.linear_b(stable)) # [B, T, D]
        
        # Element-wise Product (Eq 12)
        h_final = h_a * gate
        
        # Thêm LayerNorm cuối cùng để ổn định đầu ra
        h_final = self.norm(self.dropout(h_final))
        
        return h_final