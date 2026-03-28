# # # FILE: src/model.py

import torch
from torch import nn
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss


# # FILE: src/model.py
# import torch
# from torch import nn
# from sklearn.metrics import accuracy_score, matthews_corrcoef
# import torch.nn.functional as F

# from encoders.mutil_encoder import MultimodalSourceEncoding
# from .fusion import StableGatedCrossAttention
# from .predictor import FinegrainedMovementPrediction
# from configs.config import TrainConfig

# # Import KG Encoder an toàn
# try:
#     from encoders.hetero_kg_encoder import HeteroKGSequenceEncoder
# except ImportError:
#     pass

# class StockMovementModel(nn.Module):
#     """
#     MSGCA Framework - Optimized Sequential Fusion (Re-ordered).
    
#     NEW ORDER: Price -> News -> Macro -> KG
#     REASON: Fuse High-Frequency Data (News) first, then Low-Frequency (Macro).
#     """
#     def __init__(
#         self,
#         price_dim,
#         macro_dim,
#         news_dim,
#         dim,
#         input_dim,
#         output_dim,
#         num_head,
#         device, 
#         dropout=0.1, 
#         class_weights=None,
#         use_focal_loss=True, 
#         use_kg=True,
#         label_smoothing = TrainConfig.label_smoothing
#     ):
#         super().__init__()
#         self.device = device
#         self.use_kg = use_kg
#         self.label_smoothing = label_smoothing
#         # 1. Encoders
#         self.multimodal_encoder = MultimodalSourceEncoding(
#             price_dim=price_dim, macro_dim=macro_dim, news_dim=news_dim, dim=dim
#         )
        
#         if self.use_kg:
#             # [UPDATED] Use heterogeneous encoder
#             self.kg_encoder = HeteroKGSequenceEncoder(
#                 ticker_input_dim=1028,      # Ticker node dim
#                 event_input_dim=2061,       # Hybrid event node dim
#                 hidden_dim=256,             # Increased for richer features
#                 output_dim=dim,             # Must match MSGCA dim (128)
#                 num_heads=4,
#                 dropout=dropout
#             )
#             print("🕸️  Heterogeneous KG Module: ENABLED")

#         # 2. Sequential Fusion Modules (Re-ordered)
#         self.fuse_with_news = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        
#         self.fuse_with_macro = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        
#         # Bước 3: Context & KG (Giữ nguyên vị trí cuối)
#         if self.use_kg:
#             self.fuse_with_kg = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
            
#         # 3. Predictor
#         self.movement_predictor = FinegrainedMovementPrediction(
#             dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout
#         )   
            
#         # 4. Loss Function
#         if use_focal_loss:
#             # Ưu tiên Focal Loss với Gamma = 3.0
#             self.loss_fn = FocalLoss(alpha=class_weights, gamma=3.0)
#             print("🔧 Loss Strategy: FOCAL LOSS (Gamma=3.0)")
#         else:
#             self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
#             print(f"🔧 Loss Strategy: CrossEntropy with Label Smoothing={label_smoothing}")
            

#     def forward(self, s_o, s_h, s_c, s_m, s_n, s_kg=None, label=None, mode="train"):
#         # --- 1. Encoding ---
#         v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)  
#         # --- 2. Sequential Fusion (Re-ordered + Residual Anchor) ---
#         fused_1 = self.fuse_with_news(stable=v_i, unstable=v_n)  # Price dẫn dắt News
#         fused_2 = self.fuse_with_macro(stable=fused_1, unstable=v_m)  # Price+News dẫn dắt Macro
        
#         if self.use_kg and s_kg is not None:
#             v_kg = self.kg_encoder(s_kg)
#             v_kg = v_kg.to(v_i.device)
#             stable_for_kg = fused_2 + v_i  # "Neo" bằng Price gốc trước khi vào KG
#             fused_final = self.fuse_with_kg(stable=stable_for_kg, unstable=v_kg)  # Price+News+Macro dẫn dắt KG
#         else:
#             fused_final = fused_2  # Nếu không dùng KG, lấy kết quả sau Macro fusion
#         # --- 3. Prediction ---
#         logits = self.movement_predictor(fused_seq=fused_final, orig_seq=v_i)
#         logits = torch.clamp(logits, -15, 15)

#         # --- 4. Return Output ---
#         if mode == "train":
#             if label is None: raise ValueError("Label is None in train mode")
#             target = label if torch.is_tensor(label) else torch.tensor([x[0] for x in label], device=self.device)
#             target = target.long().to(self.device)
#             return self.loss_fn(logits, target)

#         elif mode == "test":
#             target = label if torch.is_tensor(label) else torch.tensor([x[0] for x in label], device=self.device)
#             target = target.long().to(self.device)
#             preds = torch.argmax(logits, dim=1)
#             acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
#             mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
#             return acc, mcc
            
#         elif mode == "inference":
#             return logits


import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F

from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction
from configs.config import TrainConfig

try:
    from encoders.hetero_kg_encoder import HeteroKGSequenceEncoder
except ImportError:
    pass

class StockMovementModel(nn.Module):
    def __init__(
        self, price_dim, macro_dim, news_dim, dim, input_dim, output_dim,
        num_head, device, dropout=0.1, class_weights=None,use_focal_loss=True, use_kg=True
    ):
        super().__init__()
        self.device = device
        self.use_kg = use_kg
        self.label_smoothing = getattr(TrainConfig, 'label_smoothing', 0.1)

        # 1. Encoders
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim, macro_dim=macro_dim, news_dim=news_dim, dim=dim
        )
        
        if self.use_kg:
            self.kg_encoder = HeteroKGSequenceEncoder(
                ticker_input_dim=1028, event_input_dim=2061,
                hidden_dim=256, output_dim=dim, num_heads=4, dropout=dropout
            )
            print("🕸️  Heterogeneous KG Module: ENABLED")

        # 2. Parallel-Anchored Sequential Fusion Modules
        # Chú ý: Cấu trúc đã được làm sạch, giữ nguyên số lượng tham số
        self.fusion_news = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        self.fusion_macro = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        if self.use_kg:
            self.fusion_kg = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)

        # 3. Predictor
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout
        )

        # 4. Loss Function
        if use_focal_loss:
            # Ưu tiên Focal Loss với Gamma = 3.0
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=3.0)
            print("🔧 Loss Strategy: FOCAL LOSS (Gamma=3.0)")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=self.label_smoothing)
            print(f"🔧 Loss Strategy: CrossEntropy with Label Smoothing={self.label_smoothing}")

    def forward(self, s_o, s_h, s_c, s_m, s_n, s_kg=None, label=None, mode="train"):
        # --- 1. Encoding ---
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

        # --- 2. Parallel-Anchored Fusion ---
        # GIẢI PHÁP: Luôn dùng v_i (Price thuần) làm 'stable' (Query & Gate) để đánh giá.
        # Điều này ngăn chặn nhiễu từ News làm mù Gate của Macro/KG.
        
        # Bước 1: Khai thác News dựa trên Price
        fused_news = self.fusion_news(stable=v_i, unstable=v_n)
        
        # Bước 2: Khai thác Macro dựa trên Price
        fused_macro = self.fusion_macro(stable=v_i, unstable=v_m)
        
        # Bước 3: Khai thác KG dựa trên Price
        fused_kg = 0
        if self.use_kg and s_kg is not None and isinstance(s_kg, list) and len(s_kg) > 0:
            v_kg = self.kg_encoder(s_kg).to(v_i.device)
            fused_kg = self.fusion_kg(stable=v_i, unstable=v_kg)

        # Tích hợp theo luồng (Additive Context)
        # Bằng cách cộng dồn, tín hiệu Price (v_i) được củng cố dần qua từng lớp thông tin 
        # mà không làm hỏng cơ chế Gating.
        h_final = v_i + fused_news + fused_macro + fused_kg

        # --- 3. Prediction ---
        logits = self.movement_predictor(fused_seq=h_final, orig_seq=v_i)
        
        # ĐÃ XÓA: torch.clamp(logits, -15, 15) -> Tránh làm chết Gradient!

        # --- 4. Return Output ---
        if mode == "train":
            # KHUYẾN NGHỊ: Hãy di chuyển logic xử lý list này sang class Dataset
            # Code dưới đây chỉ xử lý tạm thời để tránh lỗi.
            target = label if torch.is_tensor(label) else torch.tensor([x[0] for x in label], dtype=torch.long, device=self.device)
            return self.loss_fn(logits, target.long().to(self.device))

        elif mode == "test":
            target = label if torch.is_tensor(label) else torch.tensor([x[0] for x in label], dtype=torch.long, device=self.device)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
            return acc, mcc
            
        elif mode == "inference":
            return logits