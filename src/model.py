# # FILE: src/model.py

# import torch
# from torch import nn
# from sklearn.metrics import accuracy_score, matthews_corrcoef
# import torch.nn.functional as F
# from encoders.mutil_encoder import MultimodalSourceEncoding
# from .fusion import StableGatedCrossAttention
# from .predictor import FinegrainedMovementPrediction
# # Import KG Encoder nếu có
# try:
#     from encoders.kg_encoder import KnowledgeGraphEncoder
# except ImportError:
#     pass

# from configs.config import TrainConfig

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
#         pt = torch.exp(-ce_loss)
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss
#         if self.reduction == 'mean': return focal_loss.mean()
#         elif self.reduction == 'sum': return focal_loss.sum()
#         else: return focal_loss

# class StockMovementModel(nn.Module):
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
#         use_kg=False,  # [NEW] Flag bật KG
#     ):
#         super().__init__()
#         self.device = device
#         self.use_kg = use_kg

#         # 1. Encoders
#         self.multimodal_encoder = MultimodalSourceEncoding(
#             price_dim=price_dim, macro_dim=macro_dim, news_dim=news_dim, dim=dim
#         )
        
#         # [NEW] KG Encoder
#         if self.use_kg:
#             # Đảm bảo import đúng
#             from encoders.kg_encoder import KnowledgeGraphEncoder
#             self.kg_encoder = KnowledgeGraphEncoder(
#                 input_dim=getattr(TrainConfig, 'kg_input_dim', 772),
#                 hidden_dim=getattr(TrainConfig, 'kg_hidden_dim', 128),
#                 output_dim=getattr(TrainConfig, 'kg_output_dim', dim),
#                 dropout=getattr(TrainConfig, 'kg_dropout', 0.1)
#             )
#             self.fusion_kg = StableGatedCrossAttention(dim=dim, num_head=num_head)
#             print("🕸️  Knowledge Graph Module: ENABLED")

#         # 2. Fusion
#         self.fusion_news = StableGatedCrossAttention(dim=dim, num_head=num_head)
#         self.fusion_macro = StableGatedCrossAttention(dim=dim, num_head=num_head)
        
#         # 3. Predictor
#         self.movement_predictor = FinegrainedMovementPrediction(
#             dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout
#         )

#         # 4. Loss
#         if use_focal_loss:
#             self.loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
#             print("🔧 Loss Strategy: FOCAL LOSS")
#         else:
#             self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

#     # [CRITICAL FIX] Thêm s_kg=None vào signature
#     def forward(self, s_o, s_h, s_c, s_m, s_n, s_kg=None, label=None, mode="train"):
#         # 1. Encode Features
#         v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

#         # 2. Fusion
#         fused_news = self.fusion_news(primary=v_i, aux=v_n)
#         fused_macro = self.fusion_macro(primary=v_i, aux=v_m)
        
#         v_fused_total = (fused_news + fused_macro) / 2.0

#         # [NEW] KG Integration
#         if self.use_kg and s_kg is not None:
#             # Check s_kg valid (list of list of Data)
#             if isinstance(s_kg, list) and len(s_kg) > 0:
#                 v_kg = self.kg_encoder(s_kg) # (B, T, dim)
#                 v_kg = v_kg.to(v_i.device)
                
#                 fused_kg = self.fusion_kg(primary=v_i, aux=v_kg)
#                 # Average 3 modalities
#                 v_fused_total = (fused_news + fused_macro + fused_kg) / 3.0

#         # 3. Predict
#         logits = self.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
#         logits = torch.clamp(logits, -15, 15)

#         if mode == "train":
#             if label is None:
#                 raise ValueError("Label cannot be None in train mode")
            
#             if isinstance(label, list):
#                 target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
#             else:
#                 target = label.long().to(self.device)
            
#             loss = self.loss_fn(logits, target)
#             return loss

#         elif mode == "test":
#             if label is not None:
#                 if isinstance(label, list):
#                     target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
#                 else:
#                     target = label.long().to(self.device)
#                 preds = torch.argmax(logits, dim=1)
#                 acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
#                 mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
#                 return acc, mcc
#             return 0.0, 0.0
            
#         elif mode == "inference":
#             return logits


# FILE: src/model.py
import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F

from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction
from configs.config import TrainConfig

# Import KG Encoder an toàn
try:
    from encoders.hetero_kg_encoder import KnowledgeGraphEncoder
except ImportError:
    pass

# FILE: src/model.py
import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F

from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction
from configs.config import TrainConfig

# Import KG Encoder an toàn
try:
    from encoders.hetero_kg_encoder import KnowledgeGraphEncoder
except ImportError:
    pass

# FILE: src/model.py
import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F

from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction
from configs.config import TrainConfig

# Import KG Encoder an toàn
try:
    from encoders.hetero_kg_encoder import HeteroKGSequenceEncoder
except ImportError:
    pass

class StockMovementModel(nn.Module):
    """
    MSGCA Framework - Optimized Sequential Fusion (Re-ordered).
    
    NEW ORDER: Price -> News -> Macro -> KG
    REASON: Fuse High-Frequency Data (News) first, then Low-Frequency (Macro).
    """
    def __init__(
        self,
        price_dim,
        macro_dim,
        news_dim,
        dim,
        input_dim,
        output_dim,
        num_head,
        device, 
        dropout=0.1, 
        class_weights=None,
        use_focal_loss=False, 
        use_kg=True,
        label_smoothing = TrainConfig.label_smoothing
    ):
        super().__init__()
        self.device = device
        self.use_kg = use_kg
        self.label_smoothing = label_smoothing
        # 1. Encoders
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim, macro_dim=macro_dim, news_dim=news_dim, dim=dim
        )
        
        if self.use_kg:
            # [UPDATED] Use heterogeneous encoder
            self.kg_encoder = HeteroKGSequenceEncoder(
                ticker_input_dim=1028,      # Ticker node dim
                event_input_dim=1805,       # Hybrid event node dim
                hidden_dim=256,             # Increased for richer features
                output_dim=dim,             # Must match MSGCA dim (128)
                num_heads=4,
                dropout=dropout
            )
            print("🕸️  Heterogeneous KG Module: ENABLED")

        # 2. Sequential Fusion Modules (Re-ordered)
        
        # [MODIFIED] Bước 1: Price & News (Thay vì Macro)
        # News biến động nhanh, tương quan cao với Price -> Fuse trước
        self.fusion_price_news = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        
        # [MODIFIED] Bước 2: Context & Macro
        # Macro là xu hướng dài hạn, đóng vai trò điều chỉnh Context
        self.fusion_with_macro = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        
        # Bước 3: Context & KG (Giữ nguyên vị trí cuối)
        if self.use_kg:
            self.fusion_with_kg = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)

        # 3. Predictor
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout
        )

        # 4. Loss Function
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        print(f"🔧 Loss Strategy: CrossEntropy with Label Smoothing={label_smoothing}")

    def forward(self, s_o, s_h, s_c, s_m, s_n, s_kg=None, label=None, mode="train"):
        # --- 1. Encoding ---
        # v_i: Price (Indicator) - Stable Anchor
        # v_n: News (Unstable 1)
        # v_m: Macro (Unstable 2)
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

        # --- 2. Sequential Fusion (Re-ordered + Residual Anchor) ---
        
        # [STEP 1]: Price dẫn dắt News (Quan trọng nhất)
        # Query = v_i (Price)
        # Key/Value = v_n (News)
        h_chain = self.fusion_price_news(stable=v_i, unstable=v_n)
        
        # [STEP 2]: (Price + News) dẫn dắt Macro
        # [CRITICAL FIX]: Cộng lại v_i (Price) để "neo" tín hiệu.
        # Nếu không cộng, h_chain có thể đã bị biến đổi quá xa so với Price gốc.
        stable_for_macro = h_chain + v_i 
        h_chain = self.fusion_with_macro(stable=stable_for_macro, unstable=v_m)
        
        # [STEP 3]: (Price + News + Macro) dẫn dắt KG
        if self.use_kg and s_kg is not None:
            if isinstance(s_kg, list) and len(s_kg) > 0:
                v_kg = self.kg_encoder(s_kg)
                v_kg = v_kg.to(v_i.device)
                
                # Tiếp tục "neo" bằng v_i
                stable_for_kg = h_chain + v_i
                h_chain = self.fusion_with_kg(stable=stable_for_kg, unstable=v_kg)

        # --- 3. Prediction ---
        # Predictor vẫn đối chiếu kết quả chuỗi (h_chain) với Price gốc (v_i)
        logits = self.movement_predictor(fused_seq=h_chain, orig_seq=v_i)
        
        # v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)
        # # Sequential fusion WITHOUT intermediate residuals
        # h1 = self.fusion_price_news(stable=v_i, unstable=v_n)
        # h2 = self.fusion_with_macro(stable=h1, unstable=v_m)  # ← Remove + v_i
        
        # if self.use_kg and s_kg is not None:
        #     v_kg = self.kg_encoder(s_kg)
        #     h3 = self.fusion_with_kg(stable=h2, unstable=v_kg)  # ← Remove + v_i
        # else:
        #     h3 = h2
        
        # # Single residual at the end
        # h_final = h3 + v_i  # ← Only here
        
        # logits = self.movement_predictor(fused_seq=h_final, orig_seq=v_i)
        
        # Clamp logits
        logits = torch.clamp(logits, -15, 15)

        # --- 4. Return Output ---
        if mode == "train":
            if label is None: raise ValueError("Label is None in train mode")
            target = label if torch.is_tensor(label) else torch.tensor([x[0] for x in label], device=self.device)
            target = target.long().to(self.device)
            return self.loss_fn(logits, target)

        elif mode == "test":
            target = label if torch.is_tensor(label) else torch.tensor([x[0] for x in label], device=self.device)
            target = target.long().to(self.device)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
            return acc, mcc
            
        elif mode == "inference":
            return logits