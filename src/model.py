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
    from encoders.kg_encoder import KnowledgeGraphEncoder
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
    from encoders.kg_encoder import KnowledgeGraphEncoder
except ImportError:
    pass

class StockMovementModel(nn.Module):
    """
    MSGCA Framework - Fully Sequential Fusion Architecture.
    
    Logic bám sát Paper:
    Fuse từng module một theo chuỗi (Cascade), dùng kết quả bước trước 
    làm 'Stable Feature' để dẫn dắt bước sau.
    
    Flow:
      1. Price (Stable gốc) + Macro -> Context_1
      2. Context_1 (Stable mới) + News  -> Context_2
      3. Context_2 (Stable mới) + KG    -> Final Representation
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
        use_kg=False,
    ):
        super().__init__()
        self.device = device
        self.use_kg = use_kg

        # 1. Encoders (Đưa mọi thứ về chiều `dim`)
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim, macro_dim=macro_dim, news_dim=news_dim, dim=dim
        )
        
        if self.use_kg:
            from encoders.kg_encoder import KnowledgeGraphEncoder
            # Cấu hình KG Encoder
            self.kg_encoder = KnowledgeGraphEncoder(
                input_dim=getattr(TrainConfig, 'kg_input_dim', 772),
                hidden_dim=getattr(TrainConfig, 'kg_hidden_dim', 128),
                output_dim=getattr(TrainConfig, 'kg_output_dim', dim),
                dropout=getattr(TrainConfig, 'kg_dropout', 0.1)
            )
            print("🕸️  KG Module Initialized (Sequential Chain Mode)")

        # 2. Sequential Fusion Modules (Từng bước một)
        
        # Bước 1: Fusion Price & Macro
        # Macro thường bổ trợ trực tiếp cho Price về mặt xu hướng vĩ mô
        self.fusion_price_macro = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        
        # Bước 2: Fusion (Price+Macro) & News
        # Dùng ngữ cảnh kinh tế đã fuse để tìm tin tức liên quan
        self.fusion_with_news = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        
        # Bước 3: Fusion (Price+Macro+News) & KG
        # KG là lớp thông tin ngữ nghĩa sâu nhất, được fuse cuối cùng
        if self.use_kg:
            self.fusion_with_kg = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)

        # 3. Predictor (Dynamic Attention Pooling)
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout
        )

        # 4. Loss Function
        # Sử dụng Label Smoothing để tránh Class Collapse (lớp Flat)
        # Giữ nguyên khuyến nghị dùng CE thay vì Focal Loss cho dữ liệu nhỏ/nhiễu
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print("🔧 Loss Strategy: CrossEntropy with Label Smoothing=0.1")

    def forward(self, s_o, s_h, s_c, s_m, s_n, s_kg=None, label=None, mode="train"):
        # --- 1. Encoding ---
        # v_i: Price (Indicator) - Stable Feature gốc
        # v_m: Macro
        # v_n: News
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

        # --- 2. Fully Sequential Fusion (Cascade) ---
        
        # [STEP 1]: Price dẫn dắt Macro
        # Query (Stable) = v_i (Price)
        # Key/Value (Unstable) = v_m (Macro)
        h_chain = self.fusion_price_macro(stable=v_i, unstable=v_m)
        
        # [STEP 2]: Kết quả Step 1 dẫn dắt News
        # Query (Stable) = h_chain (Price + Macro)
        # Key/Value (Unstable) = v_n (News)
        h_chain = self.fusion_with_news(stable=h_chain, unstable=v_n)
        
        # [STEP 3]: Kết quả Step 2 dẫn dắt Knowledge Graph (Nếu có)
        if self.use_kg and s_kg is not None:
            if isinstance(s_kg, list) and len(s_kg) > 0:
                # Encode KG
                v_kg = self.kg_encoder(s_kg) # [B, T, D]
                v_kg = v_kg.to(v_i.device)
                
                # Query (Stable) = h_chain (Price + Macro + News)
                # Key/Value (Unstable) = v_kg (Graph)
                h_chain = self.fusion_with_kg(stable=h_chain, unstable=v_kg)
            # Nếu không có KG, h_chain giữ nguyên kết quả từ Step 2

        # h_chain bây giờ là vector đại diện cuối cùng (H_final)

        # --- 3. Prediction ---
        # Predictor dùng h_chain làm context, và v_i (Price gốc) để lấy Dynamic Query
        # Điều này đảm bảo ta luôn đối chiếu lại với biến động giá thực tế
        logits = self.movement_predictor(fused_seq=h_chain, orig_seq=v_i)
        
        # Clamp logits (Safety)
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