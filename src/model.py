# FILE: src/model.py

import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F
from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction
# Import KG Encoder nếu có
try:
    from encoders.kg_encoder import KnowledgeGraphEncoder
except ImportError:
    pass

from configs.config import TrainConfig

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

class StockMovementModel(nn.Module):
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
        use_focal_loss=True,
        use_kg=False,  # [NEW] Flag bật KG
    ):
        super().__init__()
        self.device = device
        self.use_kg = use_kg

        # 1. Encoders
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim, macro_dim=macro_dim, news_dim=news_dim, dim=dim
        )
        
        # [NEW] KG Encoder
        if self.use_kg:
            # Đảm bảo import đúng
            from encoders.kg_encoder import KnowledgeGraphEncoder
            self.kg_encoder = KnowledgeGraphEncoder(
                input_dim=getattr(TrainConfig, 'kg_input_dim', 772),
                hidden_dim=getattr(TrainConfig, 'kg_hidden_dim', 128),
                output_dim=getattr(TrainConfig, 'kg_output_dim', dim),
                dropout=getattr(TrainConfig, 'kg_dropout', 0.1)
            )
            self.fusion_kg = StableGatedCrossAttention(dim=dim, num_head=num_head)
            print("🕸️  Knowledge Graph Module: ENABLED")

        # 2. Fusion
        self.fusion_news = StableGatedCrossAttention(dim=dim, num_head=num_head)
        self.fusion_macro = StableGatedCrossAttention(dim=dim, num_head=num_head)

        # 3. Predictor
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout
        )

        # 4. Loss
        if use_focal_loss:
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
            print("🔧 Loss Strategy: FOCAL LOSS")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # [CRITICAL FIX] Thêm s_kg=None vào signature
    def forward(self, s_o, s_h, s_c, s_m, s_n, s_kg=None, label=None, mode="train"):
        # 1. Encode Features
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

        # 2. Fusion
        fused_news = self.fusion_news(primary=v_i, aux=v_n)
        fused_macro = self.fusion_macro(primary=v_i, aux=v_m)
        
        v_fused_total = (fused_news + fused_macro) / 2.0

        # [NEW] KG Integration
        if self.use_kg and s_kg is not None:
            # Check s_kg valid (list of list of Data)
            if isinstance(s_kg, list) and len(s_kg) > 0:
                v_kg = self.kg_encoder(s_kg) # (B, T, dim)
                v_kg = v_kg.to(v_i.device)
                
                fused_kg = self.fusion_kg(primary=v_i, aux=v_kg)
                # Average 3 modalities
                v_fused_total = (fused_news + fused_macro + fused_kg) / 3.0

        # 3. Predict
        logits = self.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
        logits = torch.clamp(logits, -15, 15)

        if mode == "train":
            if label is None:
                raise ValueError("Label cannot be None in train mode")
            
            if isinstance(label, list):
                target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
            else:
                target = label.long().to(self.device)
            
            loss = self.loss_fn(logits, target)
            return loss

        elif mode == "test":
            if label is not None:
                if isinstance(label, list):
                    target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
                else:
                    target = label.long().to(self.device)
                preds = torch.argmax(logits, dim=1)
                acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
                mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
                return acc, mcc
            return 0.0, 0.0
            
        elif mode == "inference":
            return logits