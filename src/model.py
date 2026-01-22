# =========================================================
# FILE: src/model.py (STABLE VERSION)
# =========================================================

import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F  
from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction    

class ImprovedFocalLoss(nn.Module):
    """
    Focal Loss thế hệ mới khắc phục lỗi Double Penalty:
    1. Decoupled Alpha: Alpha (Weights) được nhân sau cùng.
    2. Temperature Scaling: Làm mềm Logits trước khi tính xác suất.
    3. Label Smoothing: Chống Overconfidence.
    """
    def __init__(self, alpha=None, gamma=2.0, temperature=1.5, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        # 1. Temperature Scaling
        logits = logits / self.temperature
        
        # 2. Softmax
        probs = F.softmax(logits, dim=1)
        
        # 3. One-hot & Label Smoothing
        num_classes = logits.size(1)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
        
        if self.label_smoothing > 0:
            targets_onehot = targets_onehot * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
        
        # 4. Focal Term
        pt = (probs * targets_onehot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        
        # 5. Base CE
        ce_loss = -(targets_onehot * torch.log(probs + 1e-8)).sum(dim=1)
        
        # 6. Combine
        loss = focal_weight * ce_loss
        
        # 7. Apply Alpha
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            weight_per_sample = alpha[targets]
            loss = loss * weight_per_sample
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
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
    ):
        super().__init__()
        self.device = device

        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim, macro_dim=macro_dim, news_dim=news_dim, dim=dim
        )
        self.fusion_news = StableGatedCrossAttention(dim=dim, num_head=num_head)
        self.fusion_macro = StableGatedCrossAttention(dim=dim, num_head=num_head)
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout
        )

        if use_focal_loss:
            if class_weights is None:
                print("⚠️ Warning: Focal Loss enabled but no weights provided.")
            
            # STABLE TIER 2 CONFIG
            self.loss_fn = ImprovedFocalLoss(
                alpha=class_weights, 
                gamma=2.0, 
                temperature=1.5,      
                label_smoothing=0.1,  
                reduction='mean'
            )
            print("🔧 Using Loss Strategy: [TIER 2] IMPROVED FOCAL LOSS (Temp=1.5, Smooth=0.1)")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print("🔧 Using Loss Strategy: WEIGHTED CROSS ENTROPY")
        
    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None, mode="train"):
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)
        fused_news = self.fusion_news(primary=v_i, aux=v_n)
        fused_macro = self.fusion_macro(primary=v_i, aux=v_m)
        v_fused_total = (fused_news + fused_macro) / 2.0
        logits = self.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
        logits = torch.clamp(logits, -15, 15)

        if mode == "train":
            if isinstance(label, list):
                target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
            else:
                target = label.long().to(self.device)
            loss = self.loss_fn(logits, target)
            return loss

        elif mode == "test":
            if isinstance(label, list):
                target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
            else:
                target = label.long().to(self.device)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
            return acc, mcc