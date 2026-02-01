# =========================================================
# FILE: src/model.py
# =========================================================

import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F  
from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction    

class BalancedFocalLoss(nn.Module):
    """
    [TIER 3] SIMPLIFIED BALANCED FOCAL LOSS
    Loại bỏ Temperature Scaling và Label Smoothing để tránh xung đột gradient.
    Chỉ giữ lại Focal Term (tập trung mẫu khó) và Alpha Weighting (cân bằng lớp).
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        # 1. Compute probabilities (Standard Softmax)
        probs = F.softmax(logits, dim=1)
        
        # 2. Get target probabilities (Hard One-hot, No Smoothing)
        num_classes = logits.size(1)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
        
        # 3. Compute Pt (Probability of ground truth class)
        pt = (probs * targets_onehot).sum(dim=1)
        
        # 4. Focal Term: (1 - pt)^gamma
        # Nếu model tự tin (pt -> 1) => weight -> 0
        # Nếu model bối rối (pt -> 0) => weight -> 1
        focal_weight = (1 - pt) ** self.gamma
        
        # 5. Base CE Loss
        # ce_loss = - log(pt)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 6. Apply Focal Weight
        focal_loss = focal_weight * ce_loss
        
        # 7. Apply Class Weights (Alpha)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_weight = alpha[targets]
            focal_loss = focal_loss * alpha_weight
            
        return focal_loss.mean()

class StockMovementModel(nn.Module):
    """
    MSGCA Framework Implementation (Refactored).
    Integrates Stable Gated Fusion and Attention-based Predictor.
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
        use_focal_loss=True,    
    ):
        super().__init__()
        self.device = device

        # Encoders
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim, macro_dim=macro_dim, news_dim=news_dim, dim=dim
        )
        
        # Fusion Layers (Updated to No-Residual Gating if you applied that change)
        self.fusion_news = StableGatedCrossAttention(dim=dim, num_head=num_head)
        self.fusion_macro = StableGatedCrossAttention(dim=dim, num_head=num_head)
        
        # Predictor (Updated to Attention Pooling)
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout
        )

        # Loss Function
        if use_focal_loss:
            if class_weights is None:
                print("⚠️ Warning: Focal Loss enabled but no weights provided.")
            
            # [TIER 3 CONFIG] BalancedFocalLoss
            # Gamma=2.0 là chuẩn mực. Không còn Temp/Smooth.
            self.loss_fn = BalancedFocalLoss(
                alpha=class_weights, 
                gamma=2.0 
            )
            print("🔧 Using Loss Strategy: [TIER 3] BALANCED FOCAL LOSS (No Temp/Smooth)")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print("🔧 Using Loss Strategy: WEIGHTED CROSS ENTROPY")
        
    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None, mode="train"):
        # 1. Encode
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)
        
        # 2. Fusion
        fused_news = self.fusion_news(primary=v_i, aux=v_n)
        fused_macro = self.fusion_macro(primary=v_i, aux=v_m)
        v_fused_total = (fused_news + fused_macro) / 2.0
        
        # 3. Predict
        logits = self.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
        
        # Clamp logits để tránh NaN khi train lâu
        logits = torch.clamp(logits, -15, 15)

        # 4. Return based on mode
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
            
        elif mode == "inference":
            return logits