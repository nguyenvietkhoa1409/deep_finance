import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F

from encoders.multi_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction
from configs.config import TrainConfig


class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        num_classes = logits.size(1)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()

        pt = (probs * targets_onehot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_weight = alpha[targets]
            focal_loss = focal_loss * alpha_weight

        return focal_loss.mean()


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
        use_focal_loss=False   # khuyên tạm để False khi debug lại model
    ):
        super().__init__()
        self.device = device

        # 1) Encoders
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            news_dim=news_dim,
            dim=dim
        )
        
        # 2) Fusion layers
        self.fusion_news = StableGatedCrossAttention(dim=dim, num_head=num_head)
        self.fusion_macro = StableGatedCrossAttention(dim=dim, num_head=num_head)

        # 3) Predictor
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim,
            window_size=input_dim,
            num_classes=output_dim,
            dropout=dropout
        )

        # 5) Loss
        if use_focal_loss:
            if class_weights is None:
                print("⚠️ Warning: Focal Loss enabled but no weights provided.")
            self.loss_fn = BalancedFocalLoss(alpha=class_weights, gamma=2.0)
            print("🔧 Using Loss Strategy: BALANCED FOCAL LOSS")
        else:
            label_smoothing = getattr(TrainConfig, "label_smoothing", 0.0)
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
            ls_str = f" (label_smoothing={label_smoothing})" if label_smoothing > 0 else ""
            print(f"🔧 Using Loss Strategy: WEIGHTED CROSS ENTROPY{ls_str}")

    def _prepare_target(self, label):
        if isinstance(label, list):
            target = torch.tensor(
                [item[0] for item in label],
                dtype=torch.long,
                device=self.device
            )
        else:
            target = label.long().to(self.device)
        return target

    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None, mode="train"):
        # -------------------------------------------------
        # 1) Encode
        # Expect:
        # v_m: macro sequence
        # v_i: price sequence (primary / stable anchor)
        # v_n: news sequence
        # -------------------------------------------------
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

        # -------------------------------------------------
        # 2) Sequential fusion
        # Stage 1: Price + News
        # -------------------------------------------------
        v_pn = self.fusion_news(stable=v_i, unstable=v_n)

        # -------------------------------------------------
        # 3) Stage 2: (Price+News) + Macro
        # -------------------------------------------------
        v_pnm = self.fusion_macro(stable=v_pn, unstable=v_m)

        # -------------------------------------------------
        # 4) Predict
        # fused_seq: sequence already fused with temporal meaning richer
        # orig_seq : original price branch as stable reference
        # -------------------------------------------------
        logits = self.movement_predictor(fused_seq=v_pnm, orig_seq=v_i)

        if mode == "inference":
            return logits

        target = self._prepare_target(label)

        if mode == "train":
            loss = self.loss_fn(logits, target)
            return loss

        elif mode == "test":
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
            return acc, mcc

        else:
            raise ValueError(f"Unsupported mode: {mode}")