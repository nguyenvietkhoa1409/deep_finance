
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef

from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig, GlobalConfig

# --- Configuration ---
COMPARE_EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ModalityTestModel(StockMovementModel):
    def __init__(self, test_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_type = test_type # 'news_price' or 'macro_price' or 'full'

    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None, mode="train"):
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

        if self.test_type == 'news_price':
            # Combo A: News + Price (Skip Macro)
            v_final = self.fusion_news(stable=v_i, unstable=v_n)
        elif self.test_type == 'macro_price':
            # Combo B: Price + Macro (Skip News)
            v_final = self.fusion_macro(stable=v_i, unstable=v_m)
        else:
            # Full: (Price + News) + Macro
            v_pn = self.fusion_news(stable=v_i, unstable=v_n)
            v_final = self.fusion_macro(stable=v_pn, unstable=v_m)

        logits = self.movement_predictor(fused_seq=v_final, orig_seq=v_i)

        if mode == "inference":
            return logits

        target = self._prepare_target(label)
        if mode == "train":
            return self.loss_fn(logits, target)
        elif mode == "test":
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
            return acc, mcc
        return logits

# Re-using main.py helpers (simplified)
class StockDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict
    def __len__(self): return len(self.data["label"])
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def collate_fn(batch):
    res = {}
    for k in batch[0].keys():
        res[k] = torch.stack([item[k] for item in batch])
    return res

def train_and_eval(test_type, train_data, valid_data, test_data):
    print(f"\n--- Testing Configuration: {test_type.upper()} ---")
    set_seed(42)
    
    s_m_dim = train_data["s_m"].shape[-1]
    
    model = ModalityTestModel(
        test_type=test_type,
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        device=DEVICE
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_loader = DataLoader(StockDataset(train_data), batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    best_mcc = -1
    best_acc = 0

    for epoch in range(COMPARE_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(
                s_o=batch["s_o"].to(DEVICE), s_h=batch["s_h"].to(DEVICE), s_c=batch["s_c"].to(DEVICE),
                s_m=batch["s_m"].to(DEVICE), s_n=batch["s_n"].to(DEVICE), label=batch["label"].to(DEVICE),
                mode="train"
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Eval on valid
        model.eval()
        with torch.no_grad():
            v_acc, v_mcc = model(
                s_o=valid_data["s_o"].to(DEVICE), s_h=valid_data["s_h"].to(DEVICE), s_c=valid_data["s_c"].to(DEVICE),
                s_m=valid_data["s_m"].to(DEVICE), s_n=valid_data["s_n"].to(DEVICE), label=valid_data["label"].to(DEVICE),
                mode="test"
            )
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:02d}/{COMPARE_EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val MCC: {v_mcc:.4f}")

            if v_mcc > best_mcc:
                best_mcc = v_mcc
                # Quick test on test set
                t_acc, t_mcc = model(
                    s_o=test_data["s_o"].to(DEVICE), s_h=test_data["s_h"].to(DEVICE), s_c=test_data["s_c"].to(DEVICE),
                    s_m=test_data["s_m"].to(DEVICE), s_n=test_data["s_n"].to(DEVICE), label=test_data["label"].to(DEVICE),
                    mode="test"
                )
                best_acc = t_acc
                final_test_mcc = t_mcc

    print(f"Result for {test_type}: Test ACC={best_acc:.4f}, Test MCC={final_test_mcc:.4f} (Best Val MCC: {best_mcc:.4f})")
    return best_acc, final_test_mcc

if __name__ == "__main__":
    from main import merge_datasets # Reuse merge logic
    
    pkl_path = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    dp = data_prepare(pkl_path)
    
    tickers = getattr(GlobalConfig, "TICKERS", ['TSLA', 'AMZN', 'AAPL', 'MSFT', 'GOOGL', 'META', 'BA', 'WMT'])
    l_tr, l_va, l_te = [], [], []
    for t in tickers:
        tr, va, te = dp.prepare_data(t)
        l_tr.append(tr); l_va.append(va); l_te.append(te)
    
    f_tr = merge_datasets(l_tr, shuffle=True)
    f_va = merge_datasets(l_va, shuffle=False)
    f_te = merge_datasets(l_te, shuffle=False)

    results = {}
    results['News + Price'] = train_and_eval('news_price', f_tr, f_va, f_te)
    results['Price + Macro'] = train_and_eval('macro_price', f_tr, f_va, f_te)
    results['Full (P+N+M)'] = train_and_eval('full', f_tr, f_va, f_te)

    print("\n" + "="*40)
    print("FINAL MODALITY COMPARISON")
    print("="*40)
    for k, v in results.items():
        print(f"{k:<20}: ACC={v[0]:.4f}, MCC={v[1]:.4f}")
