# =========================================================
# FILE: main.py
# =========================================================

import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset # [NEW] Import DataLoader

from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig

# --- 1. SETUP ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if TrainConfig.use_cuda and torch.cuda.is_available() else "cpu")
set_seed(TrainConfig.seed)

# --- 2. HELPER: DATASET & MERGE ---
# [NEW CLASS] Wrapper để dùng DataLoader
class StockDataset(Dataset):
    def __init__(self, data_dict):
        self.s_o = data_dict["s_o"]
        self.s_h = data_dict["s_h"]
        self.s_c = data_dict["s_c"]
        self.s_m = data_dict["s_m"]
        self.s_n = data_dict["s_n"]
        self.label = data_dict["label"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            "s_o": self.s_o[idx],
            "s_h": self.s_h[idx],
            "s_c": self.s_c[idx],
            "s_m": self.s_m[idx],
            "s_n": self.s_n[idx],
            "label": self.label[idx]
        }

def merge_datasets(list_of_dicts, shuffle=False):
    if not list_of_dicts: return {}
    keys = list_of_dicts[0].keys()
    merged_data = {}
    for key in keys:
        tensors = [d[key] for d in list_of_dicts if d and key in d]
        if tensors:
            merged_data[key] = torch.cat(tensors, dim=0)
    
    if shuffle and "label" in merged_data:
        indices = torch.randperm(len(merged_data["label"]))
        for key in merged_data:
            merged_data[key] = merged_data[key][indices]
    return merged_data

# --- 3. HELPER: WEIGHTS (STABLE TIER 1) ---
def compute_class_weights(labels_tensor):
    """
    [TIER 1] Balanced Class Weights (Effective-Sqrt)
    Logic ổn định cũ mà bạn muốn giữ lại.
    """
    labels = labels_tensor.cpu().numpy()
    class_counts = np.bincount(labels)
    num_classes = len(class_counts)
    
    # 1. Effective Number of Samples
    beta = 0.9999 
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    
    # Normalize
    weights = weights / np.sum(weights) * num_classes
    
    # 2. Square Root Smoothing (Softening)
    weights = np.sqrt(weights)
    weights = weights / np.sum(weights) * num_classes
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    print(f"\n⚖️  [TIER 1] Balanced Class Weights (Effective-Sqrt):")
    classes = ['DOWN', 'FLAT', 'UP']
    for i, w in enumerate(weights):
        count = class_counts[i] if i < len(class_counts) else 0
        print(f"   ► {classes[i]:<4}: Count={count:<4} | Weight={w:.4f}")
        
    return weights_tensor

# --- 4. EVALUATE ---
def evaluate(model, data_dict):
    if not data_dict: return 0.0, 0.0
    model.eval()
    with torch.no_grad():
        # Evaluate có thể chạy Full Batch cho nhanh và chính xác
        acc, mcc = model(
            data_dict["s_o"].to(device), data_dict["s_h"].to(device),
            data_dict["s_c"].to(device), data_dict["s_m"].to(device),
            data_dict["s_n"].to(device), data_dict["label"].to(device),
            mode="test"
        )
    return acc, mcc

# --- 5. TRAIN ---
def train_model(train_data, valid_data, test_data):
    if not train_data: return

    s_m_dim = train_data["s_m"].shape[-1]
    
    # 1. Calculate Weights (Stable Logic)
    print("\n  Calculating Class Weights (Balancing Strategy)...")
    train_labels = train_data["label"]
    class_weights = compute_class_weights(train_labels).to(device)
    
    # 2. Setup DataLoader (Đây là phần bạn yêu cầu thêm vào)
    train_dataset = StockDataset(train_data)
    real_batch_size = TrainConfig.batch_size # Lấy từ Config (ví dụ 32 hoặc 64)
    print(f"   ► Batch Size: {real_batch_size}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=real_batch_size, 
        shuffle=True, # Shuffle mỗi epoch
        drop_last=False
    )

    print(f"\n🚀 Initializing Model on {device}...")
    print(f"   ► Strategy: FOCAL LOSS (Gamma=2.0) + ALPHA BALANCING")
    
    # 3. Init Model (Stable Version)
    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim,                 
        input_dim=TrainConfig.window_size,   
        output_dim=TrainConfig.output_dim,   
        num_head=TrainConfig.num_head,
        dropout=0.1,                         
        class_weights=class_weights, # Dùng Tier 1 Weights
        use_focal_loss=True,         # Dùng Improved Focal Loss (Temp=1.5)
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TrainConfig.learning_rate, 
        weight_decay=1e-4
    )

    best_val_mcc = -1.0
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pt")

    print("\n⚔️  STARTING TRAINING...")

    # 4. Training Loop (Epoch -> Batch)
    for epoch in range(TrainConfig.epoch_num):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # [NEW] Loop qua từng Batch
        for batch in train_loader:
            optimizer.zero_grad()
            
            loss = model(
                batch["s_o"].to(device), batch["s_h"].to(device),
                batch["s_c"].to(device), batch["s_m"].to(device),
                batch["s_n"].to(device), batch["label"].to(device),
                mode="train"
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        # Tính Loss trung bình của Epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # Validate
        val_acc, val_mcc = evaluate(model, valid_data)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | Val ACC {val_acc:.4f} | Val MCC {val_mcc:.4f}")

        # Save Best Model
        is_best = False
        if val_mcc > best_val_mcc:
            is_best = True
        elif val_mcc == best_val_mcc and val_acc > best_val_acc:
            is_best = True
            
        if is_best:
            best_val_mcc = val_mcc
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   >>> New Best Model Saved! (MCC: {val_mcc:.4f} - Acc: {val_acc:.4f})")

    # Final Check
    print("\n🏁 FINAL TEST & SANITY CHECK...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        
        print("🔍 Sanity Check on VALID SET:")
        val_acc_check, val_mcc_check = evaluate(model, valid_data) 
        print(f"   VALID RESULT -> ACC: {val_acc_check:.4f}, MCC: {val_mcc_check:.4f}")
        
        print("\n🔍 Run on TEST SET:")
        test_acc, test_mcc = evaluate(model, test_data)
        print(f"🏆 TEST RESULT  -> ACC: {test_acc:.4f}, MCC: {test_mcc:.4f}")
    else:
        print("⚠️ No best model saved.")

if __name__ == "__main__":
    pkl_path = r"D:\DeepFinance\data\processed\unified_dataset_test.pkl"
    dp = data_prepare(pkl_path)
    
    target_tickers = ["TSLA", "AMZN", "MSFT", "NFLX"] 
    list_train, list_valid, list_test = [], [], []
    
    for ticker in target_tickers:
        try:
            tr, val, te = dp.prepare_data(ticker)
            if tr and len(tr.get("label", [])) > 0:
                list_train.append(tr); list_valid.append(val); list_test.append(te)
        except: pass

    final_train = merge_datasets(list_train, shuffle=True)
    final_valid = merge_datasets(list_valid, shuffle=False)
    final_test  = merge_datasets(list_test,  shuffle=False)

    if len(final_train) > 0:
        train_model(final_train, final_valid, final_test)