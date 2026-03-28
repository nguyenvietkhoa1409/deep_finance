# FILE: main.py

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef

from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig, GlobalConfig

# --- 1. SETUP ---
def set_seed(seed: int):
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
class StockDataset(Dataset):
    def __init__(self, data_dict):
        self.s_o = data_dict["s_o"]
        self.s_h = data_dict["s_h"]
        self.s_c = data_dict["s_c"]
        self.s_m = data_dict["s_m"]
        self.s_n = data_dict["s_n"]
        self.label = data_dict["label"]
        self.s_kg = data_dict.get("s_kg")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        item = {
            "s_o": self.s_o[idx],
            "s_h": self.s_h[idx],
            "s_c": self.s_c[idx],
            "s_m": self.s_m[idx],
            "s_n": self.s_n[idx],
            "label": self.label[idx],
        }
        if self.s_kg is not None:
            item["s_kg"] = self.s_kg[idx]
        return item

def collate_fn(batch):
    s_o = torch.stack([item["s_o"] for item in batch])
    s_h = torch.stack([item["s_h"] for item in batch])
    s_c = torch.stack([item["s_c"] for item in batch])
    s_m = torch.stack([item["s_m"] for item in batch])
    s_n = torch.stack([item["s_n"] for item in batch])
    label = torch.stack([item["label"] for item in batch])
    
    s_kg = None
    if "s_kg" in batch[0]:
        if isinstance(batch[0]["s_kg"], list):
            s_kg = [item["s_kg"] for item in batch]
        else:
            s_kg = torch.stack([item["s_kg"] for item in batch])
        
    return {
        "s_o": s_o, "s_h": s_h, "s_c": s_c, "s_m": s_m, "s_n": s_n, 
        "label": label, "s_kg": s_kg
    }

def merge_datasets(list_of_dicts, shuffle: bool = False):
    if not list_of_dicts: return {}
    
    first_valid = next((d for d in list_of_dicts if d), None)
    if not first_valid: return {}
    keys = list(first_valid.keys())
    
    merged = {}
    for k in keys:
        if k == "s_kg":
            first_kg = next((d[k] for d in list_of_dicts if d and k in d), None)
            if first_kg is not None and isinstance(first_kg, list):
                parts = []
                for d in list_of_dicts:
                    if d and k in d:
                        parts.extend(d[k])
                if parts: merged[k] = parts
            else:
                parts = [d[k] for d in list_of_dicts if d and k in d and isinstance(d[k], torch.Tensor)]
                if parts: merged[k] = torch.cat(parts, dim=0)
        else:
            parts = [d[k] for d in list_of_dicts if d and k in d and isinstance(d[k], torch.Tensor)]
            if parts: merged[k] = torch.cat(parts, dim=0)
    
    if shuffle and "label" in merged:
        idx = torch.randperm(len(merged["label"]))
        for k in merged:
            if k == "s_kg" and isinstance(merged[k], list):
                merged[k] = [merged[k][i] for i in idx.tolist()]
            else:
                merged[k] = merged[k][idx]
    return merged

# --- 3. HELPER: WEIGHTS ---
def compute_class_weights(labels_tensor: torch.Tensor) -> torch.Tensor:
    labels = labels_tensor.detach().cpu().numpy()
    class_counts = np.bincount(labels, minlength=3)
    num_classes = len(class_counts)

    beta = 0.9999 
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / np.sum(weights) * num_classes
    weights = np.sqrt(weights)
    weights = weights / np.sum(weights) * num_classes

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print("\n⚖️  [TIER 1] Balanced Class Weights:")
    classes = ["DOWN", "FLAT", "UP"]
    for i, w in enumerate(weights):
        count = int(class_counts[i]) if i < len(class_counts) else 0
        print(f"   ► {classes[i]:<4}: Count={count:<4} | Weight={w:.4f}")
    return weights_tensor

# --- 4. EVALUATE ---
def evaluate(model: torch.nn.Module, data_dict: dict):
    if not data_dict or "label" not in data_dict or len(data_dict["label"]) == 0:
        return 0.0, 0.0
    model.eval()
    
    s_kg = data_dict.get("s_kg")
    
    with torch.no_grad():
        # [NEW] Ép nhánh News thành 0 hoàn toàn lúc đánh giá
        s_n_zero = torch.zeros_like(data_dict["s_n"]).to(device)
        
        acc, mcc = model(
            s_o=data_dict["s_o"].to(device),
            s_h=data_dict["s_h"].to(device),
            s_c=data_dict["s_c"].to(device),
            s_m=data_dict["s_m"].to(device),
            s_n=s_n_zero, # Truyền Zero Tensor
            s_kg=s_kg, 
            label=data_dict["label"].to(device),
            mode="test",
        )
    return float(acc), float(mcc)

# ======================================================================
# --- DIAGNOSTIC TEST: MACRO & KG ABLATION ---
# ======================================================================
def test_ablation_scenarios(model: torch.nn.Module, data_dict: dict, device):
    """Đánh giá đóng góp của Macro và KG bằng cách che từng nhánh."""
    print("\n" + "="*60)
    print("🧪 ABLATION TEST: ĐÁNH GIÁ ĐÓNG GÓP CỦA MACRO VÀ KG")
    print("="*60)
    
    s_o = data_dict["s_o"].to(device)
    s_h = data_dict["s_h"].to(device)
    s_c = data_dict["s_c"].to(device)
    orig_s_m = data_dict["s_m"].to(device)
    # News đã bị loại bỏ hoàn toàn khỏi phương trình
    s_n_zero = torch.zeros_like(data_dict["s_n"]).to(device) 
    orig_s_kg = data_dict.get("s_kg")
    labels = data_dict["label"].long().to(device)
    
    scenarios = [
        {"name": "BASELINE (Đầy đủ Price + Macro + KG)", "mask_macro": False, "mask_kg": False},
        {"name": "MASK MACRO (Chỉ dùng Price + KG)", "mask_macro": True, "mask_kg": False},
        {"name": "MASK KG (Chỉ dùng Price + Macro)", "mask_macro": False, "mask_kg": True},
        {"name": "MASK CẢ MACRO VÀ KG (Chỉ dùng Price)", "mask_macro": True, "mask_kg": True}
    ]
    
    model.eval()
    with torch.no_grad():
        for s in scenarios:
            print(f"\n🚀 Kịch bản: {s['name']}")
            
            # Mask dữ liệu
            s_m = torch.zeros_like(orig_s_m) if s["mask_macro"] else orig_s_m
            s_kg = None if s["mask_kg"] else orig_s_kg
            
            # Forward để lấy dự đoán
            logits = model(s_o, s_h, s_c, s_m, s_n_zero, s_kg=s_kg, mode="inference")
            preds = torch.argmax(logits, dim=1)
            
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            mcc = matthews_corrcoef(labels.cpu().numpy(), preds.cpu().numpy())
            
            # Thống kê distribution
            counts = torch.bincount(preds, minlength=3).cpu().numpy()
            print(f"   ► Phân phối : DOWN(0): {counts[0]} | FLAT(1): {counts[1]} | UP(2): {counts[2]}")
            print(f"   ► Hiệu suất : ACC: {acc:.4f} | MCC: {mcc:.4f}")

# ======================================================================

# --- 5. TRAIN ---
def train_model(train_data: dict, valid_data: dict, test_data: dict):
    if not train_data: return

    s_m_dim = train_data["s_m"].shape[-1]
    
    print("\n  Calculating Class Weights (Balancing Strategy)...")
    train_labels = train_data["label"]
    class_weights = compute_class_weights(train_labels).to(device)
    
    real_batch_size = getattr(TrainConfig, "batch_size", 128)
    print(f"   ► Batch Size: {real_batch_size}")
    
    train_loader = DataLoader(
        StockDataset(train_data), 
        batch_size=real_batch_size, 
        shuffle=True, 
        drop_last=False,
        collate_fn=collate_fn 
    )

    print(f"\n🚀 Initializing Model on {device}...")
    
    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim,                
        input_dim=TrainConfig.window_size,   
        output_dim=TrainConfig.output_dim,   
        num_head=TrainConfig.num_head,
        dropout=0.2,                          
        class_weights=None, # Tắt Class Weights để Focal Loss tự làm việc
        use_focal_loss=True,
        use_kg=TrainConfig.use_kg,
        device=device
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=getattr(TrainConfig, "learning_rate", 1e-4), 
        weight_decay=1e-3  
    )

    best_val_mcc = -1.0
    best_val_acc = -1.0 
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pt")

    print("\n⚔️  STARTING TRAINING...")

    warmup_epochs = 50 
    
    for epoch in range(int(TrainConfig.epoch_num)):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # [NEW] Loại bỏ News hoàn toàn
            s_n_zero = torch.zeros_like(batch["s_n"]).to(device)
            
            s_m_input = batch["s_m"].to(device)
            s_kg_input = batch.get("s_kg")
            
            
            loss = model(
                s_o=batch["s_o"].to(device), 
                s_h=batch["s_h"].to(device),
                s_c=batch["s_c"].to(device), 
                s_m=s_m_input,
                s_n=s_n_zero, # Đưa News rỗng vào
                s_kg=s_kg_input,
                label=batch["label"].to(device),
                mode="train"
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / max(num_batches, 1)

        # Validate
        val_acc, val_mcc = evaluate(model, valid_data)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | Val ACC {val_acc:.4f} | Val MCC {val_mcc:.4f}")

        if (epoch + 1) >= warmup_epochs:
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

    print("\n🏁 FINAL TEST & SANITY CHECK...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        
        print("🔍 Sanity Check on VALID SET:")
        val_acc_check, val_mcc_check = evaluate(model, valid_data) 
        print(f"   VALID RESULT -> ACC: {val_acc_check:.4f}, MCC: {val_mcc_check:.4f}")
        
        print("\n🔍 Run on TEST SET:")
        test_acc, test_mcc = evaluate(model, test_data)
        print(f"🏆 TEST RESULT  -> ACC: {test_acc:.4f}, MCC: {test_mcc:.4f}")
        
        # --- THỰC THI ABLATION TEST Ở ĐÂY ---
        test_ablation_scenarios(model, test_data, device)

    else:
        print("⚠️ No best model saved.")

# --- 6. MAIN ---
if __name__ == "__main__":
    pkl_path = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    
    use_kg = getattr(TrainConfig, "use_kg", False)
    use_hetero_kg = getattr(TrainConfig, "use_hetero_kg", True)  
    
    if use_kg:
        if use_hetero_kg:
            hybrid_kg_path = os.path.join(GlobalConfig.KG_CACHE_DIR, 'hetero_kg_graphs.pkl')
            kg_path = None
            print(f"🕸️  Using HYBRID Heterogeneous KG: {hybrid_kg_path}")
        else:
            kg_path = GlobalConfig.KG_PROCESSED_PATH
            hybrid_kg_path = None
            print(f"📊 Using LEGACY KG embeddings: {kg_path}")
    else:
        kg_path = None
        hybrid_kg_path = None
        print("⚠️  KG disabled")
    
    print(f"📦 Loading processed dataset from: {pkl_path}")
    if not os.path.exists(pkl_path):
        print(f"❌ File not found: {pkl_path}")
        raise SystemExit(1)
    
    dp = data_prepare(
        pkl_path, 
        kg_data_path=kg_path,
        hybrid_kg_path=hybrid_kg_path,
        use_hetero_kg=use_hetero_kg
    )
    
    target_tickers = getattr(GlobalConfig, "TICKERS", ["TSLA", "AMZN", "MSFT", "NFLX"])
    list_train, list_valid, list_test = [], [], []
    
    for ticker in target_tickers:
        try:
            tr, val, te = dp.prepare_data(ticker)
            if tr and len(tr.get("label", [])) > 0:
                list_train.append(tr)
                list_valid.append(val)
                list_test.append(te)
        except Exception as e:
            print(f"⚠️ Skip ticker {ticker}: {e}")

    final_train = merge_datasets(list_train, shuffle=True)
    final_valid = merge_datasets(list_valid, shuffle=False)
    final_test  = merge_datasets(list_test,  shuffle=False)

    if len(final_train) > 0:
        train_model(final_train, final_valid, final_test)