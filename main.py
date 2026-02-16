# FILE: main.py

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
        # Load s_kg nếu có (có thể là Tensor hoặc List[HeteroData])
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

# Custom Collate để xử lý cả Tensor và List of Graphs
def collate_fn(batch):
    s_o = torch.stack([item["s_o"] for item in batch])
    s_h = torch.stack([item["s_h"] for item in batch])
    s_c = torch.stack([item["s_c"] for item in batch])
    s_m = torch.stack([item["s_m"] for item in batch])
    s_n = torch.stack([item["s_n"] for item in batch])
    label = torch.stack([item["label"] for item in batch])
    
    s_kg = None
    if "s_kg" in batch[0]:
        # Check if it's a list (hetero graphs) or tensor (legacy)
        if isinstance(batch[0]["s_kg"], list):
            # Hetero graphs: keep as list
            s_kg = [item["s_kg"] for item in batch]
        else:
            # Legacy tensor: stack
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
            # Check type of first element
            first_kg = next((d[k] for d in list_of_dicts if d and k in d), None)
            
            if first_kg is not None and isinstance(first_kg, list):
                # List of graphs: extend
                parts = []
                for d in list_of_dicts:
                    if d and k in d:
                        parts.extend(d[k])
                if parts: merged[k] = parts
            else:
                # Tensor: concatenate
                parts = [d[k] for d in list_of_dicts if d and k in d and isinstance(d[k], torch.Tensor)]
                if parts:
                    merged[k] = torch.cat(parts, dim=0)
        else:
            # Regular tensor merge
            parts = [d[k] for d in list_of_dicts if d and k in d and isinstance(d[k], torch.Tensor)]
            if parts:
                merged[k] = torch.cat(parts, dim=0)
    
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
        acc, mcc = model(
            s_o=data_dict["s_o"].to(device),
            s_h=data_dict["s_h"].to(device),
            s_c=data_dict["s_c"].to(device),
            s_m=data_dict["s_m"].to(device),
            s_n=data_dict["s_n"].to(device),
            s_kg=s_kg, 
            label=data_dict["label"].to(device),
            mode="test",
        )
    return float(acc), float(mcc)

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
        class_weights=class_weights, 
        use_focal_loss=True,
        use_kg=TrainConfig.use_kg,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=getattr(TrainConfig, "learning_rate", 1e-4), 
        weight_decay=getattr(TrainConfig, "weight_decay", 1e-4)
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
            
            loss = model(
                s_o=batch["s_o"].to(device), 
                s_h=batch["s_h"].to(device),
                s_c=batch["s_c"].to(device), 
                s_m=batch["s_m"].to(device),
                s_n=batch["s_n"].to(device), 
                s_kg=batch.get("s_kg"),
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
    else:
        print("⚠️ No best model saved.")

# --- 6. MAIN ---
if __name__ == "__main__":
    pkl_path = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    
    # [UPDATED] Support both legacy KG and hybrid KG
    use_kg = getattr(TrainConfig, "use_kg", False)
    use_hetero_kg = getattr(TrainConfig, "use_hetero_kg", True)  # Default to new mode
    
    if use_kg:
        if use_hetero_kg:
            # Use new hybrid KG
            hybrid_kg_path = os.path.join(GlobalConfig.KG_CACHE_DIR, 'hetero_kg_graphs.pkl')
            kg_path = None
            print(f"🕸️  Using HYBRID Heterogeneous KG: {hybrid_kg_path}")
        else:
            # Use legacy KG
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
    
    # [UPDATED] Init loader with both KG options
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