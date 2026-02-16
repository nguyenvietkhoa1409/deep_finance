import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import copy
import sys
from torch.utils.data import DataLoader

# --- Import Safe ---
try:
    from src.model import StockMovementModel
    from src.data_loader import data_prepare, StockDataset
    from configs.config import TrainConfig, GlobalConfig
    from main import merge_datasets, set_seed, evaluate
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    sys.exit(1)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = getattr(GlobalConfig, 'PROCESSED_PATH', 'data/processed') + "/unified_dataset_test.pkl"
KG_PATH = getattr(GlobalConfig, 'KG_PROCESSED_PATH', 'data/processed/kg_graphs.pkl')

TARGET_TICKERS = ["TSLA", "AMZN", "MSFT", "NFLX"]
ABLATION_EPOCHS = 200 # Đã chỉnh lên 200 theo yêu cầu

# --- [NEW] HELPER: Class Weights (Sao chép từ main.py để đồng bộ logic) ---
def compute_class_weights(labels_tensor: torch.Tensor) -> torch.Tensor:
    labels = labels_tensor.detach().cpu().numpy()
    class_counts = np.bincount(labels, minlength=3)
    num_classes = len(class_counts)

    beta = 0.9999 
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / np.sum(weights) * num_classes
    weights = np.sqrt(weights) # Square root smoothing
    weights = weights / np.sum(weights) * num_classes

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor

# --- TRAIN FUNCTION (STRICT MODE) ---
def train_ablation_model(model, train_loader, valid_data, epochs):
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=getattr(TrainConfig, "learning_rate", 1e-4), 
        weight_decay=getattr(TrainConfig, "weight_decay", 1e-4)
    )
    
    best_mcc = -100.0
    best_state = None
    
    print(f"    -> Start Training ({epochs} epochs)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            s_o = batch["s_o"].to(DEVICE)
            s_h = batch["s_h"].to(DEVICE)
            s_c = batch["s_c"].to(DEVICE)
            s_m = batch["s_m"].to(DEVICE)
            s_n = batch["s_n"].to(DEVICE)
            s_kg = batch.get("s_kg", None)
            label = batch["label"].to(DEVICE)
            
            loss = model(
                s_o=s_o, s_h=s_h, s_c=s_c, s_m=s_m, s_n=s_n, 
                s_kg=s_kg, label=label, mode="train"
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / max(num_batches, 1)

        # In tiến độ (Overwrite dòng cũ bằng end='\r')
        print(f"       Ep {epoch+1:03d}/{epochs} | Loss: {avg_loss:.4f}", end="\r")

        # Validate mỗi 10 epoch hoặc epoch cuối
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            val_acc, val_mcc = evaluate(model, valid_data)
            
            # [FIX] Sử dụng val_acc trong print để tránh cảnh báo "unused variable"
            print(f"       Ep {epoch+1:03d}/{epochs} | Loss: {avg_loss:.4f} | Val ACC: {val_acc:.4f} | Val MCC: {val_mcc:.4f}")
            if epoch >= 40: # Bắt đầu lưu từ epoch 40 để tránh lưu quá sớm khi model chưa ổn định
                if val_mcc > best_mcc:
                    best_mcc = val_mcc
                    best_state = copy.deepcopy(model.state_dict())
                
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

class AblationWrapper(nn.Module):
    """Wrapper điều hướng dòng dữ liệu, không can thiệp vào weights."""
    def __init__(self, core_model, use_news, use_macro, use_kg):
        super().__init__()
        self.core = core_model
        self.use_news = use_news
        self.use_macro = use_macro
        self.use_kg = use_kg
        self.device = DEVICE

    def forward(self, s_o, s_h, s_c, s_m, s_n, s_kg=None, label=None, mode="train"):
        if not self.use_news:
            s_n = torch.zeros_like(s_n).to(self.device)
        if not self.use_macro:
            s_m = torch.zeros_like(s_m).to(self.device)
        
        active_kg = s_kg if self.use_kg else None
            
        return self.core(
            s_o=s_o, s_h=s_h, s_c=s_c, s_m=s_m, s_n=s_n, 
            s_kg=active_kg, label=label, mode=mode
        )

def run_scientific_ablation():
    print("\n" + "="*70)
    print("🧪 SEQUENTIAL ABLATION STUDY (STRICT REPLICATION MODE)")
    print("   Target: Replica of main.py logic (Weighted Loss, No Scheduler)")
    print("="*70)
    set_seed(42)

    # 1. CHECK KG
    kg_available = False
    kg_data_path = None
    if os.path.exists(KG_PATH):
        try:
            from torch_geometric.data import Data
            kg_data_path = KG_PATH
            kg_available = True
            print("   ✅ KG Data found.")
        except:
            print("   ⚠️  KG Library missing.")
    else:
        print("   ⚠️  KG File missing.")

    # 2. LOAD DATA
    try:
        dp = data_prepare(DATA_PATH, kg_data_path=kg_data_path if kg_available else None)
        list_train, list_valid, list_test = [], [], []
        sample_macro_dim = 0
        
        for ticker in TARGET_TICKERS:
            try:
                tr, val, te = dp.prepare_data(ticker)
                if tr and "label" in tr and len(tr["label"]) > 0:
                    list_train.append(tr)
                    list_valid.append(val)
                    list_test.append(te)
                    if sample_macro_dim == 0: sample_macro_dim = tr["s_m"].shape[-1]
            except: pass

        if not list_train: return

        final_train = merge_datasets(list_train, shuffle=True)
        final_valid = merge_datasets(list_valid, shuffle=False)
        final_test  = merge_datasets(list_test,  shuffle=False)
        
        # [NEW] Compute Class Weights giống main.py
        print("\n   ⚖️  Computing Class Weights for Ablation Models...")
        class_weights = compute_class_weights(final_train["label"]).to(DEVICE)
        
    except Exception as e:
        print(f"❌ Error Data: {e}")
        return

    # Collate FN
    def collate_fn_kg(batch):
        from torch.utils.data.dataloader import default_collate
        s_kg_batch = [item.pop("s_kg") for item in batch] if "s_kg" in batch[0] else None
        collated = default_collate(batch)
        if s_kg_batch is not None: collated["s_kg"] = s_kg_batch
        return collated

    train_loader = DataLoader(
        StockDataset(final_train), 
        batch_size=TrainConfig.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_kg
    )

    # 3. SCENARIOS
    scenarios = [
        {"name": "1. Price Only",           "news": False, "macro": False, "kg": False},
        {"name": "2. +News (P+N)",          "news": True,  "macro": False, "kg": False},
        {"name": "3. +Macro (P+N+M)",       "news": True,  "macro": True,  "kg": False},
    ]
    if kg_available:
        scenarios.append({"name": "4. +KG (Full Model)", "news": True,  "macro": True,  "kg": True})

    results = []

    # 4. RUN
    for sc in scenarios:
        print(f"\n👉 Scenario: {sc['name']}")
        try:
            # Init Core Model (Strict Match with main.py)
            core_model = StockMovementModel(
                price_dim=1,
                macro_dim=sample_macro_dim,
                news_dim=TrainConfig.news_embed_dim,
                dim=TrainConfig.dim,
                input_dim=TrainConfig.window_size,
                output_dim=TrainConfig.output_dim,
                num_head=TrainConfig.num_head,
                device=DEVICE,
                dropout=0.1,  # main.py dùng 0.1, file config dùng 0.2. Ưu tiên main.py
                use_kg=True,  # Luôn bật module để nhận weights
                use_focal_loss=True, # main.py hardcode True
                class_weights=class_weights # [CRITICAL] Truyền weights đã tính
            ).to(DEVICE)
            
            wrapped_model = AblationWrapper(core_model, sc['news'], sc['macro'], sc['kg'])
            
            # Train
            trained_model = train_ablation_model(wrapped_model, train_loader, final_valid, ABLATION_EPOCHS)
            
            acc, mcc = evaluate(trained_model, final_test)
            results.append({"Scenario": sc['name'], "ACC": acc, "MCC": mcc})
            print(f"   🏁 Result: MCC={mcc:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # 5. REPORT
    print("\n" + "="*70)
    print("📊 CONTRIBUTION REPORT")
    print("="*70)
    df = pd.DataFrame(results)
    if not df.empty:
        print(df.to_string(index=False))
        
        mccs = df["MCC"].values
        print("\n🔍 Marginal Contributions:")
        if len(mccs) >= 2: print(f"   🔹 News:  {mccs[1] - mccs[0]:+.4f}")
        if len(mccs) >= 3: print(f"   🔹 Macro: {mccs[2] - mccs[1]:+.4f}")
        if len(mccs) >= 4: print(f"   🔹 KG:    {mccs[3] - mccs[2]:+.4f}")
    else:
        print("No results.")

if __name__ == "__main__":
    run_scientific_ablation()