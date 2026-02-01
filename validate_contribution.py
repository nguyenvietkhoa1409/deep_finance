import torch
import numpy as np
import os
import pandas as pd
import copy
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Import modules từ dự án
from src.model import StockMovementModel
from src.data_loader import data_prepare, StockDataset
from configs.config import TrainConfig
from main import merge_datasets, set_seed, compute_class_weights, evaluate

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = r"D:\DeepFinance\data\processed\unified_dataset_test.pkl"
TARGET_TICKERS = ["TSLA", "AMZN", "MSFT", "NFLX"]
ABLATION_EPOCHS = 30  # Số epoch train cho mỗi case (có thể thấp hơn main train để nhanh)

def train_ablation_model(model, train_loader, valid_data, epochs):
    """Hàm train rút gọn cho ablation study"""
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.learning_rate, weight_decay=1e-4)
    best_mcc = -1.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            # Lưu ý: batch vẫn chứa đủ data, nhưng model sẽ chỉ dùng phần nó cần
            # (Logic xử lý nằm ở cách gọi model forward bên dưới)
            loss = model(
                batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                batch["s_c"].to(DEVICE), batch["s_m"].to(DEVICE),
                batch["s_n"].to(DEVICE), batch["label"].to(DEVICE),
                mode="train"
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # Validate nhanh
        if (epoch + 1) % 5 == 0:
            val_acc, val_mcc = evaluate(model, valid_data)
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                best_state = copy.deepcopy(model.state_dict())
                
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def run_scientific_ablation():
    print("="*60)
    print("🧪 SCIENTIFIC ABLATION STUDY (RETRAINING METHOD)")
    print("="*60)
    set_seed(42)

    # 1. PREPARE DATA (Chung cho tất cả các case)
    print("\n📥 Loading Data...")
    dp = data_prepare(DATA_PATH)
    list_train, list_valid, list_test = [], [], []
    
    sample_dim_check = None
    for ticker in TARGET_TICKERS:
        try:
            tr, val, te = dp.prepare_data(ticker)
            if tr and len(tr.get("label", [])) > 0:
                list_train.append(tr)
                list_valid.append(val)
                list_test.append(te)
                if sample_dim_check is None: sample_dim_check = tr
        except: pass

    final_train = merge_datasets(list_train, shuffle=True)
    final_valid = merge_datasets(list_valid, shuffle=False)
    final_test  = merge_datasets(list_test,  shuffle=False)
    
    # DataLoader
    train_dataset = StockDataset(final_train)
    train_loader = DataLoader(train_dataset, batch_size=TrainConfig.batch_size, shuffle=True)
    
    # Lấy dimension thực tế
    real_macro_dim = sample_dim_check["s_m"].shape[-1]
    real_news_dim = TrainConfig.news_embed_dim
    
    # 2. DEFINE SCENARIOS
    # Mỗi scenario định nghĩa dimension input sẽ được truyền vào model
    # Nếu dim = 0 nghĩa là tắt module đó
    scenarios = [
        {
            "name": "FULL MODEL (Price + Macro + News)",
            "use_macro": True, "use_news": True,
            "macro_dim": real_macro_dim, "news_dim": real_news_dim
        },
        {
            "name": "NO NEWS (Price + Macro)",
            "use_macro": True, "use_news": False,
            "macro_dim": real_macro_dim, "news_dim": 0 # Tắt News
        },
        {
            "name": "NO MACRO (Price + News)",
            "use_macro": False, "use_news": True,
            "macro_dim": 0, "news_dim": real_news_dim # Tắt Macro
        },
        {
            "name": "PRICE ONLY",
            "use_macro": False, "use_news": False,
            "macro_dim": 0, "news_dim": 0 # Tắt cả hai
        }
    ]
    
    results = []

    # 3. RUN EXPERIMENTS
    for sc in scenarios:
        print(f"\n🚀 Running Scenario: {sc['name']}")
        print(f"   Configs: Macro_Dim={sc['macro_dim']}, News_Dim={sc['news_dim']}")
        
        # A. Khởi tạo Model Mới
        # Lưu ý: Cần sửa class StockMovementModel một chút để handle input dim = 0 (nếu chưa support)
        # Tuy nhiên, ta có thể trick bằng cách truyền dummy tensor 0 vào forward nếu dim > 0
        # Cách sạch nhất là Model tự handle.
        # Ở đây giả định Model có thể nhận dim=0 và layer linear(0, dim) sẽ không lỗi hoặc ta handle ở forward.
        
        # Để an toàn, ta vẫn khởi tạo model với dimension thật, 
        # nhưng TRONG VÒNG LẶP TRAIN/EVAL, ta sẽ zero-out input của module bị tắt.
        # Điều này mô phỏng việc model không nhìn thấy dữ liệu đó ngay từ đầu quá trình học.
        
        model = StockMovementModel(
            price_dim=1,
            macro_dim=real_macro_dim, # Khởi tạo full để tránh lỗi kiến trúc
            news_dim=real_news_dim,   # Khởi tạo full
            dim=TrainConfig.dim,
            input_dim=TrainConfig.window_size,
            output_dim=TrainConfig.output_dim,
            num_head=TrainConfig.num_head,
            dropout=0.1,
            class_weights=None, # No weights như main mới nhất
            use_focal_loss=True,
            device=DEVICE
        ).to(DEVICE)
        
        # B. Wrapper để Masking ngay từ lúc Train (Đây là điểm khác biệt với code cũ)
        # Ta tạo một hàm forward wrapper hoặc class wrapper
        class AblationWrapper(torch.nn.Module):
            def __init__(self, core_model, use_macro, use_news):
                super().__init__()
                self.core = core_model
                self.use_macro = use_macro
                self.use_news = use_news
                
            def forward(self, s_o, s_h, s_c, s_m, s_n, label=None, mode="train"):
                # Cưỡng bức Input về 0 NGAY KHI TRAIN nếu bị tắt
                if not self.use_macro:
                    s_m = torch.zeros_like(s_m)
                if not self.use_news:
                    s_n = torch.zeros_like(s_n)
                
                return self.core(s_o, s_h, s_c, s_m, s_n, label, mode)

        wrapped_model = AblationWrapper(model, sc['use_macro'], sc['use_news'])
        
        # C. Train from Scratch
        print("   Training...")
        trained_model = train_ablation_model(wrapped_model, train_loader, final_valid, epochs=ABLATION_EPOCHS)
        
        # D. Evaluate
        print("   Evaluating...")
        test_acc, test_mcc = evaluate(trained_model, final_test)
        
        results.append({
            "Scenario": sc['name'],
            "ACC": test_acc,
            "MCC": test_mcc
        })
        print(f"   Result -> ACC: {test_acc:.4f} | MCC: {test_mcc:.4f}")

    # 4. REPORT
    print("\n" + "="*60)
    print("📊 SCIENTIFIC ABLATION REPORT")
    print("="*60)
    df_res = pd.DataFrame(results)
    
    # Tính Delta so với Baseline (Full Model)
    baseline_mcc = df_res.iloc[0]["MCC"]
    df_res["Delta MCC"] = df_res["MCC"] - baseline_mcc
    
    print(df_res.to_string(index=False, formatters={
        "ACC": "{:.4f}".format, 
        "MCC": "{:.4f}".format,
        "Delta MCC": "{:+.4f}".format
    }))
    
    print("\n📝 CONCLUSION:")
    best_row = df_res.loc[df_res['MCC'].idxmax()]
    print(f"   🏆 Best Configuration: {best_row['Scenario']}")
    if best_row['Scenario'] != scenarios[0]['name']:
        print(f"   ⚠️  Warning: Full Model is NOT the best. Consider removing noisy modules.")
    else:
        print(f"   ✅ Full Model is the best. Synergy confirmed.")

if __name__ == "__main__":
    run_scientific_ablation()