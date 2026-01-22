import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
from torch.utils.data import DataLoader, Dataset

# Import modules từ dự án
from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig, GlobalConfig
from main import merge_datasets, StockDataset, set_seed, compute_class_weights

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("output", "best_model.pt") # Đường dẫn model tốt nhất
DATA_PATH = r"D:\DeepFinance\data\processed\unified_dataset_test.pkl"
TARGET_TICKERS = ["TSLA", "AMZN", "MSFT", "NFLX"]

def run_ablation_test():
    print("="*60)
    print("🧪 MODULE CONTRIBUTION ANALYSIS (ABLATION STUDY)")
    print("="*60)
    
    # 1. LOAD DATA (TEST SET)
    print("\n📥 Loading TEST Data...")
    dp = data_prepare(DATA_PATH)
    list_test = []
    
    # Load 1 sample để lấy dimension config
    sample_dim_check = None
    
    for ticker in TARGET_TICKERS:
        try:
            _, _, te = dp.prepare_data(ticker)
            if te and len(te.get("label", [])) > 0:
                list_test.append(te)
                if sample_dim_check is None: sample_dim_check = te
        except: pass
        
    if not list_test:
        print("❌ Error: No test data found.")
        return

    final_test = merge_datasets(list_test, shuffle=False)
    print(f"✅ Total Test Samples: {len(final_test['label'])}")
    
    # Lấy Dimension thực tế từ dữ liệu
    s_m_dim = sample_dim_check["s_m"].shape[-1]
    
    # 2. LOAD TRAINED MODEL
    print(f"\n🤖 Loading Model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found! Please train a model first.")
        return

    # Khởi tạo model y hệt cấu hình training (Weights không quan trọng vì ta load state_dict)
    # Lưu ý: Cần dummy weights để init Focal Loss, ta tạo tạm
    dummy_weights = torch.tensor([1.0, 1.0, 1.0]) 
    
    model = StockMovementModel(
        price_dim=1, macro_dim=s_m_dim, news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim, num_head=TrainConfig.num_head,
        dropout=0.0, # Tắt dropout khi test
        class_weights=dummy_weights, use_focal_loss=True, device=DEVICE
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model Loaded Successfully.")

    # 3. DEFINE ABLATION EXPERIMENTS
    # Danh sách các kịch bản test
    experiments = [
        {"name": "BASELINE (Full Info)", "mask_macro": False, "mask_news": False},
        {"name": "NO MACRO (Price+News)", "mask_macro": True,  "mask_news": False},
        {"name": "NO NEWS  (Price+Macro)", "mask_macro": False, "mask_news": True},
        {"name": "PRICE ONLY",            "mask_macro": True,  "mask_news": True},
    ]
    
    results = []

    # 4. RUN EXPERIMENTS
    print("\n🚀 Running Inference Tests...")
    
    for exp in experiments:
        exp_name = exp["name"]
        print(f"   Running: {exp_name}...", end="")
        
        acc, mcc, report = evaluate_with_masking(
            model, final_test, 
            mask_macro=exp["mask_macro"], 
            mask_news=exp["mask_news"]
        )
        
        results.append({
            "Scenario": exp_name,
            "ACC": acc,
            "MCC": mcc,
            "Diff MCC": 0.0 # Sẽ tính sau
        })
        print(f" Done. (MCC: {mcc:.4f})")

    # 5. ANALYZE RESULTS
    print("\n" + "="*60)
    print("📊 CONTRIBUTION REPORT")
    print("="*60)
    
    # Lấy baseline MCC
    baseline_mcc = results[0]["MCC"]
    
    # Tạo DataFrame để hiển thị đẹp
    df_res = pd.DataFrame(results)
    df_res["Diff MCC"] = df_res["MCC"] - baseline_mcc
    df_res["Impact"] = df_res["Diff MCC"].apply(lambda x: "🔻 HURT" if x < -0.01 else ("✅ HELP" if x > 0.01 else "⚪ NEUTRAL"))
    
    print(df_res.to_string(index=False, formatters={
        "ACC": "{:.4f}".format,
        "MCC": "{:.4f}".format,
        "Diff MCC": "{:+.4f}".format
    }))
    
    print("\n📝 INTERPRETATION:")
    print("   - Nếu 'Diff MCC' ÂM LỚN (vd: -0.05): Module đó QUAN TRỌNG (bỏ đi làm model ngu đi).")
    print("   - Nếu 'Diff MCC' GẦN 0 (vd: -0.00): Module đó VÔ DỤNG (model không thèm dùng nó).")
    print("   - Nếu 'Diff MCC' DƯƠNG (vd: +0.02): Module đó GÂY NHIỄU (bỏ đi model lại chạy tốt hơn).")

def evaluate_with_masking(model, data_dict, mask_macro=False, mask_news=False):
    """
    Hàm Evaluate có khả năng 'tắt' (mask) các nguồn dữ liệu bằng cách đưa về 0.
    """
    s_o = data_dict["s_o"].to(DEVICE)
    s_h = data_dict["s_h"].to(DEVICE)
    s_c = data_dict["s_c"].to(device=DEVICE)
    
    # Xử lý Masking cho Macro
    if mask_macro:
        # Tạo tensor 0 cùng kích thước
        s_m = torch.zeros_like(data_dict["s_m"]).to(DEVICE)
    else:
        s_m = data_dict["s_m"].to(DEVICE)
        
    # Xử lý Masking cho News
    if mask_news:
        s_n = torch.zeros_like(data_dict["s_n"]).to(DEVICE)
    else:
        s_n = data_dict["s_n"].to(DEVICE)
        
    label = data_dict["label"].to(DEVICE)
    
    with torch.no_grad():
        # Forward pass
        acc, mcc = model(s_o, s_h, s_c, s_m, s_n, label, mode="test")
        
        # Lấy thêm classification report (optional)
        # logits = model(s_o, s_h, s_c, s_m, s_n, mode="inference")
        # preds = torch.argmax(logits, dim=1)
        # report = classification_report(label.cpu(), preds.cpu(), output_dict=True, zero_division=0)
        
    return acc, mcc, None

if __name__ == "__main__":
    # Đảm bảo seed để tái lập kết quả
    set_seed(42) 
    run_ablation_test()