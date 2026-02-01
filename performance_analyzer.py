import torch
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, matthews_corrcoef
from collections import Counter

# Import project modules
from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("output", "best_model.pt")
DATA_PATH = r"D:\DeepFinance\data\processed\unified_dataset_test.pkl" 

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔎 {title}")
    print(f"{'='*60}")

def load_data_per_ticker(tickers):
    """
    Load dữ liệu Test riêng biệt cho từng mã để phân tích behavior.
    """
    dp = data_prepare(DATA_PATH)
    ticker_datasets = {}
    
    print(f"📥 Loading TEST data for: {tickers}")
    for t in tickers:
        try:
            # prepare_data trả về: train, valid, test (index 2)
            _, _, test_data = dp.prepare_data(
                stock_name=t,
                window_size=TrainConfig.window_size,
            )
            
            if test_data and len(test_data.get("label", [])) > 0:
                ticker_datasets[t] = test_data
                print(f"   ✅ {t}: {len(test_data['label'])} samples")
            else:
                print(f"   ⚠️ {t}: No data or empty test set")
        except Exception as e:
            print(f"   ❌ {t}: Error {e}")
            
    return ticker_datasets

def run_prediction(model, data_dict):
    """
    Chạy forward pass để lấy Logits và Predictions.
    """
    model.eval()
    with torch.no_grad():
        s_o = data_dict["s_o"].to(DEVICE)
        s_h = data_dict["s_h"].to(DEVICE)
        s_c = data_dict["s_c"].to(DEVICE)
        s_m = data_dict["s_m"].to(DEVICE)
        s_n = data_dict["s_n"].to(DEVICE)
        
        # 1. Encoder
        v_m, v_i, v_n = model.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)
        
        # 2. Fusion (Lưu ý: Fusion mới có thể không dùng residual cho gating)
        fused_news = model.fusion_news(primary=v_i, aux=v_n)
        fused_macro = model.fusion_macro(primary=v_i, aux=v_m)
        v_fused_total = (fused_news + fused_macro) / 2.0
        
        # 3. Predictor (Attention Pooling)
        logits = model.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
        
        # Probability & Prediction
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
    return preds.cpu().numpy(), data_dict["label"].numpy(), probs.cpu().numpy()

def analyze_performance():
    # 1. Load Model
    print_header("1. LOADING MODEL")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Cannot find model at {MODEL_PATH}")
        return

    # Lấy dimension macro thực tế để init model cho đúng
    dp = data_prepare(DATA_PATH)
    # Lấy thử 1 mã để check dimension
    dummy_train, _, _ = dp.prepare_data("TSLA") 
    if dummy_train:
        macro_dim = dummy_train["s_m"].shape[-1]
    else:
        macro_dim = 6 # Fallback
    
    print(f"🔧 Model Config: Dim={TrainConfig.dim}, Heads={TrainConfig.num_head}, Macro={macro_dim}")

    # Khởi tạo model architecture
    model = StockMovementModel(
        price_dim=1,
        macro_dim=macro_dim,
        news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim,                 
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,       
        device=DEVICE,
        dropout=0.0,                         # Eval mode không cần dropout
        class_weights=None,                  # Eval không cần tính loss
        use_focal_loss=False                 # Eval không cần Focal
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 2. Load Data
    print_header("2. LOADING DATA")
    target_tickers = ["TSLA", "AMZN", "MSFT", "NFLX"] 
    datasets = load_data_per_ticker(target_tickers)
    
    if not datasets:
        print("❌ No datasets loaded.")
        return

    # 3. Deep Dive Analysis
    print_header("3. DEEP DIVE ANALYSIS")
    
    all_preds = []
    all_labels = []
    
    print(f"{'TICKER':<10} | {'SAMPLES':<8} | {'ACTUAL DIST (0/1/2)':<25} | {'PRED DIST (0/1/2)':<25} | {'ACC':<8} | {'MCC':<8}")
    print("-" * 110)

    for ticker, data in datasets.items():
        preds, labels, probs = run_prediction(model, data)
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        acc = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        
        # Count distributions
        act_counts = Counter(labels)
        pred_counts = Counter(preds)
        
        act_dist = f"{act_counts.get(0,0)}/{act_counts.get(1,0)}/{act_counts.get(2,0)}"
        pred_dist = f"{pred_counts.get(0,0)}/{pred_counts.get(1,0)}/{pred_counts.get(2,0)}"
        
        print(f"{ticker:<10} | {len(labels):<8} | {act_dist:<25} | {pred_dist:<25} | {acc:.4f}   | {mcc:.4f}")

    # 4. Global Analysis
    print_header("4. GLOBAL SUMMARY (ALL TICKERS COMBINED)")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    unique_act, counts_act = np.unique(all_labels, return_counts=True)
    unique_pred, counts_pred = np.unique(all_preds, return_counts=True)
    
    print("📉 ACTUAL Labels Distribution (Ground Truth):")
    act_dist_dict = dict(zip(unique_act, counts_act))
    print(f"   {act_dist_dict}")
    
    # Tính toán tỷ lệ phần trăm
    total_act = sum(counts_act)
    if total_act > 0:
        p0 = act_dist_dict.get(0,0)/total_act*100
        p1 = act_dist_dict.get(1,0)/total_act*100
        p2 = act_dist_dict.get(2,0)/total_act*100
        print(f"   (Down: {p0:.1f}%, Flat: {p1:.1f}%, Up: {p2:.1f}%)")
    
    print("\n🔮 PREDICTED Labels Distribution:")
    print(f"   {dict(zip(unique_pred, counts_pred))}")
    
    # Check Mode Collapse
    if len(unique_pred) == 1:
        print("\n⚠️  CRITICAL WARNING: MODE COLLAPSE DETECTED!")
        print(f"   Mô hình chỉ dự đoán duy nhất lớp {unique_pred[0]} cho toàn bộ dữ liệu.")
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print(f"      Pred 0  Pred 1  Pred 2")
    print(f"Act 0   {cm[0][0]:<7} {cm[0][1]:<7} {cm[0][2]:<7}")
    print(f"Act 1   {cm[1][0]:<7} {cm[1][1]:<7} {cm[1][2]:<7}")
    print(f"Act 2   {cm[2][0]:<7} {cm[2][1]:<7} {cm[2][2]:<7}")
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['DOWN', 'FLAT', 'UP'], zero_division=0))

if __name__ == "__main__":
    analyze_performance()