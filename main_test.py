import os
import json
import pandas as pd
from configs.config import GlobalConfig
from data_pipeline.fetchers.yahoo_fetcher import YahooFetcher
from data_pipeline.processors.price_processor import PriceProcessor
from data_pipeline.processors.macro_processor import MacroProcessor
from data_pipeline.builder import DatasetBuilder

def run_pipeline_with_llm_embeddings():
    print("🚀 STARTING PIPELINE WITH MULTI-TICKER LLM EXTRACTED SIGNALS...")

    # ==============================================================================
    # 1. SETUP PATHS & SMART LOAD PRECOMPUTED EMBEDDINGS
    # ==============================================================================
    EMBEDDING_PATH = r"D:\news_embeddings.json"
    
    if not os.path.exists(EMBEDDING_PATH):
        print(f"❌ ERROR: Không tìm thấy file embeddings tại {EMBEDDING_PATH}")
        return

    print(f"   📥 Đang load file embeddings từ: {EMBEDDING_PATH}")
    with open(EMBEDDING_PATH, 'r') as f:
        raw_embeddings = json.load(f)

    formatted_embeddings = {}
    unique_tickers = set()
    all_dates = set()

    # Thuật toán tự động nhận diện và chuẩn hóa cấu trúc JSON đa mã cổ phiếu
    for key, value in raw_embeddings.items():
        # Trường hợp 1: Key ngoài cùng là Ngày (vd: "2023-01-01")
        if key.startswith("20") or key.startswith("19"):
            clean_date = key[:10]
            all_dates.add(clean_date)
            if clean_date not in formatted_embeddings:
                formatted_embeddings[clean_date] = []
            
            # Subcase 1a: Value là list of dicts -> [{"equity": "TSLA", "embedding": [...]}, ...]
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and "equity" in value[0]:
                formatted_embeddings[clean_date].extend(value)
                for item in value:
                    unique_tickers.add(item["equity"])
            
            # Subcase 1b: Value là dict -> {"TSLA": [...], "AMZN": [...]}
            elif isinstance(value, dict):
                for ticker, emb in value.items():
                    formatted_embeddings[clean_date].append({"equity": ticker, "embedding": emb})
                    unique_tickers.add(ticker)
                    
            # Subcase 1c: Value là list of floats (Format cũ 1 mã TSLA)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], float):
                formatted_embeddings[clean_date].append({"equity": "TSLA", "embedding": value})
                unique_tickers.add("TSLA")

        # Trường hợp 2: Key ngoài cùng là Ticker (vd: "TSLA")
        else:
            ticker = key
            unique_tickers.add(ticker)
            if isinstance(value, dict):
                for date_str, emb in value.items():
                    clean_date = date_str[:10]
                    all_dates.add(clean_date)
                    if clean_date not in formatted_embeddings:
                        formatted_embeddings[clean_date] = []
                    formatted_embeddings[clean_date].append({"equity": ticker, "embedding": emb})

    # Lấy range ngày
    sorted_dates = sorted(list(all_dates))
    start_date = sorted_dates[0]
    end_date = sorted_dates[-1]
    target_tickers = list(unique_tickers)
    
    print(f"   📊 Đã nhận diện {len(target_tickers)} Tickers: {target_tickers}")
    print(f"   📊 Dataset có {len(sorted_dates)} ngày giao dịch (Từ {start_date} đến {end_date})")

    # Override config động theo file data thực tế
    GlobalConfig.TICKERS = target_tickers
    GlobalConfig.START_DATE = start_date
    GlobalConfig.END_DATE = end_date

    # Lưu file embedding tạm thời đã chuẩn hóa để builder.py đọc
    os.makedirs(GlobalConfig.INTERIM_PATH, exist_ok=True)
    TEMP_EMBED_PATH = os.path.join(GlobalConfig.INTERIM_PATH, "temp_multi_embeddings.json")
    with open(TEMP_EMBED_PATH, 'w') as f:
        json.dump(formatted_embeddings, f)

    # ==============================================================================
    # 2. Fetching Phase (Price & Macro only)
    # ==============================================================================
    print("\n--- Phase A: Fetching (Price & Macro only) ---")
    yahoo = YahooFetcher()
    
    os.makedirs(GlobalConfig.RAW_PRICE_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.RAW_MACRO_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.PROCESSED_PATH, exist_ok=True)

    print(f"   Downloading Price Data for {target_tickers} ({start_date} to {end_date})...")
    raw_price_list = yahoo.download_data(start_date, end_date, GlobalConfig.TICKERS)
    
    print("   Downloading Macro Indicators...")
    raw_macro = yahoo.fetch_macro_indicators(start_date, end_date, GlobalConfig.MACRO_SYMBOLS)

    # ==============================================================================
    # 3. Processing Phase
    # ==============================================================================
    print("\n--- Phase B: Processing ---")
    price_proc = PriceProcessor()
    macro_proc = MacroProcessor()

    print("   Processing Price & Macro...")
    price_dict = price_proc.combine_to_nested_dict(raw_price_list, GlobalConfig.TICKERS)
    processed_price_macro = macro_proc.process_and_enrich(price_dict, raw_macro)
    
    trading_dates = list(processed_price_macro.keys())
    print(f"   Detected {len(trading_dates)} trading days from Yahoo Finance.")

    # Dummy news_df 
    dummy_news_df = pd.DataFrame(columns=['date', 'equity', 'title', 'content', 'summary', 'source', 'url'])
    dummy_news_df['date'] = pd.to_datetime(dummy_news_df['date'])

    # ==============================================================================
    # 4. Building Phase (Tạo file Union cuối cùng)
    # ==============================================================================
    print("\n--- Phase C: Building Union File ---")
    builder = DatasetBuilder()
    
    filing_path = os.path.join(GlobalConfig.RAW_FILINGS_PATH, "final_summary_filing_data.parquet")
    if not os.path.exists(filing_path):
        print(f"   ⚠️ Warning: Filing file not found. Khởi tạo dataset bỏ qua filings.")
        pd.DataFrame(columns=['filedAt', 'ticker', 'formType', 'content_summary']).to_parquet("dummy_filings.parquet")
        filing_path = "dummy_filings.parquet"

    dataset = builder.create_synchronized_data(
        processed_price_macro, 
        dummy_news_df, 
        filing_path,
        embedding_path=TEMP_EMBED_PATH
    )
    
    output_filename = 'unified_dataset_test.pkl' # Đã bỏ đuôi TSLA
    builder.save(dataset, filename=output_filename)
    
    # Clean up files tạm
    if os.path.exists("dummy_filings.parquet"): os.remove("dummy_filings.parquet")
    if os.path.exists(TEMP_EMBED_PATH): os.remove(TEMP_EMBED_PATH)
    
    print(f"\n✅ TEST PIPELINE COMPLETED! Dữ liệu của 4 Tickers đã sẵn sàng tại processed/{output_filename}")

if __name__ == "__main__":
    run_pipeline_with_llm_embeddings()