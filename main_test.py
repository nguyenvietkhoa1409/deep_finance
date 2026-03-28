import os
import pandas as pd
from configs.config import GlobalConfig
from data_pipeline.fetchers.yahoo_fetcher import YahooFetcher
from data_pipeline.processors.price_processor import PriceProcessor
from data_pipeline.processors.macro_processor import MacroProcessor
from data_pipeline.processors.news_processor import NewsProcessor, NewsEmbedder
from data_pipeline.builder import DatasetBuilder

def run_test_pipeline_skipping_news_fetch():
    print("🚀 STARTING TEST PIPELINE (Skipping News Fetching)...")

    # --- SETUP PATHS ---
    # Đường dẫn đến file kết quả cũ bạn đã có
    # LƯU Ý: Đảm bảo file này nằm đúng vị trí hoặc bạn sửa đường dẫn tại đây
    EXISTING_NEWS_PATH = os.path.join(GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet")
    
    if not os.path.exists(EXISTING_NEWS_PATH):
        print(f"❌ ERROR: Không tìm thấy file tại {EXISTING_NEWS_PATH}")
        print("   Vui lòng copy file 'concatenated_news_filtered.parquet' vào thư mục 'data/interim/' hoặc cập nhật đường dẫn.")
        return

    # ==============================================================================
    # 1. Fetching Phase (Chỉ lấy Price & Macro, BỎ QUA News Fetcher)
    # ==============================================================================
    print("\n--- Phase A: Fetching (Price & Macro only) ---")
    yahoo = YahooFetcher()
    
    # Tạo thư mục nếu chưa có
    os.makedirs(GlobalConfig.RAW_PRICE_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.RAW_MACRO_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.PROCESSED_PATH, exist_ok=True)

    # 1.1 Lấy dữ liệu giá (Làm xương sống cho trục thời gian)
    print(f"   Downloading Price Data ({GlobalConfig.START_DATE} to {GlobalConfig.END_DATE})...")
    raw_price_list = yahoo.download_data(GlobalConfig.START_DATE, GlobalConfig.END_DATE, GlobalConfig.TICKERS)
    
    # 1.2 Lấy dữ liệu Vĩ mô
    print("   Downloading Macro Indicators...")
    raw_macro = yahoo.fetch_macro_indicators(GlobalConfig.START_DATE, GlobalConfig.END_DATE, GlobalConfig.MACRO_SYMBOLS)

    # ==============================================================================
    # 2. Processing Phase (Load News có sẵn -> Align -> Embed)
    # ==============================================================================
    print("\n--- Phase B: Processing ---")
    price_proc = PriceProcessor()
    macro_proc = MacroProcessor()
    news_proc = NewsProcessor()

    # 2.1 Xử lý Price & Macro
    print("   Processing Price & Macro...")
    price_dict = price_proc.combine_to_nested_dict(raw_price_list, GlobalConfig.TICKERS)
    processed_price_macro = macro_proc.process_and_enrich(price_dict, raw_macro)
    
    # Lấy danh sách ngày giao dịch chuẩn (Trading Days Backbone)
    trading_dates = list(processed_price_macro.keys())
    print(f"   Detected {len(trading_dates)} trading days.")

    # 2.2 Load Existing News
    print(f"   📥 Loading existing news from: {EXISTING_NEWS_PATH}")
    try:
        processed_news = pd.read_parquet(EXISTING_NEWS_PATH)
        print(f"   Loaded {len(processed_news)} news records.")
        
        # [ANNOTATION 1] Kiểm tra Schema
        required_cols = ['date', 'equity', 'title'] # Các cột bắt buộc cho bước sau
        missing_cols = [c for c in required_cols if c not in processed_news.columns]
        if missing_cols:
            print(f"   ⚠️ WARNING: File parquet thiếu các cột: {missing_cols}")
            print("   Logic cũ có thể dùng tên khác (ví dụ: 'headline' thay vì 'title'). Đang thử tự động sửa...")
            if 'headline' in processed_news.columns and 'title' not in processed_news.columns:
                processed_news = processed_news.rename(columns={'headline': 'title'})
                print("   ✅ Renamed 'headline' -> 'title'.")
            
            # Kiểm tra lại cột date phải là datetime
            if not pd.api.types.is_datetime64_any_dtype(processed_news['date']):
                 processed_news['date'] = pd.to_datetime(processed_news['date']).dt.date
    
    except Exception as e:
        print(f"❌ ERROR reading parquet file: {e}")
        return

    # 2.3 Align News to Trading Days
    # Bước này vẫn CẦN THIẾT vì Price Data mới tải về có thể có ngày nghỉ lễ khác hoặc range khác
    print("   Aligning news to current Trading Days...")
    aligned_news = news_proc.align_to_trading_days(processed_news, trading_dates)
    print(f"   News after alignment: {len(aligned_news)} records.")

    # ==============================================================================
    # 3. Embedding Phase (Chạy Embedding trên dữ liệu đã load)
    # ==============================================================================
    print("\n--- Phase B.1: Embedding News ---")
    
    # [ANNOTATION 2] Kiểm tra file embedding cũ
    embedder = NewsEmbedder()
    embedding_output_file = os.path.join(GlobalConfig.NEWS_EMBEDDING_OUTPUT_PATH, "embedded_news.json")
    
    if os.path.exists(embedding_output_file):
        print(f"   ⚠️ File embedding đã tồn tại: {embedding_output_file}")
        user_input = input("   Bạn có muốn chạy lại Embedding (tốn tiền/thời gian) không? (y/n): ")
        if user_input.lower() == 'y':
            embedding_json_path = embedder.process_and_save(aligned_news)
        else:
            print("   Skipping Embedding calculation. Using existing file.")
            embedding_json_path = embedding_output_file
    else:
        # Nếu chưa có file thì chạy mới
        embedding_json_path = embedder.process_and_save(aligned_news)

    # ==============================================================================
    # 4. Building Phase (Tạo file Union cuối cùng)
    # ==============================================================================
    print("\n--- Phase C: Building Union File ---")
    builder = DatasetBuilder()
    
    # Giả định file Filings đã có (hoặc bỏ qua nếu chưa cần test filings)
    filing_path = os.path.join(GlobalConfig.RAW_FILINGS_PATH, "final_summary_filing_data.parquet")
    if not os.path.exists(filing_path):
        print(f"   ⚠️ Warning: Filing file not found at {filing_path}. Creating dataset without filings.")
        # Tạo dummy empty dataframe để code không lỗi
        pd.DataFrame(columns=['filedAt', 'ticker', 'formType', 'content_summary']).to_parquet("dummy_filings.parquet")
        filing_path = "dummy_filings.parquet"

    dataset = builder.create_synchronized_data(
        processed_price_macro, 
        aligned_news, 
        filing_path,
        embedding_path=embedding_json_path
    )
    
    builder.save(dataset, filename='unified_dataset_test.pkl')
    
    # Xóa file dummy nếu có
    if os.path.exists("dummy_filings.parquet"): os.remove("dummy_filings.parquet")
    
    print("\n✅ TEST PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test_pipeline_skipping_news_fetch()
