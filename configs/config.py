# configs/config.py
import os


class GlobalConfig:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    # Data level paths
    RAW_PATH = os.path.join(DATA_DIR, 'raw')
    INTERIM_PATH = os.path.join(DATA_DIR, 'interim')
    PROCESSED_PATH = os.path.join(DATA_DIR, 'processed')

    RAW_PRICE_PATH = os.path.join(RAW_PATH, 'market_price')
    RAW_MACRO_PATH = os.path.join(RAW_PATH, 'macro')
    RAW_NEWS_PATH = os.path.join(RAW_PATH, 'news')

    # News embeddings are produced by a separate preprocessing pipeline
    # (LLM few-shot S-R-O triple extraction -> schema-constrained triples ->
    # quality filtering -> FinBERT embedding -> confidence-weighted daily
    # aggregation per ticker). This repo only consumes the resulting
    # per-date/per-ticker embedding vectors; see main_test.py / data_pipeline/builder.py.
    NEWS_EMBEDDING_OUTPUT_PATH = os.path.join(INTERIM_PATH, 'news_headline_embeddings')

    # --- API Keys (Load from env) ---
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

    TRAIN_END_DATE = '2025-06-26'

    # Ticker universe (report Section 4.1 "Dataset construction"): 8 S&P 500
    # constituents across 4 sectors.
    #   Consumer Discretionary : TSLA, AMZN
    #   Technology              : AAPL, MSFT, GOOGL, META
    #   Industrials              : BA
    #   Consumer Staples         : WMT
    TICKERS = ["TSLA", "AMZN", "AAPL", "MSFT", "GOOGL", "META", "BA", "WMT"]

    TICKER_MAPPING = {
        'TSLA': 'TSLA', 'Tesla': 'TSLA',
        'AMZN': 'AMZN', 'Amazon': 'AMZN',
        'AAPL': 'AAPL', 'Apple': 'AAPL',
        'MSFT': 'MSFT', 'Microsoft': 'MSFT',
        'GOOGL': 'GOOGL', 'GOOG': 'GOOGL', 'Alphabet': 'GOOGL', 'Google': 'GOOGL',
        'META': 'META', 'Facebook': 'META', 'Meta': 'META',
        'BA': 'BA', 'Boeing': 'BA',
        'WMT': 'WMT', 'Walmart': 'WMT',
    }

    # Macro symbols (Yahoo Finance tickers): VIX, S&P 500, USD Index, WTI crude.
    # 10Y-2Y treasury yield spread is derived separately in macro_processor.
    MACRO_SYMBOLS = {
        'vix': '^VIX',
        'sp500': '^GSPC',
        'dxy': 'DX-Y.NYB',
        'wti': 'CL=F'
    }


class TrainConfig:
    # reproducibility
    seed = 42
    use_cuda = True

    # data
    train_ratio = 0.70   # 70% train
    valid_ratio = 0.15   # 15% valid (remainder 15% test)
    batch_size = 32

    # Window Size (T): number of past days used to forecast.
    window_size = 14
    news_embed_dim = 768  # FinBERT hidden size (report Fig. 2: daily news embedding)

    # --- MODEL HYPERPARAMETERS ---
    dim = 128          # shared latent dimension d
    output_dim = 3      # Down, Flat, Up
    num_head = 4         # cross-attention heads
    news_temporal_decay = 0.1  # news encoder temporal-decay coefficient (Eq. 5)

    # training
    epoch_num = 100
    learning_rate = 1e-4
    weight_decay = 5e-4
    drop_out = 0.3

    # Loss / optimizer
    use_focal_loss = False
    label_smoothing = 0.1
