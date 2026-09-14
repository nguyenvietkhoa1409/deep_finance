import os
import json
import pandas as pd
from configs.config import GlobalConfig
from data_pipeline.fetchers.yahoo_fetcher import YahooFetcher
from data_pipeline.processors.price_processor import PriceProcessor
from data_pipeline.processors.macro_processor import MacroProcessor
from data_pipeline.builder import DatasetBuilder


def run_pipeline_with_precomputed_news_embeddings():
    """
    Builds the unified dataset consumed by src/data_loader.py.

    News embeddings are NOT computed here: they come from a separate
    preprocessing project (LLM few-shot S-R-O triple extraction ->
    schema-constrained triples -> quality filtering -> text embedding ->
    confidence-weighted daily aggregation per ticker, report Section 4.2).
    This script only ingests that project's output JSON and synchronizes it
    with price/macro data fetched here.
    """
    print("🚀 STARTING PIPELINE (price + macro fetch, precomputed news embeddings)...")

    # ==============================================================================
    # 1. LOAD PRECOMPUTED NEWS EMBEDDINGS
    # ==============================================================================
    # Expected JSON shape: { "YYYY-MM-DD": [{"equity": "TSLA", "embedding": [...]}, ...], ... }
    EMBEDDING_PATH = os.getenv("NEWS_EMBEDDINGS_PATH", "")

    if not EMBEDDING_PATH or not os.path.exists(EMBEDDING_PATH):
        print(f"❌ ERROR: News embeddings file not found. Set NEWS_EMBEDDINGS_PATH to the "
              f"output of the news preprocessing pipeline. (got: {EMBEDDING_PATH!r})")
        return

    print(f"   📥 Loading embeddings from: {EMBEDDING_PATH}")
    with open(EMBEDDING_PATH, 'r') as f:
        raw_embeddings = json.load(f)

    formatted_embeddings = {}
    unique_tickers = set()
    all_dates = set()

    # Auto-detect and normalize the embedding JSON structure.
    for key, value in raw_embeddings.items():
        if key.startswith("20") or key.startswith("19"):
            # Case 1: outer key is a date, e.g. "2023-01-01"
            clean_date = key[:10]
            all_dates.add(clean_date)
            formatted_embeddings.setdefault(clean_date, [])

            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and "equity" in value[0]:
                formatted_embeddings[clean_date].extend(value)
                for item in value:
                    unique_tickers.add(item["equity"])
            elif isinstance(value, dict):
                for ticker, emb in value.items():
                    formatted_embeddings[clean_date].append({"equity": ticker, "embedding": emb})
                    unique_tickers.add(ticker)
        else:
            # Case 2: outer key is a ticker, e.g. "TSLA"
            ticker = key
            unique_tickers.add(ticker)
            if isinstance(value, dict):
                for date_str, emb in value.items():
                    clean_date = date_str[:10]
                    all_dates.add(clean_date)
                    formatted_embeddings.setdefault(clean_date, [])
                    formatted_embeddings[clean_date].append({"equity": ticker, "embedding": emb})

    sorted_dates = sorted(all_dates)
    start_date = sorted_dates[0]
    end_date = sorted_dates[-1]
    target_tickers = list(unique_tickers)

    print(f"   📊 Detected {len(target_tickers)} tickers: {target_tickers}")
    print(f"   📊 {len(sorted_dates)} trading days (from {start_date} to {end_date})")

    GlobalConfig.TICKERS = target_tickers

    os.makedirs(GlobalConfig.INTERIM_PATH, exist_ok=True)
    TEMP_EMBED_PATH = os.path.join(GlobalConfig.INTERIM_PATH, "temp_multi_embeddings.json")
    with open(TEMP_EMBED_PATH, 'w') as f:
        json.dump(formatted_embeddings, f)

    # ==============================================================================
    # 2. FETCH (Price & Macro)
    # ==============================================================================
    print("\n--- Phase A: Fetching (Price & Macro) ---")
    yahoo = YahooFetcher()

    os.makedirs(GlobalConfig.RAW_PRICE_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.RAW_MACRO_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.PROCESSED_PATH, exist_ok=True)

    print(f"   Downloading price data for {target_tickers} ({start_date} to {end_date})...")
    raw_price_list = yahoo.download_data(start_date, end_date, GlobalConfig.TICKERS)

    print("   Downloading macro indicators...")
    raw_macro = yahoo.fetch_macro_indicators(start_date, end_date, GlobalConfig.MACRO_SYMBOLS)

    # ==============================================================================
    # 3. PROCESS
    # ==============================================================================
    print("\n--- Phase B: Processing ---")
    price_proc = PriceProcessor()
    macro_proc = MacroProcessor()

    price_dict = price_proc.combine_to_nested_dict(raw_price_list, GlobalConfig.TICKERS)
    processed_price_macro = macro_proc.process_and_enrich(price_dict, raw_macro)

    trading_dates = list(processed_price_macro.keys())
    print(f"   Detected {len(trading_dates)} trading days from Yahoo Finance.")

    # Raw article text/timestamps are optional here since embeddings are
    # already precomputed; keep an empty frame with the expected schema.
    empty_news_df = pd.DataFrame(columns=['date', 'equity', 'title', 'content', 'summary', 'source', 'url'])
    empty_news_df['date'] = pd.to_datetime(empty_news_df['date'])

    # ==============================================================================
    # 4. BUILD UNIFIED DATASET
    # ==============================================================================
    print("\n--- Phase C: Building Unified Dataset ---")
    builder = DatasetBuilder()

    dataset = builder.create_synchronized_data(
        processed_price_macro,
        empty_news_df,
        embedding_path=TEMP_EMBED_PATH
    )

    output_filename = 'unified_dataset_test.pkl'
    builder.save(dataset, filename=output_filename)

    if os.path.exists(TEMP_EMBED_PATH):
        os.remove(TEMP_EMBED_PATH)

    print(f"\n✅ PIPELINE COMPLETE. Dataset for {len(target_tickers)} tickers "
          f"saved to processed/{output_filename}")


if __name__ == "__main__":
    run_pipeline_with_precomputed_news_embeddings()
