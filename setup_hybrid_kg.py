"""
Setup Hybrid Knowledge Graph from Existing Triples
"""

import os
import sys
import pickle
import pandas as pd
from datetime import datetime
from configs.config import GlobalConfig, TrainConfig
from kg_module.hybrid_extractor import HybridTripleExtractor
from kg_module.hetero_graph_builder import HeteroGraphBuilder
from kg_module.voyage_embedder import VoyageKGEmbedder


def load_triples_from_parquet(path: str) -> list:
    """
    Load triples from parquet file.
    
    Handles numpy arrays and various formats.
    """
    import numpy as np
    
    print(f"\n📂 Loading triples from {path}...")
    df = pd.read_parquet(path)
    
    print(f"   Columns found: {df.columns.tolist()}")
    print(f"   Total rows: {len(df)}")
    
    # Detect column names
    ticker_col = 'equity' if 'equity' in df.columns else 'ticker'
    triples_col = 'triples_day_flat' if 'triples_day_flat' in df.columns else 'triples'
    
    if ticker_col not in df.columns:
        raise ValueError(f"Missing ticker/equity column! Available: {df.columns.tolist()}")
    if triples_col not in df.columns:
        raise ValueError(f"Missing triples column! Available: {df.columns.tolist()}")
    
    print(f"   Using columns: date='date', ticker='{ticker_col}', triples='{triples_col}'")
    
    records = []
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            # Parse date
            date = row['date']
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            # Get ticker
            ticker = row[ticker_col]
            
            # Get triples
            triples_raw = row[triples_col]
            
            # Convert based on type
            if isinstance(triples_raw, np.ndarray):
                # Convert numpy array to list
                triples = triples_raw.tolist()
            elif isinstance(triples_raw, list):
                triples = triples_raw
            elif isinstance(triples_raw, str):
                import json
                triples = json.loads(triples_raw)
            else:
                print(f"   ⚠️ Row {idx}: Unknown type {type(triples_raw)}, skipping")
                skipped += 1
                continue
            
            # Validate triple structure
            valid_triples = []
            for triple in triples:
                # Handle nested numpy arrays
                if isinstance(triple, np.ndarray):
                    triple = triple.tolist()
                
                if isinstance(triple, list) and len(triple) == 3:
                    # Convert elements to strings
                    valid_triples.append([str(triple[0]), str(triple[1]), str(triple[2])])
                elif isinstance(triple, dict) and all(k in triple for k in ['subject', 'predicate', 'object']):
                    valid_triples.append([
                        str(triple['subject']), 
                        str(triple['predicate']), 
                        str(triple['object'])
                    ])
            
            if len(valid_triples) > 0:
                records.append({
                    'date': date,
                    'ticker': ticker,
                    'triples': valid_triples
                })
            else:
                skipped += 1
        
        except Exception as e:
            print(f"   ⚠️ Row {idx}: Error {str(e)}")
            skipped += 1
            continue
    
    total_triples = sum(len(r['triples']) for r in records)
    
    print(f"   ✓ Loaded {len(records)} records with {total_triples} total triples")
    if skipped > 0:
        print(f"   ⚠️ Skipped {skipped} rows")
    
    # Show sample
    if len(records) > 0:
        sample = records[0]
        print(f"\n   📋 Sample record:")
        print(f"      Date: {sample['date']}")
        print(f"      Ticker: {sample['ticker']}")
        print(f"      Triples: {len(sample['triples'])} triples")
        if len(sample['triples']) > 0:
            print(f"      Example: {sample['triples'][0]}")
    else:
        print("\n   ❌ NO VALID RECORDS! Check data format.")
    
    return records

def load_news_context(news_path: str):
    """Load news DataFrame for document context."""
    if not os.path.exists(news_path):
        print("   ⚠️ News file not found, continuing without document context")
        return None
    
    print(f"\n📰 Loading news context from {news_path}...")
    
    # Try parquet first
    if news_path.endswith('.parquet'):
        news_df = pd.read_parquet(news_path)
    else:
        # Fallback to CSV
        news_df = pd.read_csv(news_path)
    
    print(f"   ✓ Loaded {len(news_df)} news articles")
    return news_df


def main():
    print("\n" + "="*60)
    print("🚀 HYBRID KNOWLEDGE GRAPH SETUP")
    print("="*60)
    
    # ===  Paths ===
    triples_path = "D:/kg_triples_day_level.parquet"  # Your existing triples
    news_path = os.path.join(GlobalConfig.RAW_NEWS_PATH, "03_primary/news.parquet")
    output_dir = GlobalConfig.KG_CACHE_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # === Step 1: Load Data ===
    triples_records = load_triples_from_parquet(triples_path)
    news_df = load_news_context(news_path)
    
    # === Step 2: Initialize Components ===
    print("\n🔧 Initializing hybrid extractor...")
    
    extractor = HybridTripleExtractor(
        cache_dir=output_dir,
        use_voyage=GlobalConfig.USE_VOYAGE_KG_EMBEDDINGS
    )
    
    # === Step 3: Process Triples (Dual-Path Extraction) ===
    print("\n⚙️  Extracting hybrid features...")
    
    features_by_date = extractor.process_batch(
        triples_records,
        news_df=news_df,
        max_workers=4
    )
    
    # Save hybrid features
    features_path = os.path.join(output_dir, 'hybrid_kg_features.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(features_by_date, f)
    
    print(f"\n💾 Saved hybrid features to {features_path}")
    
    # === Step 4: Build Heterogeneous Graphs ===
    print("\n🕸️  Building heterogeneous graphs...")
    
    # Load ticker embeddings
    voyage_embedder = VoyageKGEmbedder(cache_dir=output_dir)
    ticker_embeddings = voyage_embedder.generate_ticker_embeddings(
        GlobalConfig.TICKERS,
        force_rebuild=False
    )
    
    # Initialize graph builder
    graph_builder = HeteroGraphBuilder(ticker_embeddings)
    
    # Build graphs for all dates and tickers
    all_dates = sorted(features_by_date.keys())
    graphs_by_date = {}
    
    for date in all_dates:
        graphs_by_date[date] = {}
        
        for ticker in GlobalConfig.TICKERS:
            event_features = features_by_date[date].get(ticker, [])
            
            graph = graph_builder.build_graph(
                ticker, event_features, date
            )
            
            graphs_by_date[date][ticker] = graph
    
    # Save graphs
    graphs_path = os.path.join(output_dir, 'hetero_kg_graphs.pkl')
    with open(graphs_path, 'wb') as f:
        pickle.dump(graphs_by_date, f)
    
    print(f"\n💾 Saved heterogeneous graphs to {graphs_path}")
    
    # === Statistics ===
    graph_builder.print_stats()
    
    print("\n" + "="*60)
    print("✅ HYBRID KG SETUP COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   • Dates processed: {len(all_dates)}")
    print(f"   • Tickers: {', '.join(GlobalConfig.TICKERS)}")
    print(f"   • Total graphs: {len(all_dates) * len(GlobalConfig.TICKERS)}")
    print(f"   • Output: {graphs_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()