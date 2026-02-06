"""
Knowledge Graph Setup Script (UPDATED: Support precomputed triples)

Two modes:
    A. Extract triples from news (original)
    B. Load precomputed triples from parquet (NEW)

Usage:
    # Mode A: Extract triples (requires unified_dataset.pkl)
    python setup_kg.py --mode extract
    
    # Mode B: Use precomputed triples (faster, no LLM cost)
    python setup_kg.py --mode precomputed --triples-path kg_triples_day_level.parquet

Time:
    Mode A: ~1-2 hours
    Mode B: ~10-15 minutes
"""

import os
import sys
import argparse
from configs.config import GlobalConfig, TrainConfig
from kg_module.simple_kg import SimpleKnowledgeGraph
import pickle


def setup_from_precomputed_triples(triples_path: str):
    """
    Setup KG using pre-extracted triples.
    
    This is the RECOMMENDED approach if you already have triples.
    """
    print("\n" + "="*60)
    print("🏗️  KNOWLEDGE GRAPH SETUP (PRECOMPUTED TRIPLES MODE)")
    print("="*60)
    
    # Validate triples file
    if not os.path.exists(triples_path):
        print(f"❌ Error: Triples file not found at {triples_path}")
        sys.exit(1)
    
    print(f"\n📂 Using precomputed triples from: {triples_path}")
    
    # Initialize KG module
    print(f"\n🕸️  Initializing KG module...")
    kg = SimpleKnowledgeGraph(
        tickers=GlobalConfig.TICKERS,
        cache_dir=GlobalConfig.KG_CACHE_DIR,
        use_llm=False  # Not needed for precomputed
    )
    
    # Setup (create ticker embeddings)
    kg.setup(force_rebuild=False)
    
    # Process from precomputed triples
    print(f"\n⚙️  Processing precomputed triples...")
    print(f"   Tickers: {', '.join(GlobalConfig.TICKERS)}")
    print(f"   This should take ~10-15 minutes...\n")
    
    graphs_by_date = kg.process_from_precomputed_triples(
        triples_parquet_path=triples_path,
        save_graphs=True
    )
    
    # Final message
    print("\n" + "="*60)
    print("✅ KNOWLEDGE GRAPH SETUP COMPLETE")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   • Processed dates: {len(graphs_by_date)}")
    print(f"   • Tickers: {', '.join(GlobalConfig.TICKERS)}")
    print(f"   • Output: {GlobalConfig.KG_PROCESSED_PATH}")
    print(f"\n🚀 You can now run main.py to train with KG!")
    print(f"   Make sure TrainConfig.use_kg = True in configs/config.py")
    print("="*60 + "\n")


def setup_from_news_extraction():
    """
    Setup KG by extracting triples from news (original method).
    
    Use this if you DON'T have pre-extracted triples.
    """
    print("\n" + "="*60)
    print("🏗️  KNOWLEDGE GRAPH SETUP (EXTRACTION MODE)")
    print("="*60)
    
    # Paths
    unified_data_path = os.path.join(
        GlobalConfig.PROCESSED_PATH,
        "unified_dataset.pkl"
    )
    
    if not os.path.exists(unified_data_path):
        print(f"❌ Error: Unified dataset not found at {unified_data_path}")
        print("   Please run data pipeline first to create unified_dataset.pkl")
        sys.exit(1)
    
    print(f"\n📂 Loading unified dataset...")
    with open(unified_data_path, 'rb') as f:
        unified_dataset = pickle.load(f)
    
    print(f"   ✓ Loaded data for {len(unified_dataset)} dates")
    
    # Initialize KG module
    print(f"\n🕸️  Initializing KG module...")
    kg = SimpleKnowledgeGraph(
        tickers=GlobalConfig.TICKERS,
        cache_dir=GlobalConfig.KG_CACHE_DIR,
        use_llm=True  # Enable LLM fallback
    )
    
    # Setup
    kg.setup(force_rebuild=False)
    
    # Process news → triples → graphs
    print(f"\n⚙️  Processing news data...")
    print(f"   This may take 1-2 hours...")
    
    graphs_by_date = kg.process_news_data(
        unified_dataset,
        save_triples=True
    )
    
    # Save
    output_filename = "kg_graphs.pkl"
    kg.save(graphs_by_date, output_filename)
    
    # Final message
    print("\n" + "="*60)
    print("✅ KNOWLEDGE GRAPH SETUP COMPLETE")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   • Processed dates: {len(graphs_by_date)}")
    print(f"   • Tickers: {', '.join(GlobalConfig.TICKERS)}")
    print(f"   • Output: {GlobalConfig.KG_PROCESSED_PATH}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Setup Knowledge Graph for Stock Prediction"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['precomputed', 'extract'],
        default='precomputed',
        help='Setup mode: precomputed (use existing triples) or extract (extract from news)'
    )
    
    parser.add_argument(
        '--triples-path',
        type=str,
        default='D:/kg_triples_day_level.parquet',
        help='Path to precomputed triples parquet file (only for precomputed mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'precomputed':
        setup_from_precomputed_triples(args.triples_path)
    else:
        setup_from_news_extraction()


if __name__ == "__main__":
    main()