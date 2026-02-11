"""
Knowledge Graph Setup Script (UPDATED: Voyage embeddings)
"""

import os
import sys
import argparse
from configs.config import GlobalConfig, TrainConfig
from kg_module.simple_kg import SimpleKnowledgeGraph
import pickle


def setup_from_precomputed_triples(triples_path: str):
    """Setup KG using pre-extracted triples + Voyage embeddings."""
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH SETUP (PRECOMPUTED + VOYAGE)")
    print("="*60)
    
    if not os.path.exists(triples_path):
        print(f"❌ Error: Triples file not found at {triples_path}")
        sys.exit(1)
    
    print(f"\n📂 Using precomputed triples from: {triples_path}")
    
    # Initialize KG module with Voyage
    print(f"\n🕸️  Initializing KG module...")
    kg = SimpleKnowledgeGraph(
        tickers=GlobalConfig.TICKERS,
        cache_dir=GlobalConfig.KG_CACHE_DIR,
        use_llm=False,  # Not needed for precomputed
        use_voyage=GlobalConfig.USE_VOYAGE_KG_EMBEDDINGS  # NEW
    )
    
    # Setup (creates Voyage embeddings if enabled)
    kg.setup(force_rebuild=False)
    
    # Process
    print(f"\n⚙️  Processing triples...")
    graphs_by_date = kg.process_from_precomputed_triples(
        triples_parquet_path=triples_path,
        save_graphs=True
    )
    
    # Success message
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   • Dates: {len(graphs_by_date)}")
    print(f"   • Tickers: {', '.join(GlobalConfig.TICKERS)}")
    print(f"   • Embeddings: {'Voyage Finance' if GlobalConfig.USE_VOYAGE_KG_EMBEDDINGS else 'Random'}")
    print(f"   • Output: {GlobalConfig.KG_PROCESSED_PATH}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='precomputed')
    parser.add_argument('--triples-path', type=str, default='D:/kg_triples_day_level.parquet')
    args = parser.parse_args()
    
    setup_from_precomputed_triples(args.triples_path)


if __name__ == "__main__":
    main()