"""
scripts/verify_kg_fix.py

Verify that the regenerated KG embeddings are not all zeros.
"""

import pickle
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import GlobalConfig


def verify_embeddings(kg_path):
    """
    Quick verification of embedding quality
    """
    print("="*80)
    print("🔍 VERIFYING REGENERATED KG EMBEDDINGS")
    print("="*80)
    
    # Load
    print(f"\n📂 Loading from {kg_path}...")
    with open(kg_path, 'rb') as f:
        kg_data = pickle.load(f)
    print(f"   ✓ Loaded {len(kg_data)} dates")
    
    # Sample first non-empty graph
    print("\n🔬 Sampling first non-empty graph...")
    
    sample_graph = None
    for date, ticker_graphs in kg_data.items():
        for ticker, graph in ticker_graphs.items():
            if graph['event'].x.size(0) > 0:
                sample_graph = graph
                print(f"   ✓ Found: {ticker} on {date}")
                break
        if sample_graph:
            break
    
    if not sample_graph:
        print("   ❌ No non-empty graphs found!")
        return False
    
    # Extract features
    event_features = sample_graph['event'].x.cpu().numpy()
    
    print(f"\n📊 Feature Statistics:")
    print(f"   Shape: {event_features.shape}")
    print(f"   Total dims: {event_features.shape[1]}")
    
    # Check structured vs semantic
    STRUCTURED_END = 1037
    SEMANTIC_START = 1037
    
    structured = event_features[:, :STRUCTURED_END]
    semantic = event_features[:, SEMANTIC_START:]
    
    print(f"\n   📈 Structured part (0-1036):")
    print(f"      Mean:   {structured.mean():.6f}")
    print(f"      Std:    {structured.std():.6f}")
    print(f"      Zeros:  {np.mean(structured == 0)*100:.2f}%")
    
    print(f"\n   🎯 Semantic part (1037-1804):")
    print(f"      Mean:   {semantic.mean():.6f}")
    print(f"      Std:    {semantic.std():.6f}")
    print(f"      Zeros:  {np.mean(semantic == 0)*100:.2f}%")
    
    # === CRITICAL CHECK ===
    semantic_all_zeros = np.allclose(semantic, 0)
    
    print("\n" + "="*80)
    if semantic_all_zeros:
        print("❌ SEMANTIC EMBEDDINGS STILL ALL ZEROS!")
        print("="*80)
        print("\n   Voyage API embedding FAILED AGAIN")
        print("\n   Next steps:")
        print("      1. Check Voyage API key validity")
        print("      2. Check API logs for errors")
        print("      3. Try Option B: Use sentence-transformers")
        return False
    else:
        print("✅ SEMANTIC EMBEDDINGS SUCCESSFULLY POPULATED!")
        print("="*80)
        print(f"\n   Mean: {semantic.mean():.6f} (should be ~0)")
        print(f"   Std:  {semantic.std():.6f} (should be >0)")
        print(f"   Non-zero: {np.sum(semantic != 0)}/{semantic.size} values")
        
        # Sample values
        print(f"\n   Sample semantic values (first event, first 10 dims):")
        print(f"      {semantic[0, :10]}")
        
        return True


def compare_before_after(old_path, new_path):
    """
    Compare old (broken) vs new (hopefully fixed) embeddings
    """
    print("\n" + "="*80)
    print("📊 BEFORE vs AFTER COMPARISON")
    print("="*80)
    
    # Load both
    try:
        with open(old_path, 'rb') as f:
            old_data = pickle.load(f)
        print(f"\n   ✓ Loaded OLD: {old_path}")
    except:
        print(f"\n   ⚠️  OLD file not found (OK if deleted)")
        old_data = None
    
    with open(new_path, 'rb') as f:
        new_data = pickle.load(f)
    print(f"   ✓ Loaded NEW: {new_path}")
    
    # Compare
    if old_data:
        # Sample same graph
        date = list(new_data.keys())[0]
        ticker = list(new_data[date].keys())[0]
        
        old_graph = old_data[date][ticker]
        new_graph = new_data[date][ticker]
        
        old_semantic = old_graph['event'].x[:, 1037:].cpu().numpy()
        new_semantic = new_graph['event'].x[:, 1037:].cpu().numpy()
        
        print(f"\n   OLD semantic embeddings:")
        print(f"      Mean: {old_semantic.mean():.6f}")
        print(f"      Std:  {old_semantic.std():.6f}")
        print(f"      Zeros: {np.mean(old_semantic == 0)*100:.1f}%")
        
        print(f"\n   NEW semantic embeddings:")
        print(f"      Mean: {new_semantic.mean():.6f}")
        print(f"      Std:  {new_semantic.std():.6f}")
        print(f"      Zeros: {np.mean(new_semantic == 0)*100:.1f}%")
        
        if np.allclose(old_semantic, new_semantic):
            print("\n   ⚠️  WARNING: Embeddings are IDENTICAL (not regenerated?)")
        else:
            print("\n   ✅ Embeddings are DIFFERENT (successfully regenerated)")


if __name__ == "__main__":
    kg_path = "data/interim/kg_cache/hetero_kg_graphs.pkl"
    
    success = verify_embeddings(kg_path)
    
    # Try comparison if old backup exists
    old_path = "data/interim/kg_cache/hetero_kg_graphs.pkl.backup"
    if Path(old_path).exists():
        compare_before_after(old_path, kg_path)
    
    print("\n" + "="*80)
    if success:
        print("✅ VERIFICATION PASSED - PROCEED TO TRAINING")
        print("="*80)
        print("\nNext step:")
        print("   python main.py")
    else:
        print("❌ VERIFICATION FAILED - DO NOT TRAIN YET")
        print("="*80)
        print("\nNext step:")
        print("   1. Check setup_hybrid_kg.py logs for Voyage errors")
        print("   2. Or use sentence-transformers fallback")