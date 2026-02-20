"""
scripts/debug_event_embeddings.py

Diagnose why 43% of event embeddings are zeros.
Check if it's Voyage API failure or feature engineering issue.
"""

import numpy as np
import pickle
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import GlobalConfig


def analyze_event_embeddings(kg_path, sample_size=100):
    """
    Analyze event embedding quality
    """
    print("="*80)
    print("🔬 EVENT EMBEDDING QUALITY ANALYSIS")
    print("="*80)
    
    # Load KG data
    print(f"\n📂 Loading KG data from {kg_path}...")
    
    try:
        with open(kg_path, 'rb') as f:
            kg_data = pickle.load(f)
        print(f"   ✓ Loaded {len(kg_data)} dates")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Sample graphs
    print(f"\n📊 Sampling {sample_size} graphs...")
    
    all_event_features = []
    graph_info = []
    
    count = 0
    for date, ticker_graphs in kg_data.items():
        if count >= sample_size:
            break
        
        for ticker, graph in ticker_graphs.items():
            if count >= sample_size:
                break
            
            # Extract event features
            event_x = graph['event'].x.cpu().numpy()
            
            if event_x.shape[0] > 0:  # Non-empty graph
                all_event_features.append(event_x)
                graph_info.append({
                    'date': date,
                    'ticker': ticker,
                    'num_events': event_x.shape[0]
                })
                count += 1
    
    if not all_event_features:
        print("   ❌ No non-empty graphs found!")
        return
    
    print(f"   ✓ Collected {len(all_event_features)} graphs")
    
    # Concatenate all features
    all_features = np.concatenate(all_event_features, axis=0)
    print(f"\n   Total event nodes: {all_features.shape[0]}")
    print(f"   Feature dimension: {all_features.shape[1]}")
    
    # === ANALYSIS 1: Overall Statistics ===
    print("\n" + "="*80)
    print("📈 OVERALL STATISTICS")
    print("="*80)
    
    print(f"\n   Mean:       {all_features.mean():.6f}")
    print(f"   Std:        {all_features.std():.6f}")
    print(f"   Min:        {all_features.min():.6f}")
    print(f"   Max:        {all_features.max():.6f}")
    print(f"   Median:     {np.median(all_features):.6f}")
    
    zero_ratio = np.mean(all_features == 0)
    print(f"\n   Zero ratio: {zero_ratio*100:.2f}%")
    
    if zero_ratio > 0.4:
        print("   ⚠️  WARNING: >40% zeros detected!")
    
    # === ANALYSIS 2: By Feature Dimension ===
    print("\n" + "="*80)
    print("📊 ANALYSIS BY DIMENSION")
    print("="*80)
    
    # Check which dimensions are all zeros
    zero_dims = np.where((all_features == 0).all(axis=0))[0]
    
    print(f"\n   Dimensions that are ALL zeros: {len(zero_dims)}/{all_features.shape[1]}")
    
    # Structured vs Semantic split
    # Event features: 1805 = 1037 (structured) + 768 (semantic)
    STRUCTURED_END = 1037
    SEMANTIC_START = 1037
    SEMANTIC_END = 1805
    
    structured_zeros = zero_dims[zero_dims < STRUCTURED_END]
    semantic_zeros = zero_dims[(zero_dims >= SEMANTIC_START) & (zero_dims < SEMANTIC_END)]
    
    print(f"\n   Structured part (0-1036):")
    print(f"      All-zero dims: {len(structured_zeros)}/1037 ({len(structured_zeros)/1037*100:.1f}%)")
    
    print(f"\n   Semantic part (1037-1804):")
    print(f"      All-zero dims: {len(semantic_zeros)}/768 ({len(semantic_zeros)/768*100:.1f}%)")
    
    # === CRITICAL CHECK ===
    if len(semantic_zeros) == 768:
        print("\n" + "="*80)
        print("🔴 CRITICAL ISSUE DETECTED")
        print("="*80)
        print("\n   ALL semantic dimensions are zeros!")
        print("   → Voyage API embeddings COMPLETELY FAILED")
        print("\n   Possible causes:")
        print("      1. Voyage API key invalid/expired")
        print("      2. Network errors during embedding")
        print("      3. Silent fallback to zero vectors")
        print("\n   → MUST regenerate embeddings!")
        
    elif len(semantic_zeros) > 384:  # >50% of semantic
        print("\n" + "="*80)
        print("⚠️  WARNING: PARTIAL SEMANTIC FAILURE")
        print("="*80)
        print(f"\n   {len(semantic_zeros)}/768 semantic dims are zero")
        print("   → Partial Voyage API failure")
        print("   → Some triples embedded, others failed")
    
    # === ANALYSIS 3: Per-Feature Statistics ===
    print("\n" + "="*80)
    print("📉 PER-DIMENSION STATISTICS")
    print("="*80)
    
    # Sample first 10 structured and first 10 semantic
    print("\n   Structured features (first 10):")
    for i in range(min(10, STRUCTURED_END)):
        feat = all_features[:, i]
        nonzero = np.sum(feat != 0)
        print(f"      Dim {i:4d}: mean={feat.mean():>8.4f}, std={feat.std():>8.4f}, "
              f"nonzero={nonzero}/{len(feat)} ({nonzero/len(feat)*100:.1f}%)")
    
    print("\n   Semantic features (first 10):")
    for i in range(SEMANTIC_START, min(SEMANTIC_START + 10, SEMANTIC_END)):
        feat = all_features[:, i]
        nonzero = np.sum(feat != 0)
        print(f"      Dim {i:4d}: mean={feat.mean():>8.4f}, std={feat.std():>8.4f}, "
              f"nonzero={nonzero}/{len(feat)} ({nonzero/len(feat)*100:.1f}%)")
    
    # === ANALYSIS 4: Temporal Pattern ===
    print("\n" + "="*80)
    print("📅 TEMPORAL PATTERN")
    print("="*80)
    
    # Group by date
    features_by_date = defaultdict(list)
    for info, features in zip(graph_info, all_event_features):
        features_by_date[info['date']].append(features)
    
    # Compute zero ratio over time
    sorted_dates = sorted(features_by_date.keys())
    
    print("\n   Zero ratio by date (first 10):")
    for date in sorted_dates[:10]:
        date_features = np.concatenate(features_by_date[date], axis=0)
        zero_ratio = np.mean(date_features == 0)
        print(f"      {date}: {zero_ratio*100:.1f}%")
    
    print("\n   Zero ratio by date (last 10):")
    for date in sorted_dates[-10:]:
        date_features = np.concatenate(features_by_date[date], axis=0)
        zero_ratio = np.mean(date_features == 0)
        print(f"      {date}: {zero_ratio*100:.1f}%")
    
    # Check if worsening over time
    early_zeros = np.mean([np.mean(np.concatenate(features_by_date[d], axis=0) == 0) 
                           for d in sorted_dates[:10]])
    late_zeros = np.mean([np.mean(np.concatenate(features_by_date[d], axis=0) == 0) 
                          for d in sorted_dates[-10:]])
    
    if late_zeros > early_zeros * 1.2:
        print(f"\n   ⚠️  Zero ratio INCREASING over time!")
        print(f"      Early: {early_zeros*100:.1f}% → Late: {late_zeros*100:.1f}%")
        print("      → Possible API rate limiting or quota exhaustion")
    
    # === ANALYSIS 5: Sample Inspection ===
    print("\n" + "="*80)
    print("🔍 SAMPLE INSPECTION")
    print("="*80)
    
    # Show a few complete event feature vectors
    print("\n   Sample event #1:")
    sample1 = all_event_features[0][0]
    print(f"      Structured (first 10): {sample1[:10]}")
    print(f"      Semantic (first 10):   {sample1[1037:1047]}")
    
    if len(all_event_features[0]) > 1:
        print("\n   Sample event #2:")
        sample2 = all_event_features[0][1]
        print(f"      Structured (first 10): {sample2[:10]}")
        print(f"      Semantic (first 10):   {sample2[1037:1047]}")
    
    # === RECOMMENDATIONS ===
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    if len(semantic_zeros) == 768:
        print("\n   🔴 URGENT: Regenerate ALL embeddings")
        print("      Command: python setup_hybrid_kg.py --force-rebuild")
        
    elif zero_ratio > 0.5:
        print("\n   ⚠️  HIGH: Feature quality is poor")
        print("      Options:")
        print("         1. Check Voyage API logs")
        print("         2. Regenerate with error handling")
        print("         3. Use alternative embeddings (BERT, sentence-transformers)")
        
    else:
        print("\n   ✅ Embeddings are acceptable")
        print(f"      Zero ratio: {zero_ratio*100:.1f}% (within normal range)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg-path', type=str,
                       default='data/interim/kg_cache/hetero_kg_graphs.pkl',
                       help='Path to KG graphs')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of graphs to sample')
    
    args = parser.parse_args()
    
    analyze_event_embeddings(args.kg_path, args.sample_size)