"""
Quick test script to validate precomputed triples format.

Usage:
    python quick_test_precomputed.py
"""

import pandas as pd
import numpy as np
import sys

def test_precomputed_triples(parquet_path: str):
    """
    Validate that precomputed triples file has correct format.
    """
    print("\n" + "="*60)
    print("🧪 TESTING PRECOMPUTED TRIPLES FORMAT")
    print("="*60)
    
    # Load file
    print(f"\n📂 Loading: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"   ✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        sys.exit(1)
    
    # Check columns
    print("\n📋 Checking columns...")
    required = ['date', 'equity', 'triples_day_flat']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"   ❌ Missing columns: {missing}")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    else:
        print(f"   ✓ All required columns present")
    
    # Check data types
    print("\n🔍 Checking data...")
    
    # Sample row
    sample = df.iloc[0]
    print(f"\n   Sample row:")
    print(f"   • Date: {sample['date']} (type: {type(sample['date'])})")
    print(f"   • Equity: {sample['equity']}")
    print(f"   • Num triples: {len(sample['triples_day_flat'])}")
    
    # Sample triple
    # [FIX]: Kiểm tra độ dài thay vì if sample[...] trực tiếp
    triples_list = sample['triples_day_flat']
    
    if len(triples_list) > 0:
        first_triple = triples_list[0]
        print(f"\n   Sample triple:")
        print(f"   • Type: {type(first_triple)}")
        print(f"   • Length: {len(first_triple)}")
        print(f"   • Content: {first_triple}")
        
        # Xử lý trường hợp first_triple là numpy array
        if isinstance(first_triple, np.ndarray):
            first_triple = first_triple.tolist()

        if len(first_triple) == 3:
            print(f"   • Subject: '{first_triple[0]}'")
            print(f"   • Predicate: '{first_triple[1]}'")
            print(f"   • Object: '{first_triple[2]}'")
            print("\n   ✓ Triple format looks correct!")
        else:
            print(f"\n   ⚠️  Warning: Triple has {len(first_triple)} elements (expected 3)")
    else:
        print("\n   ⚠️  Sample row has no triples.")
    
    # Statistics
    print("\n📊 Statistics:")
    print(f"   • Total rows: {len(df):,}")
    print(f"   • Unique dates: {df['date'].nunique():,}")
    print(f"   • Unique tickers: {df['equity'].nunique()}")
    print(f"   • Tickers: {sorted(df['equity'].unique())}")
    
    # Check if 'num_triples_after_flatten_dedup' exists, else calculate
    if 'num_triples_after_flatten_dedup' in df.columns:
        print(f"   • Total triples: {df['num_triples_after_flatten_dedup'].sum():,}")
        print(f"   • Avg triples/day/ticker: {df['num_triples_after_flatten_dedup'].mean():.1f}")
    else:
        # Fallback calculation if column missing
        total_triples = df['triples_day_flat'].apply(len).sum()
        print(f"   • Total triples: {total_triples:,}")
        print(f"   • Avg triples/day/ticker: {total_triples / len(df):.1f}")
    
    # Date range
    # Convert safely
    try:
        df['date'] = pd.to_datetime(df['date'])
        print(f"\n📅 Date range:")
        print(f"   • Start: {df['date'].min()}")
        print(f"   • End: {df['date'].max()}")
        print(f"   • Days: {(df['date'].max() - df['date'].min()).days}")
    except Exception as e:
        print(f"\n⚠️  Could not parse dates: {e}")
    
    print("\n" + "="*60)
    print("✅ FORMAT VALIDATION PASSED")
    print("="*60)
    print("\nYou can now run:")
    print("  python setup_kg.py --mode precomputed --triples-path <your_path>")
    print()


if __name__ == "__main__":
    # Your path
    path = r'D:\kg_triples_day_level.parquet'
    test_precomputed_triples(path)