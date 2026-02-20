"""
scripts/analyze_feature_distributions.py

Visualize and analyze the distribution of all features.
Check if features have sufficient variance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import data_prepare
from configs.config import GlobalConfig, TrainConfig


def analyze_feature_distributions(pickle_path, ticker='TSLA', output_dir='./feature_analysis'):
    """
    Analyze all feature distributions
    """
    print("="*80)
    print("📊 FEATURE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data for {ticker}...")
    dp = data_prepare(pickle_path, kg_data_path=None)
    train, valid, test = dp.prepare_data(ticker)
    
    if not train:
        print("   ❌ Failed to load data")
        return
    
    print(f"   ✓ Loaded {len(train['label'])} training samples")
    
    # Extract features
    print("\n📦 Extracting features...")
    
    # Price features
    s_o = train['s_o'].numpy()  # (N, T, 1)
    s_h = train['s_h'].numpy()
    s_c = train['s_c'].numpy()
    
    # Other features
    s_m = train['s_m'].numpy()  # (N, T, 6) - Macro
    s_n = train['s_n'].numpy()  # (N, T, 1024) - News
    labels = train['label'].numpy()  # (N,)
    
    print(f"   Price (close): {s_c.shape}")
    print(f"   Macro:         {s_m.shape}")
    print(f"   News:          {s_n.shape}")
    print(f"   Labels:        {labels.shape}")
    
    # === ANALYSIS 1: Price Features ===
    print("\n" + "="*80)
    print("💵 PRICE FEATURES (Log Returns)")
    print("="*80)
    
    price_flat = s_c.flatten()
    
    print(f"\n   Statistics:")
    print(f"      Mean:   {price_flat.mean():.6f}")
    print(f"      Std:    {price_flat.std():.6f}")
    print(f"      Min:    {price_flat.min():.6f}")
    print(f"      Max:    {price_flat.max():.6f}")
    print(f"      Median: {np.median(price_flat):.6f}")
    
    if price_flat.std() < 0.001:
        print("\n   ⚠️  WARNING: Very low std - prices may be constant!")
    
    # === ANALYSIS 2: Macro Features ===
    print("\n" + "="*80)
    print("🌐 MACRO FEATURES")
    print("="*80)
    
    macro_names = ['VIX', 'Yield Spread', 'SP500', 'SP500 Return', 'DXY', 'WTI']
    
    for i, name in enumerate(macro_names):
        feat = s_m[:, :, i].flatten()
        print(f"\n   {name}:")
        print(f"      Mean:   {feat.mean():.6f}")
        print(f"      Std:    {feat.std():.6f}")
        print(f"      Min:    {feat.min():.6f}")
        print(f"      Max:    {feat.max():.6f}")
        
        if feat.std() < 0.01:
            print(f"      ⚠️  Low variance - may not be informative")
    
    # === ANALYSIS 3: News Features ===
    print("\n" + "="*80)
    print("📰 NEWS FEATURES (Embeddings)")
    print("="*80)
    
    news_flat = s_n.flatten()
    
    print(f"\n   Statistics:")
    print(f"      Mean:       {news_flat.mean():.6f}")
    print(f"      Std:        {news_flat.std():.6f}")
    print(f"      Min:        {news_flat.min():.6f}")
    print(f"      Max:        {news_flat.max():.6f}")
    print(f"      Zero ratio: {np.mean(news_flat == 0)*100:.2f}%")
    
    if np.mean(news_flat == 0) > 0.3:
        print("\n   ⚠️  WARNING: High zero ratio - embeddings may be sparse!")
    
    if news_flat.std() < 0.01:
        print("\n   ⚠️  WARNING: Very low std - embeddings may be degenerate!")
    
    # === VISUALIZATION ===
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Figure 1: Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Price
    axes[0, 0].hist(price_flat, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Price (Log Returns)')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(price_flat.mean(), color='red', linestyle='--', label=f'Mean: {price_flat.mean():.4f}')
    axes[0, 0].legend()
    
    # Macro (sample 3 features)
    for idx, i in enumerate([0, 1, 3]):  # VIX, Yield, SP500 Return
        ax = axes[0, idx + 1] if idx < 2 else axes[1, idx - 2]
        feat = s_m[:, :, i].flatten()
        ax.hist(feat, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'{macro_names[i]}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.axvline(feat.mean(), color='red', linestyle='--')
    
    # News
    axes[1, 1].hist(news_flat, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('News Embeddings')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(news_flat.mean(), color='red', linestyle='--', label=f'Mean: {news_flat.mean():.4f}')
    axes[1, 1].legend()
    
    # Labels
    label_counts = np.bincount(labels)
    axes[1, 2].bar(['DOWN', 'FLAT', 'UP'], label_counts, edgecolor='black', alpha=0.7)
    axes[1, 2].set_title('Label Distribution')
    axes[1, 2].set_ylabel('Count')
    
    plt.tight_layout()
    fig.savefig(f'{output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n   ✓ Saved: {output_dir}/feature_distributions.png")
    plt.close()
    
    # Figure 2: Temporal Patterns
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Use first sample
    sample_idx = 0
    
    # Price over time
    axes[0].plot(s_c[sample_idx, :, 0], marker='o', alpha=0.7)
    axes[0].set_title(f'{ticker} - Price (Sample {sample_idx})')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Log Return')
    axes[0].grid(True, alpha=0.3)
    
    # Macro over time (sample VIX)
    axes[1].plot(s_m[sample_idx, :, 0], marker='o', alpha=0.7, label='VIX')
    axes[1].set_title(f'{ticker} - Macro (Sample {sample_idx})')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Value (Normalized)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # News over time (mean of embeddings)
    news_mean = s_n[sample_idx].mean(axis=1)
    axes[2].plot(news_mean, marker='o', alpha=0.7)
    axes[2].set_title(f'{ticker} - News Embedding Mean (Sample {sample_idx})')
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('Mean Embedding Value')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(f'{output_dir}/temporal_patterns.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir}/temporal_patterns.png")
    plt.close()
    
    # Figure 3: Feature vs Label Correlation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Price (last timestep) vs Label
    price_last = s_c[:, -1, 0]
    axes[0, 0].scatter(price_last, labels, alpha=0.5)
    axes[0, 0].set_xlabel('Price (Last Timestep)')
    axes[0, 0].set_ylabel('Label')
    axes[0, 0].set_title(f'Price vs Label (r={np.corrcoef(price_last, labels)[0,1]:.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Macro (last timestep, VIX) vs Label
    macro_last = s_m[:, -1, 0]
    axes[0, 1].scatter(macro_last, labels, alpha=0.5)
    axes[0, 1].set_xlabel('VIX (Last Timestep)')
    axes[0, 1].set_ylabel('Label')
    axes[0, 1].set_title(f'VIX vs Label (r={np.corrcoef(macro_last, labels)[0,1]:.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # News (last timestep, mean) vs Label
    news_last_mean = s_n[:, -1, :].mean(axis=1)
    axes[1, 0].scatter(news_last_mean, labels, alpha=0.5)
    axes[1, 0].set_xlabel('News Embedding Mean (Last Timestep)')
    axes[1, 0].set_ylabel('Label')
    axes[1, 0].set_title(f'News vs Label (r={np.corrcoef(news_last_mean, labels)[0,1]:.4f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    features_for_corr = np.stack([
        price_last,
        s_m[:, -1, 0],  # VIX
        s_m[:, -1, 1],  # Yield
        news_last_mean,
        labels
    ], axis=1)
    
    corr_matrix = np.corrcoef(features_for_corr.T)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', 
                xticklabels=['Price', 'VIX', 'Yield', 'News', 'Label'],
                yticklabels=['Price', 'VIX', 'Yield', 'News', 'Label'],
                cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    fig.savefig(f'{output_dir}/feature_label_correlation.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir}/feature_label_correlation.png")
    plt.close()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n   Results saved to: {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='TSLA')
    parser.add_argument('--data', type=str,
                       default='data/processed/unified_dataset_test.pkl')
    parser.add_argument('--output', type=str, default='./feature_analysis')
    
    args = parser.parse_args()
    
    analyze_feature_distributions(args.data, args.ticker, args.output)