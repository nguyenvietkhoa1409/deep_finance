"""
scripts/test_label_generation.py

Compare current rolling quantile labels vs simple threshold labels.
Test if label generation is causing zero correlation.
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import GlobalConfig, TrainConfig


def load_price_data(pickle_path, ticker):
    """Load raw price data from main pickle"""
    print(f"\n📂 Loading data for {ticker}...")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract prices
    prices = []
    dates = []
    
    for date, content in sorted(data.items()):
        if 'price' in content and ticker in content['price']:
            price_dict = content['price'][ticker]
            if 'close' in price_dict:
                prices.append(float(price_dict['close']))
                dates.append(date)
    
    print(f"   ✓ Loaded {len(prices)} price points")
    
    return np.array(prices), dates


def generate_rolling_quantile_labels(prices, window=20):
    """
    Current method: Rolling quantile (from data_loader.py)
    """
    print("\n🔄 Generating ROLLING QUANTILE labels...")
    
    # Calculate returns
    returns = np.zeros(len(prices))
    for i in range(1, len(prices)):
        returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
    
    # Convert to pandas for rolling
    returns_series = pd.Series(returns)
    
    # Rolling quantiles (shifted to avoid look-ahead)
    roll_low = returns_series.rolling(window=window).quantile(0.33).shift(1)
    roll_high = returns_series.rolling(window=window).quantile(0.66).shift(1)
    
    # Generate labels
    labels = np.full(len(returns), 1, dtype=int)  # Default: FLAT
    
    labels[returns < roll_low] = 0   # DOWN
    labels[returns > roll_high] = 2  # UP
    
    # Noise filter
    labels[np.abs(returns) < 0.001] = 1  # FLAT
    
    # NaN handling
    labels[np.isnan(roll_low)] = 1
    
    # Distribution
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    
    print(f"   Distribution:")
    print(f"      DOWN: {dist.get(0, 0)} ({dist.get(0, 0)/len(labels)*100:.1f}%)")
    print(f"      FLAT: {dist.get(1, 0)} ({dist.get(1, 0)/len(labels)*100:.1f}%)")
    print(f"      UP:   {dist.get(2, 0)} ({dist.get(2, 0)/len(labels)*100:.1f}%)")
    
    return labels, returns


def generate_simple_threshold_labels(prices, threshold=0.005):
    """
    Simple method: Direct price movement with threshold
    
    Args:
        prices: Array of close prices
        threshold: Movement threshold (0.005 = 0.5%)
    """
    print(f"\n📈 Generating SIMPLE THRESHOLD labels (threshold={threshold*100:.1f}%)...")
    
    labels = np.zeros(len(prices) - 1, dtype=int)
    
    for i in range(len(prices) - 1):
        change = (prices[i+1] - prices[i]) / prices[i]
        
        if change > threshold:
            labels[i] = 2  # UP
        elif change < -threshold:
            labels[i] = 0  # DOWN
        else:
            labels[i] = 1  # FLAT
    
    # Pad first label
    labels = np.concatenate([[1], labels])
    
    # Distribution
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    
    print(f"   Distribution:")
    print(f"      DOWN: {dist.get(0, 0)} ({dist.get(0, 0)/len(labels)*100:.1f}%)")
    print(f"      FLAT: {dist.get(1, 0)} ({dist.get(1, 0)/len(labels)*100:.1f}%)")
    print(f"      UP:   {dist.get(2, 0)} ({dist.get(2, 0)/len(labels)*100:.1f}%)")
    
    return labels


def compute_feature_label_correlation(prices, labels):
    """
    Compute correlation between price features and labels
    """
    print("\n📊 Computing correlations...")
    
    # Feature 1: Recent price change
    price_change = np.zeros(len(prices))
    for i in range(1, len(prices)):
        price_change[i] = (prices[i] - prices[i-1]) / prices[i-1]
    
    # Feature 2: Price momentum (5-day)
    momentum_5d = np.zeros(len(prices))
    for i in range(5, len(prices)):
        momentum_5d[i] = (prices[i] - prices[i-5]) / prices[i-5]
    
    # Feature 3: Price volatility (10-day)
    volatility = np.zeros(len(prices))
    for i in range(10, len(prices)):
        recent_returns = price_change[i-10:i]
        volatility[i] = np.std(recent_returns)
    
    # Correlations (skip NaN values)
    valid_mask = ~(np.isnan(price_change) | np.isnan(labels))
    
    corr_change = np.corrcoef(price_change[valid_mask], labels[valid_mask])[0, 1]
    corr_momentum = np.corrcoef(momentum_5d[valid_mask], labels[valid_mask])[0, 1]
    corr_volatility = np.corrcoef(volatility[valid_mask], labels[valid_mask])[0, 1]
    
    print(f"\n   Correlations with labels:")
    print(f"      Price change (1d):  {corr_change:.4f}")
    print(f"      Momentum (5d):      {corr_momentum:.4f}")
    print(f"      Volatility (10d):   {corr_volatility:.4f}")
    
    return {
        'price_change': corr_change,
        'momentum': corr_momentum,
        'volatility': corr_volatility
    }


def test_label_methods(pickle_path, ticker='TSLA'):
    """
    Main test function
    """
    print("="*80)
    print("🧪 LABEL GENERATION METHOD COMPARISON")
    print("="*80)
    
    # Load data
    prices, dates = load_price_data(pickle_path, ticker)
    
    # Method 1: Rolling quantile (current)
    labels_rolling, returns = generate_rolling_quantile_labels(prices)
    corr_rolling = compute_feature_label_correlation(prices, labels_rolling)
    
    # Method 2: Simple threshold (0.5%)
    labels_simple_05 = generate_simple_threshold_labels(prices, threshold=0.005)
    corr_simple_05 = compute_feature_label_correlation(prices, labels_simple_05)
    
    # Method 3: Simple threshold (1.0%)
    labels_simple_10 = generate_simple_threshold_labels(prices, threshold=0.010)
    corr_simple_10 = compute_feature_label_correlation(prices, labels_simple_10)
    
    # Method 4: Simple threshold (0.2%)
    labels_simple_02 = generate_simple_threshold_labels(prices, threshold=0.002)
    corr_simple_02 = compute_feature_label_correlation(prices, labels_simple_02)
    
    # Comparison
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<30} | {'Price Change Corr':<20} | {'Momentum Corr':<20}")
    print("-"*80)
    print(f"{'Rolling Quantile (current)':<30} | {corr_rolling['price_change']:>18.4f} | {corr_rolling['momentum']:>18.4f}")
    print(f"{'Simple Threshold (0.2%)':<30} | {corr_simple_02['price_change']:>18.4f} | {corr_simple_02['momentum']:>18.4f}")
    print(f"{'Simple Threshold (0.5%)':<30} | {corr_simple_05['price_change']:>18.4f} | {corr_simple_05['momentum']:>18.4f}")
    print(f"{'Simple Threshold (1.0%)':<30} | {corr_simple_10['price_change']:>18.4f} | {corr_simple_10['momentum']:>18.4f}")
    
    # Interpretation
    print("\n" + "="*80)
    print("💡 INTERPRETATION")
    print("="*80)
    
    max_corr = max(
        abs(corr_rolling['price_change']),
        abs(corr_simple_05['price_change']),
        abs(corr_simple_10['price_change']),
        abs(corr_simple_02['price_change'])
    )
    
    if max_corr < 0.10:
        print("\n⚠️  ALL methods show WEAK correlation (<0.10)")
        print("   → Task may be fundamentally unpredictable")
        print("   → Consider:")
        print("      - Shorter prediction horizon (intraday)")
        print("      - Different target (volatility, direction confidence)")
        print("      - Additional features (order flow, sentiment)")
    
    elif max_corr < 0.20:
        print("\n⚠️  Correlations are LOW (0.10-0.20)")
        print("   → Features have weak predictive power")
        print("   → Model will struggle to learn")
        print("   → Expect modest performance (MCC ~0.05-0.10)")
    
    else:
        print("\n✅ Correlations are REASONABLE (>0.20)")
        print("   → Features have predictive power")
        print("   → Model should be able to learn")
        
        # Find best method
        methods = {
            'Rolling Quantile': abs(corr_rolling['price_change']),
            'Simple 0.2%': abs(corr_simple_02['price_change']),
            'Simple 0.5%': abs(corr_simple_05['price_change']),
            'Simple 1.0%': abs(corr_simple_10['price_change'])
        }
        best_method = max(methods, key=methods.get)
        print(f"\n   → Best method: {best_method} (corr = {methods[best_method]:.4f})")
    
    # Agreement between methods
    print("\n" + "="*80)
    print("🔀 LABEL AGREEMENT BETWEEN METHODS")
    print("="*80)
    
    # Ensure same length
    min_len = min(len(labels_rolling), len(labels_simple_05))
    
    agreement_05 = np.mean(labels_rolling[:min_len] == labels_simple_05[:min_len])
    agreement_10 = np.mean(labels_rolling[:min_len] == labels_simple_10[:min_len])
    
    print(f"\n   Rolling vs Simple 0.5%: {agreement_05*100:.1f}% agreement")
    print(f"   Rolling vs Simple 1.0%: {agreement_10*100:.1f}% agreement")
    
    if agreement_05 < 0.5:
        print("\n   ⚠️  Low agreement → Methods label differently")
        print("   → Rolling quantile may be over-smoothing")
    else:
        print("\n   ✅ High agreement → Methods are consistent")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='TSLA',
                       help='Ticker to analyze')
    parser.add_argument('--data', type=str, 
                       default='data/processed/unified_dataset_test.pkl',
                       help='Path to pickle file')
    
    args = parser.parse_args()
    
    test_label_methods(args.data, args.ticker)