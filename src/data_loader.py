# src/data_loader.py
import pandas as pd
import numpy as np
import torch
from configs.config import TrainConfig
import pickle
import os
from datetime import datetime

# Cố gắng import PyG Data
try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠️ Warning: torch_geometric not found. KG features will be disabled.")

class data_prepare:
    def __init__(self, data_path, kg_data_path=None) -> None:
        self.data_path = data_path
        self.kg_data_path = kg_data_path
        
        # Load KG data if provided
        self.kg_data = None
        if kg_data_path and os.path.exists(kg_data_path) and HAS_PYG:
            print(f"📦 Loading KG data from {kg_data_path}...")
            try:
                with open(kg_data_path, 'rb') as f:
                    self.kg_data = pickle.load(f)
                print(f"   ✓ Loaded KG graphs for {len(self.kg_data)} dates")
            except Exception as e:
                print(f"   ❌ Error loading KG data: {e}")
                self.kg_data = None

    def create_return(self, price_df):
        df = price_df.copy()
        df["return"] = df["close"] / df["close"].shift(1) - 1
        df.dropna(inplace=True)
        return df[["return"]]

    def make_window(self, data, window_size):
        """
        data: numpy array (T, D)
        return: (N, window_size, D)
        """
        X = []
        for i in range(len(data) - window_size + 1):
            X.append(data[i:i + window_size])
        return np.array(X)
    
    def _convert_dict_to_data(self, graph_dict):
        """
        Convert Graph Dict to PyG Data Object.
        
        FIXED: Proper tensor comparison
        """
        if not HAS_PYG:
            return None
            
        # Already a Data object
        if isinstance(graph_dict, Data):
            return graph_dict
        
        # Convert dict to Data
        if isinstance(graph_dict, dict):
            # [FIX] Handle key names
            node_features = graph_dict.get('node_features')
            if node_features is None:
                node_features = graph_dict.get('x')
            
            edge_index = graph_dict.get('edge_index')
            
            # [FIX] Proper validation - check shape instead of len
            if node_features is None:
                return self._get_empty_graph()
            
            # [FIX] Handle both numpy and tensor
            if isinstance(node_features, np.ndarray):
                if node_features.shape[0] == 0:
                    return self._get_empty_graph()
                node_features = torch.from_numpy(node_features).float()
            elif isinstance(node_features, torch.Tensor):
                if node_features.shape[0] == 0:
                    return self._get_empty_graph()
            else:
                return self._get_empty_graph()
            
            # Convert edge_index if needed
            if isinstance(edge_index, np.ndarray):
                edge_index = torch.from_numpy(edge_index).long()
            
            # Create Data object
            return Data(
                x=node_features,
                edge_index=edge_index
            )
            
        return self._get_empty_graph()

    def _get_empty_graph(self):
        """
        Create empty graph with correct dimensions.
        
        FIXED: Use consistent dimension from config
        """
        if not HAS_PYG:
            return None
        
        # Get dimension from config (should be 1028 for full Voyage)
        dim = getattr(TrainConfig, 'kg_input_dim', 1028)
        
        x = torch.zeros((1, dim), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def _get_kg_sequence(self, dates, ticker, window_size):
        """
        Extract KG graph sequence for given dates.
        Returns: List of Lists of PyG Data objects (length = N windows)
        """
        if not self.kg_data:
            return None
        
        all_sequences = []
        
        # Iterate to create windows (Logic matches make_window)
        for i in range(len(dates) - window_size + 1):
            window_dates = dates[i : i + window_size]
            
            sequence = []
            for date in window_dates:
                # 1. Normalize Date Key
                date_key = date
                # Try direct lookup first
                if date_key not in self.kg_data:
                    # Try converting to python datetime if it's Timestamp
                    if isinstance(date, pd.Timestamp):
                        date_key = date.to_pydatetime()
                    elif isinstance(date, str):
                        try:
                            date_key = datetime.fromisoformat(date)
                        except: pass
                
                # 2. Fetch Graph
                graph = None
                if date_key in self.kg_data and ticker in self.kg_data[date_key]:
                    raw_graph = self.kg_data[date_key][ticker]
                    graph = self._convert_dict_to_data(raw_graph)
                
                # 3. Fallback to Empty
                if graph is None:
                    graph = self._get_empty_graph()
                
                sequence.append(graph)
            
            all_sequences.append(sequence)
        
        return all_sequences

    def _normalize_date_key(self, date):
        """
        FIXED: Normalize date key for robust lookup.
        
        Handles: str, datetime, pd.Timestamp
        """
        # Try as-is first (fast path)
        if date in self.kg_data:
            return date
        
        # Convert to datetime
        if isinstance(date, str):
            try:
                date = datetime.fromisoformat(date.replace('Z', ''))
            except:
                try:
                    date = pd.to_datetime(date).to_pydatetime()
                except:
                    pass
        elif isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        
        # Try again
        if date in self.kg_data:
            return date
        
        # Try date-only (strip time)
        if hasattr(date, 'date'):
            date_only = date.date()
            if date_only in self.kg_data:
                return date_only
        
        # Last resort: try all keys with matching date
        for key in self.kg_data.keys():
            if hasattr(key, 'date') and hasattr(date, 'date'):
                if key.date() == date.date():
                    return key
        
        return date  # Return original if no match

    def prepare_data(
        self,
        stock_name,
        window_size=TrainConfig.window_size, 
        future_days=1,
        train_ratio=getattr(TrainConfig, 'train_ratio', 0.70),
        valid_ratio=getattr(TrainConfig, 'valid_ratio', 0.15),
    ):
        # ==========================
        # 1. LOAD DATA
        # ==========================
        try:
            Data = pd.read_pickle(self.data_path)
        except Exception as e:
            print(f"❌ Error loading pickle file: {e}")
            return {}, {}, {}

        rows = {}
        
        for d, content in Data.items():
            if "price" not in content or stock_name not in content["price"]:
                continue

            price = content["price"][stock_name]
            macro = content["macro"]

            # News embedding
            news_section = content.get("news_embedding", {})
            raw_vec = news_section.get(stock_name)

            if raw_vec is None:
                news_vec = np.zeros(TrainConfig.news_embed_dim, dtype=np.float32)
            else:
                try:
                    news_vec = np.array(raw_vec, dtype=np.float32)
                except Exception:
                    news_vec = np.zeros(TrainConfig.news_embed_dim, dtype=np.float32)

            news_dict = {f"news_{i}": v for i, v in enumerate(news_vec)}

            rows[d] = {
                **price,
                **macro,
                **news_dict
            }
        
        if not rows:
            print(f"❌ No data found for stock {stock_name}")
            return {}, {}, {}

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.sort_index(inplace=True)

        # ==========================
        # 2. PRE-PROCESS FEATURES
        # ==========================
        price_df = df[["open", "high", "close"]].astype(float)

        macro_df = df[
            ["vix", "yield_spread_10y_2y",
             "sp500", "sp500_return", "dxy", "wti"]
        ].astype(float)

        news_cols = [c for c in df.columns if c.startswith("news_")]
        news_df = df[news_cols]
        news_df = news_df.apply(pd.to_numeric, errors="coerce")
        news_df = news_df.fillna(0.0)

        # Create Return DataFrame
        return_df = self.create_return(price_df)

        # Align Step 1
        price_df = price_df.loc[return_df.index]
        macro_df = macro_df.loc[return_df.index]
        news_df  = news_df.loc[return_df.index]

        # Log-Return for Input Price
        price_df = np.log(price_df / price_df.shift(1))
        price_df.dropna(inplace=True)

        # Align Step 2 (Final Alignment)
        macro_df  = macro_df.loc[price_df.index]
        news_df   = news_df.loc[price_df.index]
        return_df = return_df.loc[price_df.index]

        # Macro Clean
        macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
        macro_df = macro_df.ffill().bfill()

        # ==========================
        # 3. WINDOWING
        # ==========================
        price_np  = price_df.values
        macro_np  = macro_df.values
        news_np   = news_df.values
        return_np = return_df.values

        price_win = self.make_window(price_np, window_size)
        macro_win = self.make_window(macro_np, window_size)
        news_win  = self.make_window(news_np, window_size)

        # Label slicing
        label_raw = return_np[window_size - 1 + future_days:]
        
        # Input slicing (Align with labels)
        price_win = price_win[:-future_days]
        macro_win = macro_win[:-future_days]
        news_win  = news_win[:-future_days]

        # ==========================
        # NEW: KG WINDOWING
        # ==========================
        kg_sequences = None
        if TrainConfig.use_kg and self.kg_data and HAS_PYG:
            print(f"   🕸️  Preparing KG sequences for {stock_name}...")
            dates = list(price_df.index)
            
            try:
                kg_sequences = self._get_kg_sequence(dates, stock_name, window_size)
                
                if kg_sequences:
                    # Align with other data
                    kg_sequences = kg_sequences[:-future_days]
                    
                    # [FIX] Ensure same length
                    min_len = min(len(price_win), len(kg_sequences))
                    kg_sequences = kg_sequences[:min_len]
                    
                    print(f"   ✓ Prepared {len(kg_sequences)} KG sequences")
                else:
                    print("   ⚠️  Failed to generate KG sequences")
                    
            except Exception as e:
                print(f"   ❌ KG sequence error: {e}")
                kg_sequences = None

        # ==========================
        # 4. LABELING (Rolling Quantile)
        # ==========================
        full_returns_series = pd.Series(return_np.flatten())
        rolling_window = 20
        
        # Calculate thresholds (Shift 1 to avoid look-ahead)
        roll_low  = full_returns_series.rolling(window=rolling_window).quantile(0.33).shift(1)
        roll_high = full_returns_series.rolling(window=rolling_window).quantile(0.66).shift(1)
        
        labels_temp = np.full(len(full_returns_series), 1, dtype=int)
        
        is_down = full_returns_series < roll_low
        is_up   = full_returns_series > roll_high
        is_noise = full_returns_series.abs() < 0.001
        
        labels_temp[is_down] = 0
        labels_temp[is_up]   = 2
        labels_temp[is_noise] = 1
        labels_temp[np.isnan(roll_low)] = 1
        
        # Align labels
        start_idx = window_size - 1 + future_days
        if start_idx < len(labels_temp):
            label_all = labels_temp[start_idx:]
        else:
            label_all = np.array([])

        # Check lengths consistency
        min_len = min(len(price_win), len(label_all))
        if kg_sequences:
            min_len = min(min_len, len(kg_sequences))
            
        # Truncate to min_len to ensure synchronization
        price_win = price_win[:min_len]
        macro_win = macro_win[:min_len]
        news_win  = news_win[:min_len]
        label_all = label_all[:min_len]
        if kg_sequences:
            kg_sequences = kg_sequences[:min_len]

        # Label Distribution Logging
        unique, counts = np.unique(label_all, return_counts=True)
        dist = dict(zip(unique, counts))
        total_lbl = sum(counts)
        print(f" ⚖️  Label Distribution: {dist}")
        if total_lbl > 0:
            p0 = dist.get(0,0)/total_lbl
            p1 = dist.get(1,0)/total_lbl
            p2 = dist.get(2,0)/total_lbl
            print(f"      Down: {p0:.2%}, Flat: {p1:.2%}, Up: {p2:.2%}")

        # ==========================
        # 5. SPLIT DATASETS
        # ==========================
        total_len = len(price_win)
        idx_train = int(total_len * train_ratio)
        idx_valid = int(total_len * (train_ratio + valid_ratio))

        # Normalization
        macro_mean = macro_win[:idx_train].mean(axis=(0, 1), keepdims=True)
        macro_std  = macro_win[:idx_train].std(axis=(0, 1), keepdims=True) + 1e-6
        
        news_mean = news_win[:idx_train].mean(axis=(0, 1), keepdims=True)
        news_std  = news_win[:idx_train].std(axis=(0, 1), keepdims=True) + 1e-6

        macro_win = (macro_win - macro_mean) / macro_std
        news_win  = (news_win - news_mean) / news_std

        def create_dataset(start, end):
            if start >= end:
                return {}
            
            dataset = {
                "s_o": torch.tensor(price_win[start:end, :, 0:1], dtype=torch.float32),
                "s_h": torch.tensor(price_win[start:end, :, 1:2], dtype=torch.float32),
                "s_c": torch.tensor(price_win[start:end, :, 2:3], dtype=torch.float32),
                "s_m": torch.tensor(macro_win[start:end], dtype=torch.float32),
                "s_n": torch.tensor(news_win[start:end], dtype=torch.float32),
                "label": torch.tensor(label_all[start:end], dtype=torch.long),
            }
            
            # Add KG sequences if available
            if kg_sequences:
                dataset["s_kg"] = kg_sequences[start:end]
            
            return dataset

        train_data = create_dataset(0, idx_train)
        valid_data = create_dataset(idx_train, idx_valid)
        test_data  = create_dataset(idx_valid, total_len)

        print(f"Stats: Train={len(train_data.get('label', []))}, Valid={len(valid_data.get('label', []))}, Test={len(test_data.get('label', []))}")
        
        return train_data, valid_data, test_data