# src/data_loader.py
import pandas as pd
import numpy as np
import torch
from configs.config import TrainConfig, GlobalConfig
import pickle
import os
from datetime import datetime

# Try import PyG
try:
    from torch_geometric.data import Data, HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠️ Warning: torch_geometric not found. KG features will be disabled.")

class data_prepare:
    def __init__(
        self, 
        data_path, 
        kg_data_path=None,
        hybrid_kg_path=None,
        use_hetero_kg=True
    ) -> None:
        """
        Initialize data loader.
        
        Args:
            data_path: Path to main pickle (date-keyed format)
            kg_data_path: [LEGACY] Path to old KG embeddings
            hybrid_kg_path: [NEW] Path to hybrid KG graphs
            use_hetero_kg: Use heterogeneous graphs (True) or legacy (False)
        """
        self.data_path = data_path
        self.kg_data_path = kg_data_path
        self.hybrid_kg_path = hybrid_kg_path
        self.use_hetero_kg = use_hetero_kg
        
        # Load KG data
        self.kg_data = None
        
        if use_hetero_kg and hybrid_kg_path:
            self._load_hybrid_kg_data()
        elif kg_data_path and os.path.exists(kg_data_path):
            self._load_legacy_kg_data()

    def _load_hybrid_kg_data(self):
        """Load hybrid heterogeneous graphs."""
        if self.hybrid_kg_path is None:
            self.hybrid_kg_path = os.path.join(
                GlobalConfig.KG_CACHE_DIR,
                'hetero_kg_graphs.pkl'
            )
        
        if not os.path.exists(self.hybrid_kg_path):
            print(f"   ⚠️  Hybrid KG graphs not found at {self.hybrid_kg_path}")
            return
        
        print(f"🕸️  Loading HYBRID KG graphs from {self.hybrid_kg_path}...")
        try:
            with open(self.hybrid_kg_path, 'rb') as f:
                self.kg_data = pickle.load(f)
            print(f"   ✓ Loaded hetero graphs for {len(self.kg_data)} dates")
        except Exception as e:
            print(f"   ❌ Error loading hybrid KG: {e}")
            self.kg_data = None

    def _load_legacy_kg_data(self):
        """Load legacy KG embeddings."""
        print(f"📊 Loading LEGACY KG data from {self.kg_data_path}...")
        try:
            with open(self.kg_data_path, 'rb') as f:
                self.kg_data = pickle.load(f)
            print(f"   ✓ Loaded legacy KG data")
        except Exception as e:
            print(f"   ❌ Error loading legacy KG: {e}")
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
    
    def _convert_hetero_to_legacy(self, hetero_graph):
        """
        [NEW] Convert HeteroData to legacy Data format for backward compatibility.
        
        Strategy: Extract ticker node features as graph embedding
        """
        if not HAS_PYG:
            return None
        
        if isinstance(hetero_graph, HeteroData):
            # Extract ticker node embedding
            ticker_features = hetero_graph['ticker'].x  # Shape: (1, 1028)
            
            # Create simple Data object with single node
            return Data(
                x=ticker_features,
                edge_index=torch.zeros((2, 0), dtype=torch.long)
            )
        
        return hetero_graph
    
    def _convert_dict_to_data(self, graph_dict):
        """Convert Graph Dict to PyG Data Object."""
        if not HAS_PYG:
            return None
            
        # Already a Data object
        if isinstance(graph_dict, (Data, HeteroData)):
            return graph_dict
        
        # Convert dict to Data
        if isinstance(graph_dict, dict):
            node_features = graph_dict.get('node_features') or graph_dict.get('x')
            edge_index = graph_dict.get('edge_index')
            
            if node_features is None:
                return self._get_empty_graph()
            
            # Handle both numpy and tensor
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
            
            return Data(x=node_features, edge_index=edge_index)
            
        return self._get_empty_graph()

    def _get_empty_graph(self):
        """Create empty graph."""
        if not HAS_PYG:
            return None
        
        dim = getattr(TrainConfig, 'kg_input_dim', 1028)
        x = torch.zeros((1, dim), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
    
    def _get_empty_hetero_graph(self):
        """[NEW] Create empty HeteroData graph."""
        if not HAS_PYG:
            return None
        
        graph = HeteroData()
        graph['ticker'].x = torch.zeros((1, 1028), dtype=torch.float32)
        graph['event'].x = torch.zeros((0, 1805), dtype=torch.float32)
        graph['ticker', 'has_event', 'event'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        graph['event', 'affects', 'ticker'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        return graph

    def _get_kg_sequence(self, dates, ticker, window_size):
        """
        Extract KG graph sequence for given dates.
        
        [UPDATED] Supports both legacy Data and HeteroData
        
        Returns: 
            - List of Lists of HeteroData (if use_hetero_kg=True)
            - List of Lists of Data (if use_hetero_kg=False)
        """
        if not self.kg_data:
            return None
        
        all_sequences = []
        
        # Iterate to create windows
        for i in range(len(dates) - window_size + 1):
            window_dates = dates[i : i + window_size]
            
            sequence = []
            for date in window_dates:
                # Normalize date key
                date_key = self._normalize_date_key(date)
                
                # Fetch graph
                graph = None
                if date_key in self.kg_data:
                    if ticker in self.kg_data[date_key]:
                        raw_graph = self.kg_data[date_key][ticker]
                        
                        # [NEW] Handle HeteroData
                        if isinstance(raw_graph, HeteroData):
                            graph = raw_graph
                        elif isinstance(raw_graph, Data):
                            graph = raw_graph
                        else:
                            graph = self._convert_dict_to_data(raw_graph)
                
                # Fallback to empty
                if graph is None:
                    if self.use_hetero_kg:
                        graph = self._get_empty_hetero_graph()
                    else:
                        graph = self._get_empty_graph()
                
                sequence.append(graph)
            
            all_sequences.append(sequence)
        
        return all_sequences

    def _normalize_date_key(self, date):
        """
        Normalize date key for robust lookup.
        
        Handles: str, datetime, pd.Timestamp, datetime.date
        """
        if not self.kg_data:
            return date
        
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
            # Also try datetime.date comparison
            if isinstance(key, datetime) and hasattr(date, 'date'):
                if key.date() == date.date():
                    return key
        
        return date

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

        # Align Step 2
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
        
        # Input slicing
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
                    
                    # Ensure same length
                    min_len = min(len(price_win), len(kg_sequences))
                    kg_sequences = kg_sequences[:min_len]
                    
                    print(f"   ✓ Prepared {len(kg_sequences)} KG sequences")
                    if self.use_hetero_kg:
                        print(f"   ✓ Format: HeteroData (hybrid features)")
                    else:
                        print(f"   ✓ Format: Data (legacy)")
                else:
                    print("   ⚠️  Failed to generate KG sequences")
                    
            except Exception as e:
                print(f"   ❌ KG sequence error: {e}")
                import traceback
                traceback.print_exc()
                kg_sequences = None

        # ==========================
        # 4. LABELING
        # ==========================
        full_returns_series = pd.Series(return_np.flatten())
        rolling_window = 20
        
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
        
        start_idx = window_size - 1 + future_days
        if start_idx < len(labels_temp):
            label_all = labels_temp[start_idx:]
        else:
            label_all = np.array([])

        # Synchronize lengths
        min_len = min(len(price_win), len(label_all))
        if kg_sequences:
            min_len = min(min_len, len(kg_sequences))
            
        price_win = price_win[:min_len]
        macro_win = macro_win[:min_len]
        news_win  = news_win[:min_len]
        label_all = label_all[:min_len]
        if kg_sequences:
            kg_sequences = kg_sequences[:min_len]

        # Label stats
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
        # 5. SPLIT
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