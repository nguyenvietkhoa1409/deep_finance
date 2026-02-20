"""
Heterogeneous Graph Builder for Hybrid KG
Supports ticker nodes (1028-dim) + event nodes (2061-dim)
[FIXED] Metadata attribute names to avoid conflicts
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from torch_geometric.data import HeteroData


class HeteroGraphBuilder:
    """
    Build heterogeneous star graphs for hybrid KG module.
    
    Node Types:
        - 'ticker': Center node (1028-dim)
        - 'event': Event nodes (1805-dim)
    
    Edge Types:
        - ('ticker', 'has_event', 'event')
        - ('event', 'affects', 'ticker')
    """
    
    def __init__(self, ticker_embeddings: Dict[str, np.ndarray]):
        """
        Initialize builder.
        
        Args:
            ticker_embeddings: Dict mapping ticker → embedding (1024-dim)
        """
        self.ticker_embeddings = ticker_embeddings
        
        # Get embedding dimension
        if ticker_embeddings:
            sample_key = next(iter(ticker_embeddings))
            self.ticker_embed_dim = ticker_embeddings[sample_key].shape[0]
        else:
            self.ticker_embed_dim = 1024
        
        # Statistics
        self.stats = {
            'graphs_built': 0,
            'total_event_nodes': 0,
            'empty_graphs': 0
        }
    
    def build_graph(
        self,
        ticker: str,
        event_features: List[np.ndarray],
        current_date: datetime
    ) -> HeteroData:
        """
        Build single heterogeneous star graph.
        
        Args:
            ticker: Stock ticker symbol
            event_features: List of 1805-dim feature vectors
            current_date: Current date (for metadata)
        
        Returns:
            HeteroData object with typed nodes and edges
        """
        graph = HeteroData()
        
        # === Node Features ===
        
        # 1. Ticker node (center)
        ticker_emb = self.ticker_embeddings.get(
            ticker,
            np.zeros(self.ticker_embed_dim, dtype=np.float32)
        )
        # Pad to 1028-dim
        ticker_features = np.concatenate([
            ticker_emb,  # 1024
            np.zeros(4, dtype=np.float32)  # 4 padding
        ])
        
        # [FIXED] Convert to numpy array first, then to tensor
        graph['ticker'].x = torch.tensor(
            np.array([ticker_features], dtype=np.float32),
            dtype=torch.float32
        )  # Shape: (1, 1028)
        
        # 2. Event nodes (surrounding)
        if len(event_features) > 0:
            # [FIXED] Stack numpy arrays first
            event_array = np.stack(event_features, axis=0).astype(np.float32)
            graph['event'].x = torch.tensor(
                event_array,
                dtype=torch.float32
            )  # Shape: (N_events, 1805)
            
            N_events = len(event_features)
            
            # === Edges ===
            
            # Ticker → Event
            graph['ticker', 'has_event', 'event'].edge_index = torch.tensor([
                [0] * N_events,          # Source: ticker node (id=0)
                list(range(N_events))    # Target: all event nodes
            ], dtype=torch.long)
            
            # Event → Ticker
            graph['event', 'affects', 'ticker'].edge_index = torch.tensor([
                list(range(N_events)),   # Source: all event nodes
                [0] * N_events           # Target: ticker node (id=0)
            ], dtype=torch.long)
            
            # Edge weights (uniform for now)
            edge_weights = torch.ones(N_events, dtype=torch.float32)
            
            graph['ticker', 'has_event', 'event'].edge_attr = edge_weights
            graph['event', 'affects', 'ticker'].edge_attr = edge_weights
            
            self.stats['total_event_nodes'] += N_events
        else:
            # Empty graph: only ticker node
            graph['event'].x = torch.zeros((0, 2061), dtype=torch.float32)
            graph['ticker', 'has_event', 'event'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph['event', 'affects', 'ticker'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            self.stats['empty_graphs'] += 1
        
        # [FIXED] Metadata with different names to avoid conflicts
        graph._ticker_symbol = ticker  # Use underscore prefix
        graph._graph_date = current_date
        graph._num_events = len(event_features)
        
        self.stats['graphs_built'] += 1
        
        return graph
    
    def build_batch(
        self,
        features_by_date: Dict[datetime, Dict[str, List[np.ndarray]]],
        ticker: str,
        dates: List[datetime]
    ) -> List[HeteroData]:
        """
        Build graphs for a sequence of dates.
        
        Args:
            features_by_date: {date: {ticker: [features]}}
            ticker: Target ticker
            dates: List of dates to build graphs for
        
        Returns:
            List of HeteroData graphs (one per date)
        """
        graphs = []
        
        for date in dates:
            date_features = features_by_date.get(date, {})
            ticker_features = date_features.get(ticker, [])
            
            graph = self.build_graph(ticker, ticker_features, date)
            graphs.append(graph)
        
        return graphs
    
    def print_stats(self):
        """Print build statistics."""
        print("\n" + "="*50)
        print("🕸️  HETERO GRAPH BUILD STATISTICS")
        print("="*50)
        print(f"Graphs built:       {self.stats['graphs_built']:,}")
        print(f"Total event nodes:  {self.stats['total_event_nodes']:,}")
        print(f"Empty graphs:       {self.stats['empty_graphs']:,}")
        if self.stats['graphs_built'] > 0:
            avg = self.stats['total_event_nodes'] / self.stats['graphs_built']
            print(f"Avg events/graph:   {avg:.1f}")
        print("="*50 + "\n")