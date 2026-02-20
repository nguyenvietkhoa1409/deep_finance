"""
Heterogeneous Knowledge Graph Encoder
Handles ticker nodes (1028-dim) + event nodes (1805-dim)
[FIXED] add_self_loops=False for hetero graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData
from typing import List, Dict, Optional


class HeteroKGEncoder(nn.Module):
    """
    Heterogeneous GNN encoder for hybrid knowledge graphs.
    
    Architecture:
        Input: HeteroData with 'ticker' and 'event' nodes
        
        Layer 0: Type-specific projections
            ticker (1028) → hidden_dim
            event (1805) → hidden_dim
        
        Layer 1: Heterogeneous GAT convolution
            event --affects--> ticker (aggregation)
        
        Output: ticker node embedding (output_dim, default 128)
    """
    
    def __init__(
        self,
        ticker_input_dim: int = 1028,
        event_input_dim: int = 2061,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize heterogeneous encoder.
        
        Args:
            ticker_input_dim: Ticker node feature dimension
            event_input_dim: Event node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Final output dimension (must match MSGCA dim)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.ticker_input_dim = ticker_input_dim
        self.event_input_dim = event_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # === Type-Specific Input Projections ===
        self.ticker_proj = nn.Sequential(
            nn.Linear(ticker_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.event_proj = nn.Sequential(
            nn.Linear(event_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # === Heterogeneous Convolution Layer 1 ===
        # [FIXED] add_self_loops=False for heterogeneous message passing
        self.conv1 = HeteroConv({
            ('event', 'affects', 'ticker'): GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                edge_dim=1,  # Edge weight dimension
                dropout=dropout,
                concat=False,  # Average multi-head outputs
                add_self_loops=False  # [CRITICAL FIX] Must be False for hetero graphs
            )
        }, aggr='mean')
        
        # === Output Projection ===
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # === For Empty Graphs ===
        self.register_buffer('zero_embedding', torch.zeros(output_dim))
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
        edge_attr_dict: Optional[Dict[tuple, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass for single graph.
        
        Args:
            x_dict: {'ticker': (1, 1028), 'event': (N, 1805)}
            edge_index_dict: {('event', 'affects', 'ticker'): (2, N)}
            edge_attr_dict: {('event', 'affects', 'ticker'): (N,)} [optional]
        
        Returns:
            Ticker embedding (output_dim,)
        """
        # Check if empty graph
        if x_dict['event'].size(0) == 0:
            return self.zero_embedding.clone()
        
        # === Step 1: Type-Specific Projections ===
        h_dict = {
            'ticker': self.ticker_proj(x_dict['ticker']),  # (1, hidden)
            'event': self.event_proj(x_dict['event'])      # (N, hidden)
        }
        
        # === Step 2: Heterogeneous Convolution ===
        # Prepare edge attributes if provided
        if edge_attr_dict and ('event', 'affects', 'ticker') in edge_attr_dict:
            edge_attr = edge_attr_dict[('event', 'affects', 'ticker')]
            # Expand to match GATConv edge_dim expectation
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)  # (N, 1)
        else:
            edge_attr = None
        
        # Apply heterogeneous convolution
        if edge_attr is not None:
            h_dict = self.conv1(
                h_dict,
                edge_index_dict,
                edge_attr_dict={('event', 'affects', 'ticker'): edge_attr}
            )
        else:
            h_dict = self.conv1(h_dict, edge_index_dict)
        
        # === Step 3: Output Projection ===
        ticker_emb = h_dict['ticker'].squeeze(0)  # (hidden,)
        output = self.output_proj(ticker_emb)     # (output_dim,)
        
        return output
    
    def forward_hetero_data(self, graph: HeteroData) -> torch.Tensor:
        """
        Forward pass for HeteroData object.
        
        Args:
            graph: HeteroData with 'ticker' and 'event' nodes
        
        Returns:
            Ticker embedding (output_dim,)
        """
        x_dict = {'ticker': graph['ticker'].x, 'event': graph['event'].x}
        
        edge_index_dict = {
            ('event', 'affects', 'ticker'): graph['event', 'affects', 'ticker'].edge_index
        }
        
        # Extract edge attributes if available
        edge_attr_dict = None
        if hasattr(graph['event', 'affects', 'ticker'], 'edge_attr'):
            edge_attr_dict = {
                ('event', 'affects', 'ticker'): graph['event', 'affects', 'ticker'].edge_attr
            }
        
        return self.forward(x_dict, edge_index_dict, edge_attr_dict)
    
    def forward_batch(self, graphs: List[HeteroData]) -> torch.Tensor:
        """
        Process batch of graphs.
        
        Args:
            graphs: List of HeteroData objects
        
        Returns:
            Batch of embeddings (B, output_dim)
        """
        embeddings = []
        
        for graph in graphs:
            # Move graph to same device as model
            device = next(self.parameters()).device
            graph = graph.to(device)
            
            emb = self.forward_hetero_data(graph)
            embeddings.append(emb)
        
        return torch.stack(embeddings)  # (B, output_dim)


class HeteroKGSequenceEncoder(nn.Module):
    """
    Wrapper for encoding sequences of heterogeneous graphs.
    
    Input: List of Lists of HeteroData (Batch, Time)
    Output: Tensor (Batch, Time, output_dim)
    """
    
    def __init__(
        self,
        ticker_input_dim: int = 1028,
        event_input_dim: int = 2061,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = HeteroKGEncoder(
            ticker_input_dim=ticker_input_dim,
            event_input_dim=event_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.output_dim = output_dim
    
    def forward(
        self,
        graph_sequences: List[List[HeteroData]]
    ) -> torch.Tensor:
        """
        Encode batch of graph sequences.
        
        Args:
            graph_sequences: List of sequences, each a list of T graphs
                Shape: (Batch, Time)
        
        Returns:
            Tensor (Batch, Time, output_dim)
        """
        batch_size = len(graph_sequences)
        seq_length = len(graph_sequences[0])
        
        # Flatten
        all_graphs = []
        for seq in graph_sequences:
            all_graphs.extend(seq)
        
        # Encode all graphs
        embeddings = self.encoder.forward_batch(all_graphs)  # (B*T, output_dim)
        
        # Reshape
        embeddings = embeddings.view(batch_size, seq_length, self.output_dim)
        
        return embeddings