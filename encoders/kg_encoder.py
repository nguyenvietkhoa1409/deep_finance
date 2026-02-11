"""
Stage 3: Graph Neural Network Encoder (FIXED: PyG Data support)
Simple GCN for encoding knowledge graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from typing import Optional, Union, List, Dict


class SimpleGCN(nn.Module):
    """
    Simple 2-layer GCN encoder for star graphs.
    
    FIXED: Support both dict and PyG Data objects
    
    Architecture:
        Input (N, 1028) → GCN(1028→512) → ReLU → Dropout
                        → GCN(512→128) → ReLU → Dropout
                        → Global Mean Pool → (128,)
    """
    
    def __init__(
        self,
        input_dim: int = 1028,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize GCN encoder.
        
        Args:
            input_dim: Node feature dimension (1028 for full Voyage)
            hidden_dim: Hidden layer dimension (512)
            output_dim: Output embedding dimension (128)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Layer 1: 1028 → 512
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 2: 512 → 128
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # For handling empty graphs (no events)
        self.register_buffer('zero_embedding', torch.zeros(output_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (N, 1028)
            edge_index: Edge connectivity (2, E)
            edge_weight: Edge weights (E,) - optional
        
        Returns:
            Graph embedding (128,)
        """
        # Handle empty graph (single ticker node, no events)
        if x.size(0) == 1:
            return self.zero_embedding.clone()
        
        # Layer 1
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global mean pooling
        graph_embedding = x.mean(dim=0)  # (128,)
        
        return graph_embedding
    
    def forward_batch(
        self,
        batch_graphs: Union[List[Dict], List[Data]]
    ) -> torch.Tensor:
        """
        Process a batch of graphs.
        
        FIXED: Support both dict and PyG Data objects
        
        Args:
            batch_graphs: List of graphs, each can be:
                - Dict with keys: 'node_features', 'edge_index', 'edge_weight'
                - PyG Data object with attrs: x, edge_index, edge_weight
        
        Returns:
            Batch of embeddings (B, 128)
        
        Examples:
            >>> # With dicts
            >>> graphs = [
            ...     {'node_features': tensor(...), 'edge_index': tensor(...)},
            ...     {'node_features': tensor(...), 'edge_index': tensor(...)}
            ... ]
            >>> embeddings = gcn.forward_batch(graphs)
            
            >>> # With PyG Data objects
            >>> graphs = [Data(x=tensor(...), edge_index=tensor(...)), ...]
            >>> embeddings = gcn.forward_batch(graphs)
        """
        embeddings = []
        
        for graph in batch_graphs:
            # [FIX] Extract features based on type
            if isinstance(graph, Data):
                # PyG Data object - use attributes
                x = graph.x
                edge_index = graph.edge_index
                edge_weight = getattr(graph, 'edge_weight', None)
                
            elif isinstance(graph, dict):
                # Dict - use keys
                x = graph.get('node_features') or graph.get('x')
                edge_index = graph.get('edge_index')
                edge_weight = graph.get('edge_weight')
                
            else:
                raise TypeError(f"Expected Data or dict, got {type(graph)}")
            
            # Ensure tensors on same device
            device = self.conv1.lin.weight.device
            x = x.to(device)
            edge_index = edge_index.to(device)
            if edge_weight is not None:
                edge_weight = edge_weight.to(device)
            
            emb = self.forward(x, edge_index, edge_weight)
            embeddings.append(emb)
        
        return torch.stack(embeddings)  # (B, 128)


class KnowledgeGraphEncoder(nn.Module):
    """
    Full KG encoding module for integration with MSGCA.
    
    Wraps SimpleGCN and handles temporal sequencing.
    """
    
    def __init__(
        self,
        input_dim: int = 1028,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize KG encoder.
        
        Args:
            input_dim: Node feature dimension (1028 for full Voyage)
            hidden_dim: GCN hidden dimension (512)
            output_dim: Final embedding dimension (128, must match MSGCA dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.gcn = SimpleGCN(input_dim, hidden_dim, output_dim, dropout)
        self.output_dim = output_dim
    
    def forward(
        self, 
        graph_sequences: List[List[Union[Dict, Data]]]
    ) -> torch.Tensor:
        """
        Encode a batch of graph sequences.
        
        Args:
            graph_sequences: List of sequences, each a list of T graphs
                Shape: (Batch, Time, Graph)
                Each graph can be Dict or PyG Data
                
                Example: [
                    [graph_day1, graph_day2, ..., graph_day20],  # Sample 1
                    [graph_day1, graph_day2, ..., graph_day20],  # Sample 2
                    ...
                ]
        
        Returns:
            Tensor (B, T, 128) - Embedded graph sequences
        
        Examples:
            >>> encoder = KnowledgeGraphEncoder()
            >>> sequences = [[graph1, graph2], [graph3, graph4]]
            >>> output = encoder(sequences)
            >>> output.shape
            torch.Size([2, 2, 128])
        """
        batch_size = len(graph_sequences)
        seq_length = len(graph_sequences[0])
        
        # Flatten: (B, T) → (B*T,) graphs
        all_graphs = []
        for seq in graph_sequences:
            all_graphs.extend(seq)
        
        # Encode all graphs in batch
        embeddings = self.gcn.forward_batch(all_graphs)  # (B*T, 128)
        
        # Reshape back to (B, T, 128)
        embeddings = embeddings.view(batch_size, seq_length, self.output_dim)
        
        return embeddings


# ============================================
# OPTIONAL: Advanced Variant (If GPU allows)
# ============================================

class GraphSAGEEncoder(nn.Module):
    """
    Alternative: GraphSAGE encoder (slightly better for star graphs).
    
    Use this if you have 10-12GB GPU and want ~1-2% better performance.
    """
    
    def __init__(
        self,
        input_dim: int = 1028,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        from torch_geometric.nn import SAGEConv
        
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('zero_embedding', torch.zeros(output_dim))
    
    def forward(self, x, edge_index):
        if x.size(0) == 1:
            return self.zero_embedding.clone()
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        return x.mean(dim=0)