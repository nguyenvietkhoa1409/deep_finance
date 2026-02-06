"""
Stage 3: Graph Neural Network Encoder
Simple GCN for encoding knowledge graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


class SimpleGCN(nn.Module):
    """
    Simple 2-layer GCN encoder for star graphs.
    
    Architecture:
        Input (N, 772) → GCN(772→256) → ReLU → Dropout
                      → GCN(256→128) → ReLU → Dropout
                      → Global Mean Pool → (128,)
    
    Memory: ~2GB (vs 12GB+ for R-GAT)
    """
    
    def __init__(
        self,
        input_dim: int = 772,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize GCN encoder.
        
        Args:
            input_dim: Node feature dimension (772)
            hidden_dim: Hidden layer dimension (256)
            output_dim: Output embedding dimension (128)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Layer 1: 772 → 256
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 2: 256 → 128
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
            x: Node features (N, 772)
            edge_index: Edge connectivity (2, E)
            edge_weight: Edge weights (E,) - optional
        
        Returns:
            Graph embedding (128,)
        
        Examples:
            >>> gcn = SimpleGCN()
            >>> x = torch.randn(10, 772)  # 10 nodes
            >>> edge_index = torch.tensor([[0,1,2], [1,2,0]])  # 3 edges
            >>> embedding = gcn(x, edge_index)
            >>> embedding.shape
            torch.Size([128])
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
        # Average all node embeddings to get graph-level embedding
        graph_embedding = x.mean(dim=0)  # (128,)
        
        return graph_embedding
    
    def forward_batch(
        self,
        batch_graphs: list
    ) -> torch.Tensor:
        """
        Process a batch of graphs.
        
        Args:
            batch_graphs: List of graph dicts, each with:
                - 'node_features': Tensor (N_i, 772)
                - 'edge_index': Tensor (2, E_i)
                - 'edge_weight': Tensor (E_i,)
        
        Returns:
            Batch of embeddings (B, 128)
        
        Examples:
            >>> graphs = [graph1, graph2, graph3]
            >>> embeddings = gcn.forward_batch(graphs)
            >>> embeddings.shape
            torch.Size([3, 128])
        """
        embeddings = []
        
        for graph in batch_graphs:
            x = graph['node_features']
            edge_index = graph['edge_index']
            edge_weight = graph.get('edge_weight')
            
            # Ensure tensors on same device
            x = x.to(self.conv1.lin.weight.device)
            edge_index = edge_index.to(self.conv1.lin.weight.device)
            if edge_weight is not None:
                edge_weight = edge_weight.to(self.conv1.lin.weight.device)
            
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
        input_dim: int = 772,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize KG encoder.
        
        Args:
            input_dim: Node feature dimension
            hidden_dim: GCN hidden dimension
            output_dim: Final embedding dimension (must match MSGCA dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.gcn = SimpleGCN(input_dim, hidden_dim, output_dim, dropout)
        self.output_dim = output_dim
    
    def forward(self, graph_sequences: list) -> torch.Tensor:
        """
        Encode a batch of graph sequences.
        
        Args:
            graph_sequences: List of sequences, each a list of T graphs
                Shape: (Batch, Time, Graph)
                Example: [
                    [graph_day1, graph_day2, ..., graph_day20],  # Sample 1
                    [graph_day1, graph_day2, ..., graph_day20],  # Sample 2
                    ...
                ]
        
        Returns:
            Tensor (B, T, 128) - Embedded graph sequences
        
        Examples:
            >>> encoder = KnowledgeGraphEncoder()
            >>> sequences = [[graph1, graph2], [graph3, graph4]]  # 2 samples, 2 days each
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
        input_dim: int = 772,
        hidden_dim: int = 256,
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