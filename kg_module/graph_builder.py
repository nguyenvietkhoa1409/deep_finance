"""
Stage 2: Star Graph Construction
Build ticker-centric graphs from extracted triples
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .config import (
    RELATION_TYPES, MAX_EVENTS_PER_GRAPH, 
    MIN_CERTAINTY_THRESHOLD, NODE_FEATURE_DIM
)
from .utils import compute_temporal_decay
from .voyage_embedder import VoyageKGEmbedder, adapt_embeddings_dimension  # NEW


class StarGraphBuilder:
    """
    Build star topology graphs for Knowledge Graph module.
    
    Graph Structure:
        - Center node: Ticker (e.g., AMZN)
        - Surrounding nodes: Events
        - Edges: All connect center to events (star topology)
    
    Node Features:
        - Center: [ticker_embedding (768,)]
        - Event: [relation_embedding (768,), magnitude, polarity, certainty, decay]
    """
    
    def __init__(
        self,
        ticker_embeddings: Dict[str, np.ndarray],
        use_voyage: bool = True,
        voyage_embedder: Optional['VoyageKGEmbedder'] = None
    ):
        """
        Initialize graph builder.
        
        Args:
            ticker_embeddings: Dict mapping ticker → embedding (1024-dim)
            use_voyage: Use Voyage embeddings
            voyage_embedder: VoyageKGEmbedder instance
        """
        self.ticker_embeddings = ticker_embeddings
        self.use_voyage = use_voyage
        self.voyage_embedder = voyage_embedder
        
        # Get embedding dimension from first ticker
        if ticker_embeddings:
            sample_key = next(iter(ticker_embeddings))
            self.embedding_dim = ticker_embeddings[sample_key].shape[0]
            print(f"   📐 Embedding dimension: {self.embedding_dim}")
        else:
            self.embedding_dim = 1024  # Default
        
        # Initialize relation embeddings
        self.relation_embeddings = self._init_relation_embeddings()
        
        # Statistics
        self.stats = {
            'graphs_built': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'avg_nodes_per_graph': 0,
            'empty_graphs': 0,
            'failed_builds': 0
        }
    
    def _init_relation_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Initialize relation embeddings.
        
        UPDATED: Use full 1024-dim
        """
        if self.use_voyage and self.voyage_embedder:
            print("   Using Voyage AI for relation embeddings...")
            
            embeddings = self.voyage_embedder.generate_relation_embeddings(
                RELATION_TYPES,
                force_rebuild=False
            )
            
            # REMOVED: adapt_embeddings_dimension()
            # Return full 1024-dim
            
            return embeddings
        
        else:
            print("   Using random initialization...")
            embeddings = {}
            for rel_type in RELATION_TYPES.keys():
                # Match Voyage dimension
                embeddings[rel_type] = np.random.randn(1024).astype(np.float32) * 0.01
            return embeddings
    
    def build_graph(
        self,
        triples: List[Dict],
        ticker: str,
        current_date: datetime
    ) -> Dict:
        """
        Build a single star graph from triples.
        
        FIXED:
        - Efficient tensor creation (numpy stack → torch)
        - Consistent dimensions
        - Proper error handling
        """
        try:
            # Filter valid triples
            valid_triples = [
                t for t in triples
                if t.get('certainty', 0) >= MIN_CERTAINTY_THRESHOLD
                and t.get('relation') in RELATION_TYPES
            ]
            
            # Limit number of events
            if len(valid_triples) > MAX_EVENTS_PER_GRAPH:
                scored_triples = []
                for t in valid_triples:
                    decay = compute_temporal_decay(
                        t['date'], current_date, t['relation']
                    )
                    score = t['certainty'] * decay
                    scored_triples.append((score, t))
                
                scored_triples.sort(reverse=True, key=lambda x: x[0])
                valid_triples = [t for _, t in scored_triples[:MAX_EVENTS_PER_GRAPH]]
            
            # Build star structure
            nodes, edges, edge_weights = self._build_star_structure(
                valid_triples, ticker, current_date
            )
            
            # [FIX] Efficient tensor conversion
            # Stack numpy arrays first, then convert to tensor
            if len(nodes) > 0:
                nodes_array = np.stack(nodes, axis=0)  # (N, 1028)
                node_features = torch.from_numpy(nodes_array).float()
            else:
                # Empty graph
                node_features = torch.zeros((1, self.embedding_dim + 4), dtype=torch.float32)
            
            # Convert edges
            if len(edges) > 0:
                edges_array = np.array(edges, dtype=np.int64).T  # (2, E)
                edge_index = torch.from_numpy(edges_array).long()
                
                weights_array = np.array(edge_weights, dtype=np.float32)
                edge_weight = torch.from_numpy(weights_array).float()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_weight = torch.zeros(0, dtype=torch.float32)
            
            # Update stats
            self.stats['graphs_built'] += 1
            self.stats['total_nodes'] += len(nodes) if nodes else 1
            self.stats['total_edges'] += len(edges)
            if len(nodes) <= 1:
                self.stats['empty_graphs'] += 1
            
            return {
                'node_features': node_features,
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'num_nodes': node_features.shape[0],
                'num_edges': edge_index.shape[1],
                'ticker': ticker,
                'date': current_date
            }
            
        except Exception as e:
            # Fallback: create empty graph
            self.stats['failed_builds'] += 1
            print(f"   ⚠️  Graph build failed for {ticker} on {current_date}: {e}")
            return self._create_empty_graph_fallback(ticker, current_date)
    
    def _create_empty_graph_fallback(self, ticker: str, current_date: datetime) -> Dict:
        """
        Create empty graph when build fails.
        
        FIXED: Use consistent embedding dimension
        """
        ticker_embedding = self.ticker_embeddings.get(
            ticker, np.zeros(self.embedding_dim, dtype=np.float32)  # Use class attribute
        )
        
        # Create single node: ticker + padding
        ticker_features = np.concatenate([
            ticker_embedding,
            np.zeros(4, dtype=np.float32)
        ])
        
        # Convert to tensors efficiently
        node_features = torch.from_numpy(ticker_features[np.newaxis, :]).float()  # (1, dim+4)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'num_nodes': 1,
            'num_edges': 0,
            'ticker': ticker,
            'date': current_date
        }
        
    def _build_star_structure(
        self,
        triples: List[Dict],
        ticker: str,
        current_date: datetime
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[float]]:
        """
        Build star topology with efficient numpy arrays.
        
        Returns:
            (list of node arrays, edge list, edge weights)
        """
        nodes = []
        edges = []
        edge_weights = []
        
        # Node 0: Center ticker
        ticker_embedding = self.ticker_embeddings.get(
            ticker, np.zeros(self.embedding_dim, dtype=np.float32)
        )
        ticker_features = np.concatenate([
            ticker_embedding,
            np.zeros(4, dtype=np.float32)
        ])
        nodes.append(ticker_features)
        
        # Nodes 1+: Event nodes
        for i, triple in enumerate(triples):
            event_node = self._create_event_node(triple, current_date)
            nodes.append(event_node)
            
            # Bidirectional edges
            edges.append((0, i+1))
            edges.append((i+1, 0))
            
            decay = compute_temporal_decay(
                triple['date'], current_date, triple['relation']
            )
            edge_weights.append(decay)
            edge_weights.append(decay)
        
        return nodes, edges, edge_weights
    
    def _create_event_node(
        self,
        triple: Dict,
        current_date: datetime
    ) -> np.ndarray:
        """
        Create event node features.
        
        Returns:
            np.ndarray of shape (embedding_dim + 4,)
        """
        relation = triple['relation']
        
        # Relation embedding
        rel_embedding = self.relation_embeddings.get(
            relation, np.zeros(self.embedding_dim, dtype=np.float32)
        )
        
        # Numerical features
        magnitude = triple.get('magnitude', 0.0)
        if magnitude and abs(magnitude) > 1.0:
            magnitude = np.log1p(abs(magnitude)) * np.sign(magnitude)
            magnitude = np.clip(magnitude, -5, 5) / 5.0
        
        polarity = float(triple.get('polarity', 0))
        certainty = float(triple.get('certainty', 1.0))
        decay = compute_temporal_decay(triple['date'], current_date, relation)
        
        # Concatenate
        features = np.concatenate([
            rel_embedding,
            np.array([magnitude, polarity, certainty, decay], dtype=np.float32)
        ])
        
        return features
    
    def build_batch(
        self,
        triples_by_date: Dict[datetime, List[Dict]],
        ticker: str,
        dates: List[datetime]
    ) -> List[Dict]:
        """Build graphs for multiple dates."""
        graphs = []
        
        for date in dates:
            day_triples = triples_by_date.get(date, [])
            ticker_triples = [
                t for t in day_triples
                if t.get('subject') == ticker
            ]
            
            graph = self.build_graph(ticker_triples, ticker, date)
            graphs.append(graph)
        
        return graphs
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        stats = self.stats.copy()
        if stats['graphs_built'] > 0:
            stats['avg_nodes_per_graph'] = (
                stats['total_nodes'] / stats['graphs_built']
            )
            stats['empty_rate'] = (
                stats['empty_graphs'] / stats['graphs_built']
            )
        return stats
    
    def print_stats(self):
        """Print statistics."""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("🕸️  GRAPH CONSTRUCTION STATISTICS")
        print("="*50)
        print(f"Graphs built:          {stats['graphs_built']:,}")
        print(f"Total nodes:           {stats['total_nodes']:,}")
        print(f"Total edges:           {stats['total_edges']:,}")
        print(f"Avg nodes per graph:   {stats['avg_nodes_per_graph']:.1f}")
        print(f"Empty graphs:          {stats['empty_graphs']:,} ({stats.get('empty_rate', 0)*100:.1f}%)")
        if stats['failed_builds'] > 0:
            print(f"Failed builds:         {stats['failed_builds']:,}")
        print("="*50 + "\n")