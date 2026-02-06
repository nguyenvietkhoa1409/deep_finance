"""
Stage 2: Star Graph Construction
Build ticker-centric graphs from extracted triples
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

from .config import (
    RELATION_TYPES, MAX_EVENTS_PER_GRAPH, 
    MIN_CERTAINTY_THRESHOLD
)
from .utils import compute_temporal_decay


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
    
    def __init__(self, ticker_embeddings: Dict[str, np.ndarray]):
        """
        Initialize graph builder.
        
        Args:
            ticker_embeddings: Dict mapping ticker → embedding vector
                Example: {'AMZN': array([0.1, 0.2, ..., 0.5])}  # shape (768,)
        """
        self.ticker_embeddings = ticker_embeddings
        
        # Initialize relation embeddings (learnable or pre-computed)
        self.relation_embeddings = self._init_relation_embeddings()
        
        # Statistics
        self.stats = {
            'graphs_built': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'avg_nodes_per_graph': 0,
            'empty_graphs': 0
        }
    
    def _init_relation_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Initialize embeddings for each relation type.
        
        Strategy:
            - Option A: Random initialization (will be learned during training)
            - Option B: Use FinBERT to encode relation keywords (better cold start)
        
        Returns:
            Dict mapping relation_type → embedding (768,)
        """
        embeddings = {}
        
        for rel_type, config in RELATION_TYPES.items():
            # Option A: Random (simple, works well with end-to-end training)
            embeddings[rel_type] = np.random.randn(768).astype(np.float32) * 0.01
            
            # TODO (Optional): Option B - Use sentence transformer
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('all-MiniLM-L6-v2')
            # text = ' '.join(config['keywords'])
            # embeddings[rel_type] = model.encode(text)
        
        return embeddings
    
    def build_graph(
        self,
        triples: List[Dict],
        ticker: str,
        current_date: datetime
    ) -> Dict:
        """
        Build a single star graph from triples for a given ticker and date.
        
        Args:
            triples: List of triple dicts (from TripleExtractor)
            ticker: Center ticker
            current_date: Current date (for temporal decay)
        
        Returns:
            Graph dict with structure:
                {
                    'node_features': torch.Tensor (N, 772),
                    'edge_index': torch.Tensor (2, E),
                    'edge_weight': torch.Tensor (E,),
                    'num_nodes': int,
                    'num_edges': int,
                    'ticker': str,
                    'date': datetime
                }
        
        Examples:
            >>> triples = [
            ...     {'subject': 'AMZN', 'relation': 'revenue_change', ...},
            ...     {'subject': 'AMZN', 'relation': 'guidance_update', ...}
            ... ]
            >>> graph = builder.build_graph(triples, 'AMZN', datetime.now())
            >>> graph['num_nodes']
            3  # 1 ticker + 2 events
        """
        # Filter valid triples
        valid_triples = [
            t for t in triples
            if t.get('certainty', 0) >= MIN_CERTAINTY_THRESHOLD
            and t.get('relation') in RELATION_TYPES
        ]
        
        # Limit number of events
        if len(valid_triples) > MAX_EVENTS_PER_GRAPH:
            # Sort by certainty × decay, keep top-k
            scored_triples = []
            for t in valid_triples:
                decay = compute_temporal_decay(
                    t['date'], current_date, t['relation']
                )
                score = t['certainty'] * decay
                scored_triples.append((score, t))
            
            scored_triples.sort(reverse=True, key=lambda x: x[0])
            valid_triples = [t for _, t in scored_triples[:MAX_EVENTS_PER_GRAPH]]
        
        # Build nodes
        nodes, edges, edge_weights = self._build_star_structure(
            valid_triples, ticker, current_date
        )
        
        # Convert to tensors
        node_features = torch.tensor(nodes, dtype=torch.float32)  # (N, 772)
        edge_index = torch.tensor(edges, dtype=torch.long).T  # (2, E)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)  # (E,)
        
        # Update stats
        self.stats['graphs_built'] += 1
        self.stats['total_nodes'] += len(nodes)
        self.stats['total_edges'] += len(edges)
        if len(nodes) == 1:  # Only ticker node
            self.stats['empty_graphs'] += 1
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'num_nodes': len(nodes),
            'num_edges': len(edges),
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
        Build star topology: ticker at center, events around it.
        
        Returns:
            (node_features, edge_list, edge_weights)
        """
        nodes = []
        edges = []
        edge_weights = []
        
        # Node 0: Center ticker
        ticker_embedding = self.ticker_embeddings.get(
            ticker, np.zeros(768, dtype=np.float32)
        )
        # Add 4 padding zeros to match event node size (772)
        ticker_features = np.concatenate([
            ticker_embedding,
            np.zeros(4, dtype=np.float32)
        ])
        nodes.append(ticker_features)
        
        # Nodes 1+: Event nodes
        for i, triple in enumerate(triples):
            event_node = self._create_event_node(triple, current_date)
            nodes.append(event_node)
            
            # Edge: ticker (0) ↔ event (i+1)
            edges.append((0, i+1))  # Directed: center → event
            edges.append((i+1, 0))  # Bidirectional
            
            # Edge weight: temporal decay
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
        Create feature vector for an event node.
        
        Returns:
            Feature vector of shape (772,)
                - [0:768]: Relation embedding
                - [768]: Magnitude (normalized)
                - [769]: Polarity {-1, 0, 1}
                - [770]: Certainty [0, 1]
                - [771]: Temporal decay [0, 1]
        """
        relation = triple['relation']
        
        # Relation embedding (768-dim)
        rel_embedding = self.relation_embeddings.get(
            relation, np.zeros(768, dtype=np.float32)
        )
        
        # Numerical features
        magnitude = triple.get('magnitude', 0.0)
        if magnitude and abs(magnitude) > 1.0:
            # Normalize large magnitudes (e.g., billions)
            magnitude = np.log1p(abs(magnitude)) * np.sign(magnitude)
            magnitude = np.clip(magnitude, -5, 5) / 5.0  # Scale to [-1, 1]
        
        polarity = float(triple.get('polarity', 0))
        certainty = float(triple.get('certainty', 1.0))
        
        decay = compute_temporal_decay(
            triple['date'], current_date, relation
        )
        
        # Concatenate
        features = np.concatenate([
            rel_embedding,
            [magnitude, polarity, certainty, decay]
        ]).astype(np.float32)
        
        return features
    
    def build_batch(
        self,
        triples_by_date: Dict[datetime, List[Dict]],
        ticker: str,
        dates: List[datetime]
    ) -> List[Dict]:
        """
        Build graphs for multiple dates (for windowing).
        
        Args:
            triples_by_date: Dict mapping date → triples for that date
            ticker: Ticker symbol
            dates: Ordered list of dates to build graphs for
        
        Returns:
            List of graph dicts (one per date)
        
        Examples:
            >>> triples = {
            ...     datetime(2024, 1, 1): [triple1, triple2],
            ...     datetime(2024, 1, 2): [triple3]
            ... }
            >>> graphs = builder.build_batch(triples, 'AMZN', [date1, date2])
            >>> len(graphs)
            2
        """
        graphs = []
        
        for date in dates:
            # Get triples for this date (empty list if none)
            day_triples = triples_by_date.get(date, [])
            
            # Filter to this ticker
            ticker_triples = [
                t for t in day_triples
                if t.get('subject') == ticker
            ]
            
            # Build graph
            graph = self.build_graph(ticker_triples, ticker, date)
            graphs.append(graph)
        
        return graphs
    
    def get_stats(self) -> Dict:
        """Get graph construction statistics."""
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
        """Print graph construction statistics."""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("🕸️  GRAPH CONSTRUCTION STATISTICS")
        print("="*50)
        print(f"Graphs built:          {stats['graphs_built']:,}")
        print(f"Total nodes:           {stats['total_nodes']:,}")
        print(f"Total edges:           {stats['total_edges']:,}")
        print(f"Avg nodes per graph:   {stats['avg_nodes_per_graph']:.1f}")
        print(f"Empty graphs:          {stats['empty_graphs']:,} ({stats.get('empty_rate', 0)*100:.1f}%)")
        print("="*50 + "\n")