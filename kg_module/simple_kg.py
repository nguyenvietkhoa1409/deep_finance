"""
Main Knowledge Graph Module (UPDATED: Support precomputed triples)
"""

import os
import pickle
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

from .triple_extractor import TripleExtractor
from .graph_builder import StarGraphBuilder
from .utils import normalize_features


class SimpleKnowledgeGraph:
    """
    Main orchestrator for the simplified KG pipeline.
    
    UPDATED: Support loading pre-extracted triples from parquet file.
    
    Usage Option A (Extract triples):
        >>> kg = SimpleKnowledgeGraph(tickers=['AMZN', 'TSLA'])
        >>> kg.setup()
        >>> kg_data = kg.process_news_data(unified_dataset)
    
    Usage Option B (Use precomputed triples):
        >>> kg = SimpleKnowledgeGraph(tickers=['AMZN', 'TSLA'])
        >>> kg.setup()
        >>> kg_data = kg.process_from_precomputed_triples('kg_triples_day_level.parquet')
    """
    
    def __init__(
        self,
        tickers: List[str],
        cache_dir: str = './data/interim/kg_cache',
        use_llm: bool = True
    ):
        """
        Initialize KG module.
        
        Args:
            tickers: List of ticker symbols to process
            cache_dir: Directory for caching intermediate results
            use_llm: Whether to use LLM fallback (only relevant for new extraction)
        """
        self.tickers = tickers
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize components
        self.extractor = TripleExtractor(use_llm=use_llm)
        self.graph_builder = None  # Initialized in setup()
        
        # Data
        self.ticker_embeddings = {}
        self.all_triples = {}
        
        print(f"\n🔧 Initialized SimpleKnowledgeGraph")
        print(f"   Tickers: {', '.join(tickers)}")
        print(f"   Cache dir: {cache_dir}")
    
    def setup(self, force_rebuild: bool = False):
        """
        One-time setup: Create ticker embeddings and initialize builder.
        
        Args:
            force_rebuild: If True, rebuild even if cache exists
        """
        print("\n" + "="*60)
        print("🚀 KNOWLEDGE GRAPH SETUP")
        print("="*60)
        
        # Check cache
        cache_path = os.path.join(self.cache_dir, 'ticker_embeddings.pkl')
        
        if os.path.exists(cache_path) and not force_rebuild:
            print("📦 Loading cached ticker embeddings...")
            with open(cache_path, 'rb') as f:
                self.ticker_embeddings = pickle.load(f)
            print(f"   ✓ Loaded {len(self.ticker_embeddings)} embeddings")
        else:
            print("🔨 Building ticker embeddings...")
            self._build_ticker_embeddings()
            
            # Save cache
            with open(cache_path, 'wb') as f:
                pickle.dump(self.ticker_embeddings, f)
            print(f"   ✓ Cached to {cache_path}")
        
        # Initialize graph builder
        print("🏗️  Initializing graph builder...")
        self.graph_builder = StarGraphBuilder(self.ticker_embeddings)
        print("   ✓ Graph builder ready")
        
        print("="*60)
        print("✅ Setup complete!\n")
    
    def _build_ticker_embeddings(self):
        """Create embeddings for each ticker."""
        for ticker in tqdm(self.tickers, desc="Creating embeddings"):
            # Random initialization (will be learned during training)
            embedding = np.random.randn(768).astype(np.float32) * 0.01
            self.ticker_embeddings[ticker] = embedding
    
    # ============================================
    # NEW: Load Pre-Extracted Triples
    # ============================================
    
    def process_from_precomputed_triples(
        self,
        triples_parquet_path: str,
        save_graphs: bool = True
    ) -> Dict:
        """
        NEW METHOD: Process graphs from pre-extracted triples.
        
        This method SKIPS the triple extraction stage and directly loads
        triples from a parquet file, then builds graphs.
        
        Args:
            triples_parquet_path: Path to parquet file with columns:
                - date: datetime/str
                - equity: str (ticker)
                - triples_day_flat: list of [subject, predicate, object] lists
            save_graphs: Whether to save built graphs
        
        Returns:
            Dict mapping date → ticker → graph_dict
        
        Parquet Schema Expected:
            date           | equity | triples_day_flat
            ---------------|--------|------------------
            2024-05-01     | AMZN   | [[subj, pred, obj], [subj2, pred2, obj2], ...]
            2024-05-01     | MSFT   | [[subj, pred, obj], ...]
            2024-05-02     | AMZN   | [[subj, pred, obj], ...]
        
        Example:
            >>> kg = SimpleKnowledgeGraph(tickers=['AMZN', 'TSLA'])
            >>> kg.setup()
            >>> graphs = kg.process_from_precomputed_triples('kg_triples_day_level.parquet')
        """
        print("\n" + "="*60)
        print("⚙️  PROCESSING FROM PRE-EXTRACTED TRIPLES")
        print("="*60)
        
        # Load parquet
        print(f"\n📂 Loading triples from {triples_parquet_path}...")
        df = pd.read_parquet(triples_parquet_path)
        print(f"   ✓ Loaded {len(df)} records")
        
        # Validate columns
        required_cols = ['date', 'equity', 'triples_day_flat']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert to triples_by_date format
        print("\n🔄 Converting triples to internal format...")
        triples_by_date = self._convert_parquet_to_triples_dict(df)
        print(f"   ✓ Processed {len(triples_by_date)} dates")
        
        # Save converted triples (optional, for inspection)
        triples_path = os.path.join(self.cache_dir, 'converted_triples.pkl')
        with open(triples_path, 'wb') as f:
            pickle.dump(triples_by_date, f)
        print(f"   💾 Converted triples saved to {triples_path}")
        
        # Build graphs
        print("\n🕸️  Stage 2: Graph Construction")
        graphs_by_date = self._build_all_graphs(triples_by_date)
        
        # Print statistics
        self.graph_builder.print_stats()
        
        # Save graphs
        if save_graphs:
            output_filename = "kg_graphs.pkl"
            self.save(graphs_by_date, output_filename)
        
        print("="*60)
        print("✅ Processing complete!\n")
        
        return graphs_by_date
    
    def _convert_parquet_to_triples_dict(self, df: pd.DataFrame) -> Dict:
        """
        Convert parquet dataframe to internal triples format.
        
        Input format (parquet):
            date | equity | triples_day_flat
            -----|--------|------------------
            2024-05-01 | AMZN | [["Amazon", "expects", "higher capex"], [...]]
        
        Output format (dict):
            {
                datetime(2024, 5, 1): [
                    {
                        'subject': 'AMZN',
                        'relation': 'revenue_change',  # Classified
                        'object': 'higher capex',
                        'magnitude': None,
                        'polarity': 1,
                        'certainty': 0.8,
                        'date': datetime(...),
                        'source': None
                    },
                    ...
                ]
            }
        """
        from .utils import (
            classify_relation, extract_magnitude,
            determine_polarity, determine_certainty
        )
        
        triples_by_date = {}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting triples"):
            # Parse date
            date_obj = pd.to_datetime(row['date']).to_pydatetime()
            
            # Parse ticker
            ticker = row['equity']
            
            # Skip if ticker not in our target list
            if ticker not in self.tickers:
                continue
            
            # Parse triples
            # [CRITICAL FIX]: Ensure raw_triples is a Python list
            raw_triples = row['triples_day_flat']
            if isinstance(raw_triples, np.ndarray):
                raw_triples = raw_triples.tolist()
            
            # Check if list is empty or None
            if not raw_triples:
                continue

            if date_obj not in triples_by_date:
                triples_by_date[date_obj] = []
            
            # Convert each raw triple
            for raw_triple in raw_triples:
                # [CRITICAL FIX]: Handle individual triple safely
                # 1. Convert numpy array/tuple to list
                if isinstance(raw_triple, (np.ndarray, tuple)):
                    raw_triple = list(raw_triple)
                
                # 2. Check length safely
                if not isinstance(raw_triple, list) or len(raw_triple) < 3:
                    continue
                
                # Extract components safely (ensure string format)
                subject_raw = str(raw_triple[0])
                predicate_raw = str(raw_triple[1])
                object_raw = str(raw_triple[2])
                
                # Build full text for analysis
                full_text = f"{subject_raw} {predicate_raw} {object_raw}"
                
                # Classify relation
                relation = classify_relation(full_text)
                
                # Extract components
                magnitude = extract_magnitude(full_text)
                polarity = determine_polarity(full_text)
                certainty = determine_certainty(full_text)
                
                # Build triple dict
                triple = {
                    'subject': ticker,  # Use canonical ticker
                    'relation': relation,
                    'object': full_text[:100], # Store truncated full text as object/desc
                    'magnitude': magnitude,
                    'polarity': polarity,
                    'certainty': certainty,
                    'date': date_obj,
                    'source': None
                }
                
                triples_by_date[date_obj].append(triple)
        
        return triples_by_date
    
    # ============================================
    # Original Methods (Keep for backward compatibility)
    # ============================================
    
    def process_news_data(
        self,
        unified_dataset: Dict,
        save_triples: bool = True
    ) -> Dict:
        """
        Original method: Extract triples from news then build graphs.
        
        Use this if you DON'T have pre-extracted triples.
        Use process_from_precomputed_triples() if you DO have triples.
        """
        print("\n" + "="*60)
        print("⚙️  PROCESSING NEWS DATA TO KNOWLEDGE GRAPHS")
        print("="*60)
        
        # Stage 1: Extract Triples
        print("\n📝 Stage 1: Triple Extraction")
        triples_by_date = self._extract_all_triples(unified_dataset)
        
        if save_triples:
            triples_path = os.path.join(self.cache_dir, 'extracted_triples.pkl')
            with open(triples_path, 'wb') as f:
                pickle.dump(triples_by_date, f)
            print(f"   💾 Triples saved to {triples_path}")
        
        # Stage 2: Build Graphs
        print("\n🕸️  Stage 2: Graph Construction")
        graphs_by_date = self._build_all_graphs(triples_by_date)
        
        # Print statistics
        self.extractor.print_stats()
        self.graph_builder.print_stats()
        
        print("="*60)
        print("✅ Processing complete!\n")
        
        return graphs_by_date
    
    def _extract_all_triples(self, unified_dataset: Dict) -> Dict:
        """Extract triples from all news data."""
        triples_by_date = {}
        
        dates = sorted(unified_dataset.keys())
        
        for date_str in tqdm(dates, desc="Extracting triples"):
            day_data = unified_dataset[date_str]
            news_data = day_data.get('news', {})
            
            # Parse date
            if isinstance(date_str, str):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            else:
                date_obj = date_str
            
            day_triples = []
            
            # Process each ticker's news
            for ticker in self.tickers:
                if ticker not in news_data:
                    continue
                
                news_records = news_data[ticker]
                
                for record in news_records:
                    headline = record.get('title', '')
                    if not headline:
                        continue
                    
                    triple = self.extractor.extract(
                        headline=headline,
                        ticker=ticker,
                        date=date_obj,
                        source=record.get('source')
                    )
                    
                    if triple:
                        day_triples.append(triple)
            
            triples_by_date[date_obj] = day_triples
        
        return triples_by_date
    
    def _build_all_graphs(self, triples_by_date: Dict) -> Dict:
        """Build graphs from triples for all dates and tickers."""
        graphs_by_date = {}
        
        dates = sorted(triples_by_date.keys())
        
        for date in tqdm(dates, desc="Building graphs"):
            day_triples = triples_by_date[date]
            
            graphs_by_date[date] = {}
            
            for ticker in self.tickers:
                # Filter triples for this ticker
                ticker_triples = [
                    t for t in day_triples
                    if t.get('subject') == ticker
                ]
                
                # Build graph
                graph = self.graph_builder.build_graph(
                    ticker_triples, ticker, date
                )
                
                graphs_by_date[date][ticker] = graph
        
        return graphs_by_date
    
    def save(self, graphs_by_date: Dict, filename: str):
        """Save processed graphs to disk."""
        output_path = os.path.join(self.cache_dir, filename)
        
        with open(output_path, 'wb') as f:
            pickle.dump(graphs_by_date, f)
        
        print(f"\n💾 Saved KG data to: {output_path}")
        
        # Print summary
        total_graphs = sum(len(g) for g in graphs_by_date.values())
        print(f"   📊 Total graphs: {total_graphs:,}")
        print(f"   📅 Date range: {min(graphs_by_date.keys())} to {max(graphs_by_date.keys())}")
    
    def load(self, filename: str) -> Dict:
        """Load processed graphs from disk."""
        input_path = os.path.join(self.cache_dir, filename)
        
        with open(input_path, 'rb') as f:
            graphs_by_date = pickle.load(f)
        
        print(f"\n📦 Loaded KG data from: {input_path}")
        return graphs_by_date
    
    def get_graph_sequence(
        self,
        graphs_by_date: Dict,
        ticker: str,
        start_date: datetime,
        window_size: int
    ) -> List[Dict]:
        """Extract a sequence of graphs for windowing."""
        from datetime import timedelta
        
        sequence = []
        
        for i in range(window_size):
            date = start_date + timedelta(days=i)
            
            if date in graphs_by_date and ticker in graphs_by_date[date]:
                graph = graphs_by_date[date][ticker]
            else:
                # Create empty graph for missing dates
                graph = self._create_empty_graph(ticker, date)
            
            sequence.append(graph)
        
        return sequence
    
    def _create_empty_graph(self, ticker: str, date: datetime) -> Dict:
        """Create an empty graph (only ticker node, no events)."""
        ticker_embedding = self.ticker_embeddings.get(
            ticker, np.zeros(768, dtype=np.float32)
        )
        
        # Single node: ticker with padding
        ticker_features = np.concatenate([
            ticker_embedding,
            np.zeros(4, dtype=np.float32)
        ])
        
        node_features = torch.tensor([ticker_features], dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'num_nodes': 1,
            'num_edges': 0,
            'ticker': ticker,
            'date': date
        }