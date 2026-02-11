"""
Main Knowledge Graph Module (UPDATED: Bug fixes + Voyage embeddings)
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
from .voyage_embedder import VoyageKGEmbedder, adapt_embeddings_dimension  # NEW


class SimpleKnowledgeGraph:
    """
    Main orchestrator for the simplified KG pipeline.
    
    ENHANCEMENTS:
    - Voyage AI embeddings (finance-specific)
    - Robust error handling
    - Better date matching
    """
    
    def __init__(
        self,
        tickers: List[str],
        cache_dir: str = './data/interim/kg_cache',
        use_llm: bool = True,
        use_voyage: bool = True  # NEW: Enable Voyage embeddings
    ):
        """
        Initialize KG module.
        
        Args:
            tickers: List of ticker symbols to process
            cache_dir: Directory for caching intermediate results
            use_llm: Whether to use LLM fallback
            use_voyage: Use Voyage AI for embeddings (vs random)
        """
        self.tickers = tickers
        self.cache_dir = cache_dir
        self.use_voyage = use_voyage
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize components
        self.extractor = TripleExtractor(use_llm=use_llm)
        self.graph_builder = None  # Initialized in setup()
        
        # Initialize Voyage embedder if enabled
        if self.use_voyage:
            self.voyage_embedder = VoyageKGEmbedder(cache_dir=cache_dir)
        
        # Data
        self.ticker_embeddings = {}
        self.all_triples = {}
        
        print(f"\n🔧 Initialized SimpleKnowledgeGraph")
        print(f"   Tickers: {', '.join(tickers)}")
        print(f"   Cache dir: {cache_dir}")
        print(f"   Voyage embeddings: {'✓' if use_voyage else '✗'}")
    
    def setup(self, force_rebuild: bool = False):
        """
        One-time setup: Create ticker embeddings and initialize builder.
        
        FIXED: Actually save the embeddings returned by _build_ticker_embeddings()
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
            
            # [NEW] Verify completeness
            missing = [t for t in self.tickers if t not in self.ticker_embeddings]
            if missing:
                print(f"   ⚠️  Cache incomplete: missing {missing}")
                print(f"   🔄 Rebuilding...")
                force_rebuild = True  # Fall through to rebuild
        
        if not os.path.exists(cache_path) or force_rebuild:
            print("🔨 Building ticker embeddings...")
            # [FIX] Actually assign the returned embeddings!
            self.ticker_embeddings = self._build_ticker_embeddings(force_rebuild)
            
            # Save cache
            with open(cache_path, 'wb') as f:
                pickle.dump(self.ticker_embeddings, f)
            print(f"   ✓ Cached to {cache_path}")
        
        # Verify embeddings loaded
        if not self.ticker_embeddings:
            raise ValueError("Failed to load or create ticker embeddings!")
        
        print(f"   ✓ {len(self.ticker_embeddings)} ticker embeddings ready")
        
        # Initialize graph builder
        print("🏗️  Initializing graph builder...")
        self.graph_builder = StarGraphBuilder(
            self.ticker_embeddings,
            use_voyage=self.use_voyage,
            voyage_embedder=self.voyage_embedder if self.use_voyage else None
        )
        print("   ✓ Graph builder ready")
        
        print("="*60)
        print("✅ Setup complete!\n")

    def _build_ticker_embeddings(self, force_rebuild: bool = False) -> Dict[str, np.ndarray]:
        """
        Build ticker embeddings.
        
        FIXED: Use consistent cache path with voyage_embedder
        """
        # Check cache first (load from simple path for backward compatibility)
        cache_path = os.path.join(self.cache_dir, 'ticker_embeddings.pkl')
        
        if os.path.exists(cache_path) and not force_rebuild:
            print("📦 Loading cached ticker embeddings...")
            with open(cache_path, 'rb') as f:
                cached_embeddings = pickle.load(f)
            
            # Verify completeness
            missing = [t for t in self.tickers if t not in cached_embeddings]
            
            if not missing:
                print(f"   ✓ Loaded {len(cached_embeddings)} embeddings")
                return cached_embeddings
            else:
                print(f"   ⚠️  Cache incomplete: missing {missing}")
                print(f"   🔄 Rebuilding all embeddings...")
                # Fall through to rebuild
        
        # Generate new embeddings
        if self.use_voyage:
            print("   Using Voyage AI (finance model)...")
            
            # Generate via Voyage (this uses its own cache internally)
            embeddings = self.voyage_embedder.generate_ticker_embeddings(
                self.tickers,
                force_rebuild=force_rebuild
            )
            
            # Verify all tickers present
            assert len(embeddings) == len(self.tickers), \
                f"Embedding mismatch! Got {len(embeddings)}, expected {len(self.tickers)}"
            
            print(f"   ✓ All {len(self.tickers)} tickers found in cache")
            
        else:
            print("   Using random initialization...")
            embeddings = {}
            for ticker in self.tickers:
                embeddings[ticker] = np.random.randn(768).astype(np.float32) * 0.01
        
        # Save to unified cache
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"   ✓ Cached to {cache_path}")
        
        return embeddings
    
    # ============================================
    # FIXED: Load Pre-Extracted Triples
    # ============================================
    
    def process_from_precomputed_triples(
        self,
        triples_parquet_path: str,
        save_graphs: bool = True
    ) -> Dict:
        """
        Process graphs from pre-extracted triples.
        
        FIXES:
        - Robust numpy array handling
        - Better error recovery
        - Progress tracking
        """
        print("\n" + "="*60)
        print("⚙️  PROCESSING FROM PRE-EXTRACTED TRIPLES")
        print("="*60)
        
        # Load parquet
        print(f"\n📂 Loading triples from {triples_parquet_path}...")
        try:
            df = pd.read_parquet(triples_parquet_path)
            print(f"   ✓ Loaded {len(df)} records")
        except Exception as e:
            print(f"   ❌ Error loading parquet: {e}")
            raise
        
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
        
        FIXES:
        - Robust numpy array handling
        - Type coercion safety
        - Error recovery per row
        """
        from .utils import (
            classify_relation, extract_magnitude,
            determine_polarity, determine_certainty
        )
        
        triples_by_date = {}
        failed_rows = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting triples"):
            try:
                # Parse date
                date_obj = pd.to_datetime(row['date']).to_pydatetime()
                
                # Parse ticker
                ticker = str(row['equity']).strip()
                
                # Skip if ticker not in our target list
                if ticker not in self.tickers:
                    continue
                
                # [FIX] Parse triples with robust type handling
                raw_triples = row['triples_day_flat']
                
                # Handle different storage types
                if isinstance(raw_triples, np.ndarray):
                    raw_triples = raw_triples.tolist()
                elif isinstance(raw_triples, str):
                    # If stored as string, try to eval (risky but fallback)
                    import ast
                    try:
                        raw_triples = ast.literal_eval(raw_triples)
                    except:
                        raw_triples = []
                
                # Ensure it's a list
                if not isinstance(raw_triples, list):
                    continue
                
                # Check if list is empty
                if not raw_triples:
                    continue

                if date_obj not in triples_by_date:
                    triples_by_date[date_obj] = []
                
                # Convert each raw triple
                for raw_triple in raw_triples:
                    try:
                        # [FIX] Handle individual triple safely
                        # 1. Convert various types to list
                        if isinstance(raw_triple, (np.ndarray, tuple)):
                            raw_triple = list(raw_triple)
                        
                        # 2. Validate structure
                        if not isinstance(raw_triple, list) or len(raw_triple) < 3:
                            continue
                        
                        # 3. Extract components safely with type coercion
                        subject_raw = str(raw_triple[0]) if raw_triple[0] is not None else ""
                        predicate_raw = str(raw_triple[1]) if raw_triple[1] is not None else ""
                        object_raw = str(raw_triple[2]) if raw_triple[2] is not None else ""
                        
                        # Skip if any component is empty
                        if not (subject_raw and predicate_raw and object_raw):
                            continue
                        
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
                            'object': full_text[:100],  # Truncate
                            'magnitude': magnitude,
                            'polarity': polarity,
                            'certainty': certainty,
                            'date': date_obj,
                            'source': None
                        }
                        
                        triples_by_date[date_obj].append(triple)
                        
                    except Exception as e:
                        # Skip individual triple if it fails
                        continue
                
            except Exception as e:
                failed_rows += 1
                continue
        
        if failed_rows > 0:
            print(f"   ⚠️  Skipped {failed_rows} rows due to errors")
        
        return triples_by_date
    
    # ... (rest of methods unchanged but add similar error handling)
    
    def _build_all_graphs(self, triples_by_date: Dict) -> Dict:
        """Build graphs from triples for all dates and tickers."""
        graphs_by_date = {}
        failed_graphs = 0
        
        dates = sorted(triples_by_date.keys())
        
        for date in tqdm(dates, desc="Building graphs"):
            day_triples = triples_by_date[date]
            
            graphs_by_date[date] = {}
            
            for ticker in self.tickers:
                try:
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
                    
                except Exception as e:
                    # Create empty graph on failure
                    failed_graphs += 1
                    graphs_by_date[date][ticker] = self._create_empty_graph(ticker, date)
        
        if failed_graphs > 0:
            print(f"   ⚠️  {failed_graphs} graphs failed, created empty fallbacks")
        
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
    
    def _create_empty_graph(self, ticker: str, date: datetime) -> Dict:
        """
        Create an empty graph (only ticker node, no events).
        
        Use full 1024-dim + 4 padding = 1028-dim
        """
        ticker_embedding = self.ticker_embeddings.get(
            ticker, np.zeros(1024, dtype=np.float32)
        )
        
        # Single node: ticker with padding to match event node size (772)
        ticker_features = np.concatenate([
            ticker_embedding,  # 1024-dim
            np.zeros(4, dtype=np.float32)  # 4-dim padding
        ])
        
        node_features = torch.tensor([ticker_features], dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
        
        return {
            'node_features': node_features,  # 1028
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'num_nodes': 1,
            'num_edges': 0,
            'ticker': ticker,
            'date': date
        }
    
    # ... (other methods with similar error handling added)