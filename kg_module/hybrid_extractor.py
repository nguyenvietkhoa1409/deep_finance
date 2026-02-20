"""
Hybrid Triple Extractor - Dual Path Architecture
Combines structured features + dense semantic embeddings
[OPTIMIZED] Fixed cache write bottleneck + parallel processing
"""

import numpy as np
import voyageai
import pickle
import os
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .config import (
    RELATION_TYPES, POLARITY_KEYWORDS, CERTAINTY_KEYWORDS,
    MAGNITUDE_PATTERNS
)
from .utils import (
    extract_magnitude, determine_polarity, 
    determine_certainty, classify_relation
)
from configs.config import GlobalConfig


class HybridTripleExtractor:
    """
    Enhanced triple processing with dual-path architecture.
    
    Path A (Structured): Relation embedding + numerical features
    Path B (Dense Semantic): Document context + triple semantics
    
    Output: Hybrid event node features (1037 + 768 = 1805-dim)
    
    [OPTIMIZED] 
    - Loads entire cache to memory (no incremental file I/O)
    - Parallel batch processing with ThreadPoolExecutor
    - Single cache write at end
    """
    
    def __init__(
        self,
        cache_dir: str = './data/interim/kg_cache',
        use_voyage: bool = True
    ):
        """
        Initialize hybrid extractor.
        
        Args:
            cache_dir: Directory for caching embeddings
            use_voyage: Use Voyage AI (True) or random init (False)
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.use_voyage = use_voyage
        
        if use_voyage:
            self.client = voyageai.Client(api_key=GlobalConfig.VOYAGE_API_KEY)
            self.model = 'voyage-finance-2'
            
            # Rate limiting
            limits = GlobalConfig.VOYAGE_RATE_LIMITS[GlobalConfig.PAYMENT_ADDED]
            self.sleep_time = limits["SLEEP"]
        
        # Cache paths
        self.relation_embed_cache = os.path.join(cache_dir, 'relation_embeddings.pkl')
        self.doc_embed_cache = os.path.join(cache_dir, 'document_embeddings.pkl')
        self.triple_embed_cache = os.path.join(cache_dir, 'triple_embeddings.pkl')
        
        # Load or create relation embeddings
        self.relation_embeddings = self._load_or_create_relation_embeddings()
        
        # Statistics
        self.stats = {
            'triples_processed': 0,
            'voyage_calls': 0,
            'cache_hits': 0
        }
    
    def _load_or_create_relation_embeddings(self) -> Dict[str, np.ndarray]:
        """Load cached relation embeddings or create new ones."""
        if os.path.exists(self.relation_embed_cache):
            print("📦 Loading cached relation embeddings...")
            with open(self.relation_embed_cache, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"   ✓ Loaded {len(embeddings)} relation types")
            return embeddings
        
        print("🔨 Creating relation embeddings...")
        embeddings = {}
        
        if self.use_voyage:
            # Generate via Voyage
            texts = []
            rel_types = []
            
            for rel_name, rel_config in RELATION_TYPES.items():
                keywords = rel_config.get('keywords', [])
                text = f"Financial event: {rel_name.replace('_', ' ')}. Keywords: {', '.join(keywords)}"
                texts.append(text)
                rel_types.append(rel_name)
            
            # Batch embed
            print(f"   Embedding {len(texts)} relation types...")
            result = self.client.embed(
                texts=texts,
                model=self.model,
                input_type='document'
            )
            
            for rel_name, emb in zip(rel_types, result.embeddings):
                embeddings[rel_name] = np.array(emb, dtype=np.float32)
            
            time.sleep(self.sleep_time)
        else:
            # Random initialization
            for rel_name in RELATION_TYPES.keys():
                embeddings[rel_name] = np.random.randn(1024).astype(np.float32) * 0.01
        
        # Cache
        with open(self.relation_embed_cache, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"   ✓ Created and cached {len(embeddings)} embeddings")
        return embeddings
    
    def _infer_meta_features(self, triple: List[str]) -> Dict[str, str]:
        """
        Infer meta features from triple text (rule-based).
        
        Returns:
            Dict with 'time_horizon', 'causality', 'scope'
        """
        subject, predicate, obj = triple
        full_text = f"{subject} {predicate} {obj}".lower()
        
        # Time horizon heuristics
        if any(word in full_text for word in ['expects', 'plans', 'will', 'forecast', 'guidance']):
            time_horizon = 'forward_looking'
        elif any(word in full_text for word in ['achieved', 'reported', 'announced', 'posted']):
            time_horizon = 'immediate'
        else:
            time_horizon = 'current'
        
        # Causality heuristics
        if any(word in predicate.lower() for word in ['caused', 'led to', 'resulted in', 'drove']):
            causality = 'direct'
        elif any(word in full_text for word in ['due to', 'because', 'following']):
            causality = 'indirect'
        else:
            causality = 'correlation'
        
        # Scope heuristics
        if any(word in full_text for word in ['industry', 'sector', 'market', 'global']):
            scope = 'sector_wide'
        elif any(word in full_text for word in ['gdp', 'inflation', 'interest rate', 'unemployment']):
            scope = 'macro'
        else:
            scope = 'company_specific'
        
        return {
            'time_horizon': time_horizon,
            'causality': causality,
            'scope': scope
        }
    
    def _encode_meta_features(self, meta: Dict[str, str]) -> np.ndarray:
        """
        Encode meta features as one-hot vectors.
        
        Returns:
            9-dim vector (3 categories × 3 values)
        """
        # Define vocabularies
        time_vocab = ['immediate', 'current', 'forward_looking']
        causality_vocab = ['direct', 'indirect', 'correlation']
        scope_vocab = ['company_specific', 'sector_wide', 'macro']
        
        # One-hot encode
        time_vec = np.zeros(3)
        causality_vec = np.zeros(3)
        scope_vec = np.zeros(3)
        
        try:
            time_vec[time_vocab.index(meta['time_horizon'])] = 1.0
            causality_vec[causality_vocab.index(meta['causality'])] = 1.0
            scope_vec[scope_vocab.index(meta['scope'])] = 1.0
        except ValueError:
            # Fallback to default
            time_vec[1] = 1.0
            causality_vec[2] = 1.0
            scope_vec[0] = 1.0
        
        return np.concatenate([time_vec, causality_vec, scope_vec])
    
    def extract_structured_features(
        self,
        triple: List[str],
        event_date: datetime,
        current_date: datetime
    ) -> np.ndarray:
        """
        Path A: Extract structured features.
        
        Returns:
            1037-dim vector: [relation_emb(1024), m, p, c, τ, meta(9)]
        """
        subject, predicate, obj = triple
        full_text = f"{subject} {predicate} {obj}"
        
        # 1. Relation embedding (1024-dim)
        relation_type = classify_relation(full_text)
        if relation_type and relation_type in self.relation_embeddings:
            rel_emb = self.relation_embeddings[relation_type]
        else:
            # Fallback: average of all relations
            rel_emb = np.mean(list(self.relation_embeddings.values()), axis=0)
        
        # 2. Numerical features
        magnitude = extract_magnitude(full_text) or 0.0
        # Normalize magnitude (log-scale clipping)
        if magnitude != 0:
            magnitude = np.sign(magnitude) * np.log1p(abs(magnitude))
            magnitude = np.clip(magnitude, -5, 5) / 5.0
        
        polarity = float(determine_polarity(full_text))
        certainty = float(determine_certainty(full_text))
        
        # Temporal decay
        days_elapsed = (current_date - event_date).days
        if relation_type and relation_type in RELATION_TYPES:
            half_life = RELATION_TYPES[relation_type].get('half_life_days', 5)
        else:
            half_life = 5
        decay = float(0.5 ** (days_elapsed / half_life))
        
        # 3. Meta features (9-dim)
        meta = self._infer_meta_features(triple)
        meta_vec = self._encode_meta_features(meta)
        
        # Concatenate
        features = np.concatenate([
            rel_emb,                                    # 1024
            np.array([magnitude, polarity, certainty, decay]),  # 4
            meta_vec                                    # 9
        ])
        
        assert features.shape[0] == 1037, f"Expected 1037-dim, got {features.shape[0]}"
        
        return features.astype(np.float32)
    
    def process_batch(
        self,
        triples_data: List[Dict],
        news_df: Optional[object] = None,
        max_workers: int = 4
    ) -> Dict[datetime, Dict[str, List[np.ndarray]]]:
        """
        [OPTIMIZED] Process batch with parallel embedding + fixed cache bottleneck.
        
        KEY OPTIMIZATIONS:
        1. Load entire cache to memory at start (no incremental file I/O)
        2. Parallel batch processing (4 workers)
        3. Single cache write at end (not per embedding)
        
        Args:
            triples_data: List of triple records
            news_df: Optional news dataframe
            max_workers: Number of parallel threads (default: 4)
        
        Returns:
            Dict: {date: {ticker: [feature_vectors]}}
        """
        print(f"\n🔄 Processing {len(triples_data)} triple records...")
        print(f"   Total triples: {sum(len(r['triples']) for r in triples_data)}")
        print(f"   Parallel workers: {max_workers}")
        
        # === STEP 1: Deduplication ===
        print("\n📊 Step 1: Deduplicating triples...")
        
        unique_triples = {}
        triple_to_records = {}
        
        for rec_idx, record in enumerate(triples_data):
            for tri_idx, triple in enumerate(record['triples']):
                triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
                
                if triple_text not in unique_triples:
                    unique_triples[triple_text] = triple
                    triple_to_records[triple_text] = []
                
                triple_to_records[triple_text].append((rec_idx, tri_idx))
        
        print(f"   Original triples: {sum(len(r['triples']) for r in triples_data)}")
        print(f"   Unique triples: {len(unique_triples)}")
        print(f"   Dedup ratio: {len(unique_triples) / sum(len(r['triples']) for r in triples_data) * 100:.1f}%")
        
        # === STEP 2: Load Cache to Memory ONCE ===
        print("\n💾 Step 2: Loading cache to memory...")
        
        cache_mem = {}
        if os.path.exists(self.triple_embed_cache):
            try:
                with open(self.triple_embed_cache, 'rb') as f:
                    cache_mem = pickle.load(f)
                print(f"   ✓ Loaded {len(cache_mem)} cached embeddings ({len(cache_mem) * 4 / 1024:.1f} MB)")
            except Exception as e:
                print(f"   ⚠️ Cache corrupted: {e}")
                print("   Starting with empty cache")
        else:
            print("   No existing cache found")
        
        triple_embeddings = {}
        embeddings_lock = threading.Lock()
        
        # === STEP 3: Parallel Batch Embedding ===
        print(f"\n🔄 Step 3: Parallel batch embedding (workers={max_workers})...")
        
        if self.use_voyage:
            unique_triple_texts = list(unique_triples.keys())
            batch_size = 40  # Optimal for Voyage API
            
            # Create batches
            batches = []
            for i in range(0, len(unique_triple_texts), batch_size):
                batch_texts = unique_triple_texts[i:i + batch_size]
                batches.append((i // batch_size, batch_texts))
            
            total_batches = len(batches)
            print(f"   Total batches: {total_batches} (batch_size={batch_size})")
            
            # Thread-safe function to embed a single batch
            def embed_single_batch(batch_id, batch_texts):
                """
                Embed a single batch of texts.
                [FIXED] No file I/O - only memory operations
                """
                uncached_texts = []
                local_embeddings = {}
                
                # Check memory cache (no file I/O!)
                for text in batch_texts:
                    cache_key = hash(text)
                    
                    if cache_key in cache_mem:
                        local_embeddings[text] = cache_mem[cache_key]
                        with embeddings_lock:
                            self.stats['cache_hits'] += 1
                    else:
                        uncached_texts.append(text)
                
                # Embed uncached texts
                if uncached_texts:
                    try:
                        result = self.client.embed(
                            texts=uncached_texts,
                            model=self.model,
                            input_type='document'
                        )
                        
                        for text, emb in zip(uncached_texts, result.embeddings):
                            emb_array = np.array(emb, dtype=np.float32)
                            local_embeddings[text] = emb_array
                            
                            # [FIXED] Update memory cache only (no file writes!)
                            cache_key = hash(text)
                            with embeddings_lock:
                                cache_mem[cache_key] = emb_array
                        
                        with embeddings_lock:
                            self.stats['voyage_calls'] += 1
                        
                        # Sleep (reduced for parallel)
                        time.sleep(self.sleep_time / max_workers)
                    
                    except Exception as e:
                        # Fallback: zero embeddings
                        print(f"   ⚠️ Batch {batch_id} failed: {e}")
                        for text in uncached_texts:
                            local_embeddings[text] = np.zeros(1024, dtype=np.float32)
                
                return batch_id, local_embeddings
            
            # Process batches in parallel
            completed_batches = 0
            pbar = tqdm(total=total_batches, desc="Embedding batches")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batch jobs
                future_to_batch = {
                    executor.submit(embed_single_batch, batch_id, batch_texts): batch_id
                    for batch_id, batch_texts in batches
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch_id, local_embeddings = future.result()
                    
                    # Merge into global dictionary (thread-safe)
                    with embeddings_lock:
                        triple_embeddings.update(local_embeddings)
                    
                    completed_batches += 1
                    pbar.update(1)
                    
                    # [OPTIONAL] Periodic checkpoint every 100 batches
                    if completed_batches % 100 == 0:
                        print(f"\n   💾 Checkpoint: {completed_batches}/{total_batches} batches")
            
            pbar.close()
            
            # === STEP 4: Save Cache ONCE at End ===
            print(f"\n💾 Step 4: Saving cache to disk (one-time write)...")
            try:
                start_save = time.time()
                with open(self.triple_embed_cache, 'wb') as f:
                    pickle.dump(cache_mem, f)
                save_time = time.time() - start_save
                cache_size_mb = len(cache_mem) * 4 / 1024
                print(f"   ✓ Saved {len(cache_mem)} embeddings ({cache_size_mb:.1f} MB) in {save_time:.1f}s")
            except Exception as e:
                print(f"   ⚠️ Cache save failed: {e}")
            
            print(f"   ✓ Embedded {len(triple_embeddings)} unique triples")
        
        else:
            # Random init (no API calls)
            print("   Using random embeddings (no Voyage API)")
            for text in unique_triples.keys():
                triple_embeddings[text] = np.random.randn(1024).astype(np.float32) * 0.01
        
        # === STEP 5: Build Hybrid Features ===
        print("\n🔨 Step 5: Building hybrid features...")
        
        results = {}
        
        for record in tqdm(triples_data, desc="Building features"):
            date = record['date']
            if isinstance(date, str):
                date = datetime.fromisoformat(date)
            
            ticker = record['ticker']
            triples = record['triples']
            
            # Document embedding (simplified - no doc context)
            doc_emb = np.zeros(1024, dtype=np.float32)
            
            # Process each triple
            features = []
            
            for triple in triples:
                try:
                    # Path A: Structured features
                    f_struct = self.extract_structured_features(
                        triple,
                        event_date=date,
                        current_date=date
                    )
                    
                    # Path B: Semantic features
                    triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
                    f_sem = triple_embeddings.get(
                        triple_text,
                        np.zeros(1024, dtype=np.float32)
                    )
                    # Hybrid (1037 + 1024 = 2061-dim)
                    f_hybrid = np.concatenate([f_struct, f_sem])
                    features.append(f_hybrid)
                    
                    self.stats['triples_processed'] += 1
                
                except Exception as e:
                    # Skip bad triples
                    continue
            
            # Store
            if date not in results:
                results[date] = {}
            results[date][ticker] = features
        
        print(f"\n✅ Processing complete!")
        print(f"   Triples processed: {self.stats['triples_processed']:,}")
        print(f"   Voyage API calls: {self.stats['voyage_calls']:,}")
        print(f"   Cache hits: {self.stats['cache_hits']:,}")
        print(f"   Cache hit rate: {self.stats['cache_hits'] / len(unique_triples) * 100:.1f}%")
        
        return results

    # kg_module/hybrid_extractor.py - ADD ERROR HANDLING

    def _safe_embed_with_retry(self, texts, max_retries=3):
        """
        Embed with proper error handling and logging
        """
        for attempt in range(max_retries):
            try:
                print(f"   Attempt {attempt+1}: Embedding {len(texts)} texts...")
                
                result = self.client.embed(
                    texts=texts,
                    model=self.model,
                    input_type='document'
                )
                
                # CRITICAL: Check if result is valid
                if result is None:
                    print(f"   ⚠️  Voyage returned None!")
                    raise ValueError("API returned None")
                
                if not hasattr(result, 'embeddings') or not result.embeddings:
                    print(f"   ⚠️  Voyage returned empty embeddings!")
                    raise ValueError("API returned empty")
                
                # CRITICAL: Check embedding dimension
                if len(result.embeddings[0]) != 1024:
                    print(f"   ⚠️  Wrong dimension: {len(result.embeddings[0])}")
                    raise ValueError(f"Wrong dimension")
                
                # SUCCESS
                embeddings = [np.array(emb, dtype=np.float32) for emb in result.embeddings]
                
                # Verify not all zeros
                if all(np.allclose(emb, 0) for emb in embeddings):
                    print(f"   ⚠️  All embeddings are zeros!")
                    raise ValueError("All zeros")
                
                print(f"   ✓ Successfully embedded {len(texts)} texts")
                return embeddings
                
            except Exception as e:
                print(f"   ❌ Attempt {attempt+1} failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"   Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"   🔴 All retries exhausted!")
                    raise RuntimeError(f"Embedding failed after {max_retries} attempts: {e}")