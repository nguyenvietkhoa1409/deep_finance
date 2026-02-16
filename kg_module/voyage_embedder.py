"""
Voyage AI Embeddings for Knowledge Graph
Generate high-quality embeddings for tickers and relations
[UPDATED] Added parallel batch processing support
"""

import os
import voyageai
import numpy as np
import pickle
from typing import Dict, List
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from configs.config import GlobalConfig


class VoyageKGEmbedder:
    """
    Generate embeddings for KG components using Voyage Finance model.
    [NEW] Supports parallel batch processing
    """
    
    def __init__(
        self,
        cache_dir: str = './data/interim/kg_cache',
        model: str = 'voyage-finance-2',
        max_workers: int = 1  # [NEW] Parallel workers
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.model = model
        self.max_workers = max_workers  # [NEW]
        
        # Initialize Client safely
        api_key = getattr(GlobalConfig, 'VOYAGE_API_KEY', None)
        self.client = voyageai.Client(api_key=api_key) if api_key else None
        
        self.ticker_cache_path = os.path.join(cache_dir, 'voyage_ticker_embeddings.pkl')
        self.relation_cache_path = os.path.join(cache_dir, 'voyage_relation_embeddings.pkl')
        
        # Rate limiting
        limits = getattr(GlobalConfig, 'VOYAGE_RATE_LIMITS', {True: {'SLEEP': 1.0}})
        payment_added = getattr(GlobalConfig, 'PAYMENT_ADDED', True)
        self.sleep_time = limits.get(payment_added, {'SLEEP': 1.0})['SLEEP']
        
        # [NEW] Thread safety
        self.embeddings_lock = threading.Lock()
    
    def _load_cache(self, path: str) -> Dict:
        """Helper to safely load cache."""
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"   ⚠️  Cache corrupted at {path}: {e}")
                print("   🔄 Will regenerate embeddings.")
        return None

    def generate_ticker_embeddings(
        self,
        tickers: List[str],
        force_rebuild: bool = False
    ) -> Dict[str, np.ndarray]:
        """Generate ticker embeddings with cache support."""
        # Check cache
        cached = None
        if not force_rebuild:
            cached = self._load_cache(self.ticker_cache_path)

        if cached:
            missing = [t for t in tickers if t not in cached]
            if not missing:
                print(f"   ✓ All {len(tickers)} tickers found in cache")
                return cached
            else:
                print(f"   ⚠️  {len(missing)} tickers missing from cache")
        else:
            cached = {}
            missing = tickers
        
        # Generate embeddings for missing tickers
        descriptions = self._get_company_descriptions()
        print(f"\n🚀 Generating Voyage embeddings for {len(missing)} tickers...")
        
        texts_to_embed = []
        ticker_order = []
        
        for ticker in missing:
            desc = descriptions.get(ticker, f"{ticker} stock")
            text = f"Company: {ticker}. {desc}"
            texts_to_embed.append(text)
            ticker_order.append(ticker)
        
        # [UPDATED] Use parallel or serial batch embedding
        embeddings = self._batch_embed(texts_to_embed)
        
        # Update cache
        for ticker, emb in zip(ticker_order, embeddings):
            cached[ticker] = emb
        
        # Save cache
        with open(self.ticker_cache_path, 'wb') as f:
            pickle.dump(cached, f)
        
        print(f"   ✓ Generated and cached {len(missing)} ticker embeddings")
        
        return cached
    
    def generate_relation_embeddings(
        self,
        relation_types: Dict[str, Dict],
        force_rebuild: bool = False
    ) -> Dict[str, np.ndarray]:
        """Generate relation embeddings with cache support."""
        # Check cache
        cached = None
        if not force_rebuild:
            cached = self._load_cache(self.relation_cache_path)
            
        if cached:
            missing = [r for r in relation_types.keys() if r not in cached]
            if not missing:
                print(f"   ✓ All {len(relation_types)} relations found in cache")
                return cached
        else:
            cached = {}
            missing = list(relation_types.keys())
        
        print(f"\n🚀 Generating Voyage embeddings for {len(missing)} relation types...")
        
        texts_to_embed = []
        relation_order = []
        
        for rel_name in missing:
            # Handle different relation_types formats
            if isinstance(relation_types, list):
                keywords = []
                print(f"   ⚠️ Warning: relation_types passed as List, expected Dict.")
            else:
                rel_config = relation_types.get(rel_name, {})
                keywords = rel_config.get('keywords', [])
            
            text = f"Financial event type: {rel_name.replace('_', ' ')}. Keywords: {', '.join(keywords)}"
            texts_to_embed.append(text)
            relation_order.append(rel_name)
        
        # [UPDATED] Use parallel or serial batch embedding
        embeddings = self._batch_embed(texts_to_embed)
        
        # Update cache
        for rel_name, emb in zip(relation_order, embeddings):
            cached[rel_name] = emb
        
        # Save cache
        with open(self.relation_cache_path, 'wb') as f:
            pickle.dump(cached, f)
        
        print(f"   ✓ Generated and cached {len(missing)} relation embeddings")
        
        return cached
    
    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Batch embed texts with optional parallelization.
        
        [UPDATED] Supports both serial and parallel processing
        """
        if not self.client:
            print("   ⚠️  Voyage Client not initialized (No API Key). Using Random.")
            return [np.random.randn(1024).astype(np.float32) * 0.01 for _ in texts]

        batch_size = 40  # Voyage optimal batch size
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"   Processing {len(texts)} texts in {num_batches} batches...")
        
        if self.max_workers > 1 and num_batches > 1:
            # Parallel processing
            return self._batch_embed_parallel(texts, batch_size, num_batches)
        else:
            # Serial processing (original)
            return self._batch_embed_serial(texts, batch_size)
    
    def _batch_embed_serial(self, texts: List[str], batch_size: int) -> List[np.ndarray]:
        """Original serial batch embedding."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i + batch_size]
            try:
                result = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type='document'
                )
                batch_embeddings = [np.array(emb, dtype=np.float32) for emb in result.embeddings]
                all_embeddings.extend(batch_embeddings)
                time.sleep(self.sleep_time)
                
            except Exception as e:
                print(f"\n⚠️  Batch {i//batch_size} failed: {e}")
                print("   👉 Switching to Random Embeddings for this batch.")
                for _ in range(len(batch)):
                    all_embeddings.append(np.random.randn(1024).astype(np.float32) * 0.01)
        
        return all_embeddings
    
    def _batch_embed_parallel(
        self, 
        texts: List[str], 
        batch_size: int,
        num_batches: int
    ) -> List[np.ndarray]:
        """
        [NEW] Parallel batch embedding.
        
        Uses ThreadPoolExecutor to process multiple batches concurrently.
        """
        print(f"   🚀 Parallel mode: {self.max_workers} workers")
        
        # Create batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batches.append((i, batch_texts))
        
        # Results storage (preserve order)
        results = [None] * num_batches
        
        def embed_single_batch(batch_id, batch_texts):
            """Embed a single batch (thread-safe)."""
            try:
                result = self.client.embed(
                    texts=batch_texts,
                    model=self.model,
                    input_type='document'
                )
                batch_embeddings = [np.array(emb, dtype=np.float32) for emb in result.embeddings]
                
                # Sleep (reduced for parallel)
                time.sleep(self.sleep_time / self.max_workers)
                
                return batch_id, batch_embeddings, None
            
            except Exception as e:
                # Fallback: random embeddings
                batch_embeddings = [
                    np.random.randn(1024).astype(np.float32) * 0.01 
                    for _ in batch_texts
                ]
                return batch_id, batch_embeddings, str(e)
        
        # Process in parallel
        pbar = tqdm(total=num_batches, desc="Embedding batches (parallel)")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_batch = {
                executor.submit(embed_single_batch, batch_id, batch_texts): batch_id
                for batch_id, batch_texts in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_id, batch_embeddings, error = future.result()
                
                if error:
                    print(f"\n   ⚠️ Batch {batch_id} failed: {error}")
                
                results[batch_id] = batch_embeddings
                pbar.update(1)
        
        pbar.close()
        
        # Flatten results (preserve order)
        all_embeddings = []
        for batch_embeddings in results:
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def _get_company_descriptions(self) -> Dict[str, str]:
        """Company descriptions for ticker embeddings."""
        return {
            'AMZN': 'Amazon.com Inc. technology e-commerce cloud computing AWS.',
            'TSLA': 'Tesla Inc. electric vehicles energy storage solar.',
            'MSFT': 'Microsoft Corporation software cloud services Azure.',
            'NFLX': 'Netflix Inc. streaming entertainment content production.',
        }


def adapt_embeddings_dimension(
    embeddings: Dict[str, np.ndarray], 
    target_dim: int = 768
) -> Dict[str, np.ndarray]:
    """
    Adapt embedding dimensions via truncation or padding.
    
    Args:
        embeddings: Dict of embeddings
        target_dim: Target dimension
    
    Returns:
        Adapted embeddings
    """
    if not embeddings:
        return {}
    
    sample_key = next(iter(embeddings))
    current_dim = embeddings[sample_key].shape[0]
    
    if current_dim == target_dim:
        return embeddings
    
    print(f"📐 Adapting embeddings: {current_dim}D → {target_dim}D")
    
    if current_dim > target_dim:
        # Truncate
        return {k: v[:target_dim] for k, v in embeddings.items()}
    else:
        # Pad with zeros
        adapted = {}
        for k, v in embeddings.items():
            padding = np.zeros(target_dim - current_dim, dtype=np.float32)
            adapted[k] = np.concatenate([v, padding])
        return adapted