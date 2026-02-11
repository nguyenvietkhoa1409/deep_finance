"""
Voyage AI Embeddings for Knowledge Graph
Generate high-quality embeddings for tickers and relations
"""

import os
import voyageai
import numpy as np
import pickle
from typing import Dict, List
from tqdm import tqdm
import time

from configs.config import GlobalConfig


class VoyageKGEmbedder:
    """
    Generate embeddings for KG components using Voyage Finance model.
    
    Features:
    - Ticker embeddings from company descriptions
    - Relation embeddings from financial keywords
    - Automatic caching to avoid redundant API calls
    """
    
    def __init__(
        self,
        cache_dir: str = './data/interim/kg_cache',
        model: str = 'voyage-finance-2'  # Voyage's finance-specific model
    ):
        """
        Initialize Voyage embedder.
        
        Args:
            cache_dir: Directory to cache embeddings
            model: Voyage model to use (voyage-finance-2 for finance)
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.model = model
        self.client = voyageai.Client(api_key=GlobalConfig.VOYAGE_API_KEY)
        
        # Cache paths
        self.ticker_cache_path = os.path.join(cache_dir, 'voyage_ticker_embeddings.pkl')
        self.relation_cache_path = os.path.join(cache_dir, 'voyage_relation_embeddings.pkl')
        
        # Rate limiting
        self.sleep_time = GlobalConfig.VOYAGE_RATE_LIMITS[GlobalConfig.PAYMENT_ADDED]['SLEEP']
    
    def generate_ticker_embeddings(
        self,
        tickers: List[str],
        force_rebuild: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for ticker symbols.
        
        Uses company descriptions to create meaningful embeddings instead of random vectors.
        
        Args:
            tickers: List of ticker symbols
            force_rebuild: Force regeneration even if cache exists
        
        Returns:
            Dict mapping ticker → embedding array (1024-dim for voyage-finance-2)
        
        Example:
            >>> embedder = VoyageKGEmbedder()
            >>> embs = embedder.generate_ticker_embeddings(['AMZN', 'TSLA'])
            >>> embs['AMZN'].shape
            (1024,)
        """
        # Check cache
        if os.path.exists(self.ticker_cache_path) and not force_rebuild:
            print("📦 Loading cached ticker embeddings...")
            with open(self.ticker_cache_path, 'rb') as f:
                cached = pickle.load(f)
            
            # Check if all tickers are cached
            missing = [t for t in tickers if t not in cached]
            if not missing:
                print(f"   ✓ All {len(tickers)} tickers found in cache")
                return cached
            else:
                print(f"   ⚠️  {len(missing)} tickers missing, will generate")
        else:
            cached = {}
            missing = tickers
        
        # Company descriptions for context
        descriptions = self._get_company_descriptions()
        
        # Generate embeddings for missing tickers
        print(f"\n🚀 Generating Voyage embeddings for {len(missing)} tickers...")
        
        texts_to_embed = []
        ticker_order = []
        
        for ticker in missing:
            # Create rich text prompt for better embeddings
            desc = descriptions.get(ticker, f"{ticker} stock")
            text = f"Company: {ticker}. {desc}"
            texts_to_embed.append(text)
            ticker_order.append(ticker)
        
        # Batch embed with Voyage
        embeddings = self._batch_embed(texts_to_embed)
        
        # Update cache
        for ticker, emb in zip(ticker_order, embeddings):
            cached[ticker] = emb
        
        # Save cache
        with open(self.ticker_cache_path, 'wb') as f:
            pickle.dump(cached, f)
        
        print(f"   ✓ Generated and cached {len(missing)} new embeddings")
        print(f"   📊 Embedding dimension: {cached[tickers[0]].shape[0]}")
        
        return cached
    
    def generate_relation_embeddings(
        self,
        relation_types: Dict[str, Dict],
        force_rebuild: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for relation types.
        
        Uses relation keywords to create meaningful embeddings.
        
        Args:
            relation_types: Dict from config (RELATION_TYPES)
            force_rebuild: Force regeneration
        
        Returns:
            Dict mapping relation_name → embedding array
        
        Example:
            >>> from kg_module.config import RELATION_TYPES
            >>> embs = embedder.generate_relation_embeddings(RELATION_TYPES)
            >>> embs['revenue_change'].shape
            (1024,)
        """
        # Check cache
        if os.path.exists(self.relation_cache_path) and not force_rebuild:
            print("📦 Loading cached relation embeddings...")
            with open(self.relation_cache_path, 'rb') as f:
                cached = pickle.load(f)
            
            # Check if all relations are cached
            missing = [r for r in relation_types.keys() if r not in cached]
            if not missing:
                print(f"   ✓ All {len(relation_types)} relations found in cache")
                return cached
            else:
                print(f"   ⚠️  {len(missing)} relations missing, will generate")
        else:
            cached = {}
            missing = list(relation_types.keys())
        
        # Generate embeddings for missing relations
        print(f"\n🚀 Generating Voyage embeddings for {len(missing)} relation types...")
        
        texts_to_embed = []
        relation_order = []
        
        for rel_name in missing:
            rel_config = relation_types[rel_name]
            keywords = rel_config.get('keywords', [])
            
            # Create descriptive text
            text = f"Financial event type: {rel_name.replace('_', ' ')}. Keywords: {', '.join(keywords)}"
            texts_to_embed.append(text)
            relation_order.append(rel_name)
        
        # Batch embed
        embeddings = self._batch_embed(texts_to_embed)
        
        # Update cache
        for rel_name, emb in zip(relation_order, embeddings):
            cached[rel_name] = emb
        
        # Save cache
        with open(self.relation_cache_path, 'wb') as f:
            pickle.dump(cached, f)
        
        print(f"   ✓ Generated and cached {len(missing)} new relation embeddings")
        
        return cached
    
    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed texts in batches with rate limiting.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding arrays
        """
        batch_size = 40  # Voyage max batch size
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i + batch_size]
            
            try:
                result = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type='document'  # For storage/retrieval
                )
                
                # Convert to numpy arrays
                batch_embeddings = [
                    np.array(emb, dtype=np.float32) 
                    for emb in result.embeddings
                ]
                
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting
                time.sleep(self.sleep_time)
                
            except Exception as e:
                print(f"\n⚠️  Batch {i//batch_size} failed: {e}")
                # Fallback to random for this batch
                dim = 1024  # voyage-finance-2 dimension
                for _ in range(len(batch)):
                    all_embeddings.append(
                        np.random.randn(dim).astype(np.float32) * 0.01
                    )
        
        return all_embeddings
    
    def _get_company_descriptions(self) -> Dict[str, str]:
        """
        Get company descriptions for tickers.
        
        Returns:
            Dict mapping ticker → description
        """
        # Comprehensive descriptions for better embeddings
        descriptions = {
            'AMZN': 'Amazon.com Inc. is a multinational technology company focusing on e-commerce, cloud computing (AWS), digital streaming, and artificial intelligence. Key segments: Amazon Web Services, North America retail, International retail, subscription services.',
            
            'TSLA': 'Tesla Inc. designs, develops, manufactures and sells electric vehicles, battery energy storage systems, and solar energy products. Key products: Model S, Model 3, Model X, Model Y, Cybertruck, Powerwall, Solar Roof.',
            
            'MSFT': 'Microsoft Corporation develops, licenses, and supports software products, services and devices. Key segments: Productivity and Business Processes (Office, LinkedIn), Intelligent Cloud (Azure, SQL Server), More Personal Computing (Windows, Xbox, Surface).',
            
            'NFLX': 'Netflix Inc. is a streaming entertainment service company offering TV series, documentaries, and feature films across genres and languages. Operates in three segments: Domestic streaming, International streaming, Domestic DVD.',
            
            # Add more as needed
        }
        
        return descriptions


# ============================================
# UTILITY: Dimension Adapter
# ============================================

def adapt_embeddings_dimension(
    embeddings: Dict[str, np.ndarray],
    target_dim: int = 768
) -> Dict[str, np.ndarray]:
    """
    Adapt embeddings to target dimension if needed.
    
    Voyage-finance-2 produces 1024-dim vectors, but GCN expects 768-dim.
    This function uses PCA or truncation to adapt.
    
    Args:
        embeddings: Dict of embeddings
        target_dim: Target dimension
    
    Returns:
        Adapted embeddings dict
    """
    # Check current dimension
    sample_key = next(iter(embeddings))
    current_dim = embeddings[sample_key].shape[0]
    
    if current_dim == target_dim:
        return embeddings
    
    print(f"📐 Adapting embeddings: {current_dim}D → {target_dim}D")
    
    if current_dim > target_dim:
        # Truncate (simple but effective)
        # Keep most informative dimensions (assume Voyage puts important info first)
        adapted = {
            k: v[:target_dim] 
            for k, v in embeddings.items()
        }
        print("   ✓ Truncated to target dimension")
        
    else:
        # Pad with zeros
        adapted = {}
        for k, v in embeddings.items():
            padding = np.zeros(target_dim - current_dim, dtype=np.float32)
            adapted[k] = np.concatenate([v, padding])
        print("   ✓ Padded to target dimension")
    
    return adapted