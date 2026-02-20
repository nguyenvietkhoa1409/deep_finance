import os
import pickle
import numpy as np
import voyageai
from pathlib import Path

# Đảm bảo import được config của bạn
try:
    from configs.config import GlobalConfig
except ImportError:
    class GlobalConfig:
        VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")

def verify_voyage_api():
    print("="*60)
    print("1. CHECKING VOYAGE API CONNECTIVITY")
    print("="*60)
    api_key = getattr(GlobalConfig, 'VOYAGE_API_KEY', None)
    if not api_key:
        print("❌ VOYAGE_API_KEY is missing in GlobalConfig.")
        return False
        
    try:
        client = voyageai.Client(api_key=api_key)
        result = client.embed(["Test finance prediction KG"], model='voyage-finance-2', input_type='document')
        emb = np.array(result.embeddings[0])
        print("✅ Voyage API Connection Successful!")
        print(f"   Returned embedding shape: {emb.shape}")
        print(f"   Non-zero ratio: {np.mean(emb != 0)*100:.2f}%")
        return True
    except Exception as e:
        print(f"❌ Voyage API Call Failed: {e}")
        return False

def inspect_embedding_cache():
    print("\n" + "="*60)
    print("2. INSPECTING TRIPLE EMBEDDINGS CACHE")
    print("="*60)
    
    cache_path = Path('./data/interim/kg_cache/triple_embeddings.pkl')
    if not cache_path.exists():
        print(f"⚠️ Cache file not found at {cache_path}")
        return
        
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            
        print(f"✅ Cache loaded successfully. Total cached triples: {len(cache_data)}")
        
        if len(cache_data) == 0:
            print("⚠️ Cache is empty.")
            return

        # Lấy mẫu một vài embeddings để kiểm tra
        all_zeros_count = 0
        valid_count = 0
        
        for text, emb in list(cache_data.items())[:100]: # Check 100 mẫu đầu tiên
            if np.allclose(emb, 0):
                all_zeros_count += 1
            else:
                valid_count += 1
                
        print(f"   Sampled 100 embeddings from cache:")
        print(f"   - Valid non-zero vectors: {valid_count}")
        print(f"   - All-zero vectors (API Fallbacks): {all_zeros_count}")
        
        if all_zeros_count > 0:
            print("   ❌ ALERT: The cache contains zero-vectors. The API silently failed during generation.")
            print("      Action needed: Delete this cache file and regenerate.")
        else:
            print("   ✅ The cached embeddings are healthy (non-zero). The issue is strictly in the code logic.")
            
    except Exception as e:
        print(f"❌ Error reading cache: {e}")

def simulate_slicing_bug():
    print("\n" + "="*60)
    print("3. SIMULATING HYBRID EXTRACTOR SLICING LOGIC")
    print("="*60)
    
    # Mô phỏng dữ liệu giống hệt dòng 324-342 trong hybrid_extractor.py
    doc_emb = np.zeros(1024, dtype=np.float32)
    
    # Giả lập Voyage API trả về vector hoàn hảo (toàn số 1 để dễ nhìn)
    triple_emb = np.ones(1024, dtype=np.float32) 
    
    # Logic hiện tại của bạn
    ctx_emb = np.concatenate([doc_emb, triple_emb])
    f_sem_current = ctx_emb[:768]
    
    print("CURRENT LOGIC IN YOUR CODE (f_sem = ctx_emb[:768]):")
    print(f"   f_sem shape: {f_sem_current.shape}")
    print(f"   f_sem mean : {f_sem_current.mean()} (Should be > 0 if it contains triple_emb data)")
    
    if f_sem_current.mean() == 0:
        print("   ❌ DEMONSTRATED BUG: The current slicing grabs ONLY the zeros from doc_emb!")
        
    # Logic đúng
    f_sem_fixed = triple_emb[:768]  # Cắt trực tiếp từ triple_emb
    print("\nFIXED LOGIC (f_sem = triple_emb[:768]):")
    print(f"   f_sem shape: {f_sem_fixed.shape}")
    print(f"   f_sem mean : {f_sem_fixed.mean()} (Value 1.0 means it captures the triple representation)")

if __name__ == "__main__":
    verify_voyage_api()
    inspect_embedding_cache()
    simulate_slicing_bug()
    print("\n✅ Debugging completed.")