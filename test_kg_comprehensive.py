"""
Comprehensive KG Module Test Script (FIXED)
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime


print("="*60)
print("🧪 COMPREHENSIVE KG MODULE TEST")
print("="*60)

# Test 1: Imports
print("\n1️⃣  Testing imports...")
try:
    from kg_module.voyage_embedder import VoyageKGEmbedder
    from kg_module.simple_kg import SimpleKnowledgeGraph
    from kg_module.config import NODE_FEATURE_DIM
    from encoders.kg_encoder import KnowledgeGraphEncoder
    from torch_geometric.data import Data
    print("   ✓ All imports successful")
    print(f"   📐 Node feature dim: {NODE_FEATURE_DIM}")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Voyage Embeddings
print("\n2️⃣  Testing Voyage embeddings...")
try:
    from kg_module.config import RELATION_TYPES
    
    embedder = VoyageKGEmbedder()
    test_tickers = ['AMZN']
    
    # Generate ticker embeddings
    embs = embedder.generate_ticker_embeddings(test_tickers, force_rebuild=False)
    assert len(embs) == 1
    dim = embs['AMZN'].shape[0]
    print(f"   ✓ Ticker embedding dim: {dim}")
    
    # Generate relation embeddings
    rel_embs = embedder.generate_relation_embeddings(RELATION_TYPES, force_rebuild=False)
    assert len(rel_embs) == len(RELATION_TYPES)
    print(f"   ✓ Relation embeddings: {len(rel_embs)} types")
    
except Exception as e:
    print(f"   ⚠️  Voyage test failed (may need API key): {e}")

# Test 3: Graph Construction
print("\n3️⃣  Testing graph construction...")
try:
    kg = SimpleKnowledgeGraph(
        tickers=['AMZN'],
        use_llm=False,
        use_voyage=True
    )
    kg.setup(force_rebuild=False)
    
    # Create mock triple
    mock_triple = {
        'subject': 'AMZN',
        'relation': 'revenue_change',
        'object': 'Q1 revenue increased 23%',
        'magnitude': 0.23,
        'polarity': 1,
        'certainty': 1.0,
        'date': datetime(2024, 1, 1),
        'source': 'Test'
    }
    
    graph = kg.graph_builder.build_graph(
        [mock_triple], 'AMZN', datetime(2024, 1, 1)
    )
    
    assert 'node_features' in graph
    print(f"   ✓ Graph structure: {graph['num_nodes']} nodes, {graph['num_edges']} edges")
    print(f"   ✓ Node features shape: {graph['node_features'].shape}")
    print(f"   ✓ Expected dim: {NODE_FEATURE_DIM}, Got: {graph['node_features'].shape[1]}")
    
    # Verify dimension
    assert graph['node_features'].shape[1] == NODE_FEATURE_DIM, \
        f"Dimension mismatch! Expected {NODE_FEATURE_DIM}, got {graph['node_features'].shape[1]}"
    
except Exception as e:
    print(f"   ❌ Graph construction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: PyG Data Conversion
print("\n4️⃣  Testing PyG data conversion...")
try:
    from src.data_loader import data_prepare
    
    dp = data_prepare('dummy.pkl', kg_data_path=None)
    
    # Test with correct dimension
    test_dict = {
        'node_features': torch.randn(5, NODE_FEATURE_DIM),
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        'edge_weight': torch.tensor([1.0, 1.0, 1.0])
    }
    
    data_obj = dp._convert_dict_to_data(test_dict)
    assert isinstance(data_obj, Data)
    assert data_obj.x.shape == (5, NODE_FEATURE_DIM)
    
    print(f"   ✓ PyG Data object: {data_obj.x.shape}")
    
    # Test edge case: numpy array input
    test_dict_numpy = {
        'node_features': np.random.randn(3, NODE_FEATURE_DIM).astype(np.float32),
        'edge_index': np.array([[0, 1], [1, 0]], dtype=np.int64)
    }
    
    data_obj2 = dp._convert_dict_to_data(test_dict_numpy)
    assert isinstance(data_obj2, Data)
    print(f"   ✓ Numpy conversion: {data_obj2.x.shape}")
    
    # Test empty graph
    empty = dp._get_empty_graph()
    assert empty.x.shape == (1, NODE_FEATURE_DIM)
    print(f"   ✓ Empty graph: {empty.x.shape}")
    
except Exception as e:
    print(f"   ❌ PyG conversion failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: GCN Encoder
print("\n5️⃣  Testing GCN encoder...")
try:
    from kg_module.config import NODE_FEATURE_DIM
    
    encoder = KnowledgeGraphEncoder(
        input_dim=NODE_FEATURE_DIM,
        hidden_dim=512,
        output_dim=128
    )
    
    # Create mock sequence of PyG Data graphs
    mock_graphs = []
    for _ in range(2):  # 2 samples
        day_graphs = []
        for _ in range(20):  # 20 days
            # Create simple graph as PyG Data
            x = torch.randn(3, NODE_FEATURE_DIM)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index)
            day_graphs.append(graph)
        mock_graphs.append(day_graphs)
    
    output = encoder(mock_graphs)
    assert output.shape == (2, 20, 128)
    
    print(f"   ✓ Encoder output: {output.shape}")
    print(f"   ✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test with dict format too
    print("   Testing dict format...")
    mock_dicts = []
    for _ in range(2):
        day_dicts = []
        for _ in range(20):
            graph_dict = {
                'node_features': torch.randn(3, NODE_FEATURE_DIM),
                'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            }
            day_dicts.append(graph_dict)
        mock_dicts.append(day_dicts)
    
    output_dict = encoder(mock_dicts)
    assert output_dict.shape == (2, 20, 128)
    print(f"   ✓ Dict format works: {output_dict.shape}")
    
except Exception as e:
    print(f"   ❌ Encoder test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Full Pipeline Simulation
print("\n6️⃣  Testing mini pipeline...")
try:
    # Create small KG
    kg = SimpleKnowledgeGraph(
        tickers=['AMZN'],
        use_llm=False,
        use_voyage=True
    )
    kg.setup(force_rebuild=False)
    
    # Create sequence of triples
    dates = [datetime(2024, 1, i) for i in range(1, 6)]
    triples_by_date = {}
    
    for i, date in enumerate(dates):
        triples_by_date[date] = [{
            'subject': 'AMZN',
            'relation': 'revenue_change',
            'object': f'Event {i}',
            'magnitude': 0.1 * i,
            'polarity': 1,
            'certainty': 0.9,
            'date': date,
            'source': 'Test'
        }]
    
    # Build graphs
    graphs = []
    for date in dates:
        graph = kg.graph_builder.build_graph(
            triples_by_date[date], 'AMZN', date
        )
        graphs.append(graph)
    
    print(f"   ✓ Built {len(graphs)} graphs")
    
    # Convert to Data objects
    data_objs = []
    dp = data_prepare('dummy.pkl')
    for g in graphs:
        data_obj = dp._convert_dict_to_data(g)
        data_objs.append(data_obj)
    
    print(f"   ✓ Converted to {len(data_objs)} PyG Data objects")
    
    # Encode with GCN
    encoder = KnowledgeGraphEncoder(input_dim=NODE_FEATURE_DIM, output_dim=128)
    output = encoder([data_objs])  # Batch of 1 sequence
    
    print(f"   ✓ Encoded to: {output.shape}")
    
except Exception as e:
    print(f"   ⚠️  Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ TEST SUITE COMPLETE")
print("="*60)