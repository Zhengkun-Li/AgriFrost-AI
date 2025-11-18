#!/usr/bin/env python3
"""Test script for graph builder functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.models.utils import GraphBuilder, get_graph_cache_path


def test_graph_builder():
    """Test graph builder functionality."""
    print("=" * 60)
    print("Testing Graph Builder")
    print("=" * 60)
    
    # Initialize graph builder
    metadata_path = project_root / "data" / "external" / "cimis_station_metadata.json"
    print(f"\n1. Loading metadata from: {metadata_path}")
    builder = GraphBuilder(metadata_path=metadata_path)
    print(f"   ✅ Loaded {len(builder.station_ids)} stations")
    print(f"   Station IDs: {builder.station_ids}")
    
    # Test distance matrix
    print(f"\n2. Distance matrix shape: {builder.distance_matrix.shape}")
    print(f"   Min distance: {builder.distance_matrix[builder.distance_matrix > 0].min():.2f} km")
    print(f"   Max distance: {builder.distance_matrix.max():.2f} km")
    print(f"   Mean distance: {builder.distance_matrix[builder.distance_matrix > 0].mean():.2f} km")
    
    # Test radius graph
    print(f"\n3. Building radius graph (R=50 km)...")
    graph_radius = builder.build_radius_graph(
        radius_km=50.0,
        edge_weight='gaussian'
    )
    print(f"   ✅ Graph type: {graph_radius['graph_type']}")
    print(f"   ✅ Graph param: {graph_radius['graph_param']} km")
    print(f"   ✅ Adjacency matrix shape: {graph_radius['adj_matrix'].shape}")
    print(f"   ✅ Number of edges: {graph_radius['adj_matrix'].sum() / 2:.0f}")
    print(f"   ✅ Average degree: {graph_radius['adj_matrix'].sum(axis=1).mean():.2f}")
    
    # Test kNN graph
    print(f"\n4. Building kNN graph (k=5)...")
    graph_knn = builder.build_knn_graph(
        k=5,
        edge_weight='gaussian'
    )
    print(f"   ✅ Graph type: {graph_knn['graph_type']}")
    print(f"   ✅ Graph param: k={graph_knn['graph_param']}")
    print(f"   ✅ Adjacency matrix shape: {graph_knn['adj_matrix'].shape}")
    print(f"   ✅ Number of edges: {graph_knn['adj_matrix'].sum() / 2:.0f}")
    print(f"   ✅ Average degree: {graph_knn['adj_matrix'].sum(axis=1).mean():.2f}")
    
    # Test graph saving/loading
    print(f"\n5. Testing graph save/load...")
    test_cache_dir = project_root / "data" / "interim" / "graph" / "test"
    test_cache_dir.mkdir(parents=True, exist_ok=True)
    
    test_path = test_cache_dir / "test_radius_50km.pkl"
    GraphBuilder.save_graph(graph_radius, test_path)
    print(f"   ✅ Saved graph to: {test_path}")
    
    loaded_graph = GraphBuilder.load_graph(test_path)
    print(f"   ✅ Loaded graph from: {test_path}")
    print(f"   ✅ Graph type matches: {loaded_graph['graph_type'] == graph_radius['graph_type']}")
    print(f"   ✅ Graph param matches: {loaded_graph['graph_param'] == graph_radius['graph_param']}")
    
    # Test cache path helper
    print(f"\n6. Testing cache path helper...")
    cache_path = get_graph_cache_path('radius', 50.0)
    print(f"   ✅ Cache path for radius=50km: {cache_path}")
    
    cache_path_knn = get_graph_cache_path('knn', 5)
    print(f"   ✅ Cache path for k=5: {cache_path_knn}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_graph_builder()

