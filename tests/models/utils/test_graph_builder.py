"""Unit tests for GraphBuilder."""

import pytest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from src.models.utils.graph_builder import GraphBuilder, get_graph_cache_path


class TestGraphBuilder:
    """Test GraphBuilder functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_metadata(self, temp_dir):
        """Create sample station metadata."""
        metadata_path = temp_dir / "station_metadata.json"
        metadata = [
            {
                "Stn Id": 1,
                "Latitude": 38.5,
                "Longitude": -121.5
            },
            {
                "Stn Id": 2,
                "Latitude": 38.6,
                "Longitude": -121.6
            },
            {
                "Stn Id": 3,
                "Latitude": 38.7,
                "Longitude": -121.7
            }
        ]
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        return metadata_path
    
    @pytest.fixture
    def builder(self, sample_metadata):
        """Create GraphBuilder instance."""
        return GraphBuilder(metadata_path=sample_metadata)
    
    def test_init(self, builder):
        """Test GraphBuilder initialization."""
        assert builder.metadata is not None
        assert builder.station_ids is not None
        assert builder.station_coords is not None
        assert builder.distance_matrix is not None
        assert len(builder.station_ids) == 3
    
    def test_build_radius_graph(self, builder):
        """Test build_radius_graph."""
        graph = builder.build_radius_graph(radius_km=100.0, edge_weight='gaussian')
        
        assert graph['graph_type'] == 'radius'
        assert graph['graph_param'] == 100.0
        assert 'adj_matrix' in graph
        assert 'edge_weights' in graph
        assert 'station_ids' in graph
        assert 'distance_matrix' in graph
        assert graph['adj_matrix'].shape == (3, 3)
    
    def test_build_knn_graph(self, builder):
        """Test build_knn_graph."""
        graph = builder.build_knn_graph(k=2, edge_weight='gaussian')
        
        assert graph['graph_type'] == 'knn'
        assert graph['graph_param'] == 2
        assert 'adj_matrix' in graph
        assert 'edge_weights' in graph
        assert 'station_ids' in graph
    
    def test_save_graph_with_metadata_export(self, builder, temp_dir):
        """Test save_graph with metadata export (2×2+1 compatibility)."""
        graph = builder.build_radius_graph(radius_km=50.0)
        
        # Create run_metadata.json
        from src.utils.metadata import ExperimentMetadata
        metadata = ExperimentMetadata(
            model_name="dcrnn",
            horizon_h=12
        )
        metadata_path = temp_dir / "run_metadata.json"
        metadata.save(temp_dir)
        
        # Save graph with metadata export
        graph_path = temp_dir / "graph.pkl"
        GraphBuilder.save_graph(graph, graph_path, metadata_path=metadata_path)
        
        # Verify graph saved
        assert graph_path.exists()
        
        # Verify metadata updated
        loaded_metadata = ExperimentMetadata.load(metadata_path)
        assert loaded_metadata.radius_km == 50.0
    
    def test_save_graph_knn_metadata_export(self, builder, temp_dir):
        """Test save_graph with knn metadata export."""
        graph = builder.build_knn_graph(k=5)
        
        # Create run_metadata.json
        from src.utils.metadata import ExperimentMetadata
        metadata = ExperimentMetadata(
            model_name="dcrnn",
            horizon_h=12
        )
        metadata_path = temp_dir / "run_metadata.json"
        metadata.save(temp_dir)
        
        # Save graph with metadata export
        graph_path = temp_dir / "graph.pkl"
        GraphBuilder.save_graph(graph, graph_path, metadata_path=metadata_path)
        
        # Verify metadata updated
        loaded_metadata = ExperimentMetadata.load(metadata_path)
        assert loaded_metadata.knn_k == 5
    
    def test_load_graph(self, builder, temp_dir):
        """Test load_graph."""
        graph = builder.build_radius_graph(radius_km=50.0)
        
        # Save graph
        graph_path = temp_dir / "graph.pkl"
        GraphBuilder.save_graph(graph, graph_path)
        
        # Load graph
        loaded_graph = GraphBuilder.load_graph(graph_path)
        
        assert loaded_graph['graph_type'] == 'radius'
        assert loaded_graph['graph_param'] == 50.0
        assert np.array_equal(loaded_graph['station_ids'], graph['station_ids'])
    
    def test_get_graph_cache_path_enhanced(self, temp_dir):
        """Test get_graph_cache_path with enhanced cache validation."""
        station_ids = np.array([1, 2, 3])
        station_coords = np.array([
            [38.5, -121.5],
            [38.6, -121.6],
            [38.7, -121.7]
        ])
        
        cache_path = get_graph_cache_path(
            graph_type='radius',
            graph_param=50.0,
            cache_dir=temp_dir,
            station_ids=station_ids,
            station_coords=station_coords
        )
        
        # Should include hash in filename
        assert 'radius_50.0km' in str(cache_path)
        assert '_stations_' in str(cache_path)
        assert '_coords_' in str(cache_path)
        assert cache_path.suffix == '.pkl'
    
    def test_get_graph_cache_path_basic(self, temp_dir):
        """Test get_graph_cache_path without validation."""
        cache_path = get_graph_cache_path(
            graph_type='knn',
            graph_param=5,
            cache_dir=temp_dir
        )
        
        assert 'knn_5' in str(cache_path)
        assert cache_path.suffix == '.pkl'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

