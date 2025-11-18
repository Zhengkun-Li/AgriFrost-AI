"""Graph structure builder for spatial-temporal models.

This module provides utilities for building graph structures from station metadata,
including distance-based graphs (radius) and kNN graphs.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, coo_matrix


class GraphBuilder:
    """Build graph structures for spatial-temporal models."""
    
    def __init__(self, metadata_path: Optional[Union[str, Path]] = None):
        """Initialize graph builder.
        
        Args:
            metadata_path: Path to station metadata JSON file.
                If None, will try to find it automatically.
        """
        self.metadata_path = metadata_path
        self.metadata = None
        self.station_coords = None
        self.station_ids = None
        self.distance_matrix = None
        
        if metadata_path:
            self.load_metadata(metadata_path)
        else:
            self._auto_load_metadata()
    
    def _auto_load_metadata(self):
        """Automatically find and load station metadata."""
        # Try common locations
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "data" / "external" / "cimis_station_metadata.json",
            Path(__file__).parent.parent.parent.parent.parent / "data" / "external" / "cimis_station_metadata.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                self.load_metadata(path)
                return
        
        raise FileNotFoundError(
            "Could not find station metadata file. "
            "Please provide metadata_path or ensure cimis_station_metadata.json exists."
        )
    
    def load_metadata(self, metadata_path: Union[str, Path]):
        """Load station metadata from JSON file.
        
        Args:
            metadata_path: Path to metadata JSON file.
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Extract station coordinates
        self.station_ids = np.array([s['Stn Id'] for s in self.metadata])
        self.station_coords = np.array([
            [s['Latitude'], s['Longitude']] for s in self.metadata
        ])
        
        # Compute distance matrix (in km)
        self._compute_distance_matrix()
    
    def _compute_distance_matrix(self):
        """Compute pairwise distance matrix between stations (in km).
        
        Uses Haversine formula for great-circle distance on Earth's surface.
        """
        if self.station_coords is None:
            raise ValueError("Station coordinates not loaded. Call load_metadata() first.")
        
        # Convert to radians
        lat_rad = np.deg2rad(self.station_coords[:, 0])
        lon_rad = np.deg2rad(self.station_coords[:, 1])
        
        # Create meshgrid for pairwise computation
        lat1_grid, lat2_grid = np.meshgrid(lat_rad, lat_rad)
        lon1_grid, lon2_grid = np.meshgrid(lon_rad, lon_rad)
        
        # Haversine formula
        dlat = lat2_grid - lat1_grid
        dlon = lon2_grid - lon1_grid
        a = np.sin(dlat/2)**2 + np.cos(lat1_grid) * np.cos(lat2_grid) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to avoid numerical errors
        R = 6371.0  # Earth radius in km
        self.distance_matrix = R * c
    
    def build_radius_graph(
        self,
        radius_km: float,
        edge_weight: str = 'gaussian',
        sigma: Optional[float] = None,
        self_loops: bool = True
    ) -> Dict:
        """Build graph based on distance radius.
        
        Args:
            radius_km: Maximum distance in km for edges.
            edge_weight: Type of edge weight ('gaussian', 'distance', 'binary', 'learnable').
            sigma: Standard deviation for Gaussian weights (default: radius_km / 3).
            self_loops: Whether to include self-loops (diagonal = 1).
        
        Returns:
            Dictionary containing:
                - 'adj_matrix': Adjacency matrix (sparse or dense)
                - 'edge_weights': Edge weight matrix (if applicable)
                - 'station_ids': Station ID array
                - 'distance_matrix': Distance matrix
                - 'graph_type': 'radius'
                - 'graph_param': radius_km
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not computed. Load metadata first.")
        
        n = len(self.station_ids)
        adj_matrix = (self.distance_matrix <= radius_km).astype(float)
        
        if not self_loops:
            np.fill_diagonal(adj_matrix, 0)
        
        # Compute edge weights
        edge_weights = None
        if edge_weight == 'gaussian':
            if sigma is None:
                sigma = radius_km / 3.0
            # Gaussian weight: exp(-d^2 / (2*sigma^2))
            edge_weights = np.exp(-(self.distance_matrix**2) / (2 * sigma**2))
            edge_weights = edge_weights * adj_matrix  # Mask by adjacency
            if self_loops:
                np.fill_diagonal(edge_weights, 1.0)
        elif edge_weight == 'distance':
            # Inverse distance (closer = stronger)
            edge_weights = 1.0 / (self.distance_matrix + 1e-6)  # Add small epsilon to avoid division by zero
            edge_weights = edge_weights * adj_matrix
            if self_loops:
                np.fill_diagonal(edge_weights, 1.0)
        elif edge_weight == 'binary':
            edge_weights = adj_matrix.copy()
        elif edge_weight == 'learnable':
            # Initialize with Gaussian weights, will be learned during training
            if sigma is None:
                sigma = radius_km / 3.0
            edge_weights = np.exp(-(self.distance_matrix**2) / (2 * sigma**2))
            edge_weights = edge_weights * adj_matrix
            if self_loops:
                np.fill_diagonal(edge_weights, 1.0)
        else:
            raise ValueError(f"Unknown edge_weight type: {edge_weight}")
        
        return {
            'adj_matrix': adj_matrix,
            'edge_weights': edge_weights,
            'station_ids': self.station_ids.copy(),
            'distance_matrix': self.distance_matrix.copy(),
            'graph_type': 'radius',
            'graph_param': radius_km,
            'edge_weight_type': edge_weight,
        }
    
    def build_knn_graph(
        self,
        k: int,
        edge_weight: str = 'gaussian',
        sigma: Optional[float] = None,
        self_loops: bool = True
    ) -> Dict:
        """Build k-nearest neighbor graph.
        
        Args:
            k: Number of nearest neighbors (excluding self).
            edge_weight: Type of edge weight ('gaussian', 'distance', 'binary', 'learnable').
            sigma: Standard deviation for Gaussian weights (default: mean distance to k-th neighbor).
            self_loops: Whether to include self-loops.
        
        Returns:
            Dictionary containing graph structure.
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not computed. Load metadata first.")
        
        n = len(self.station_ids)
        adj_matrix = np.zeros((n, n))
        
        # Find k nearest neighbors for each node
        for i in range(n):
            distances = self.distance_matrix[i, :].copy()
            if not self_loops:
                distances[i] = np.inf  # Exclude self
            
            # Get k+1 nearest (including self if self_loops=True)
            k_nearest = np.argsort(distances)[:k+1] if self_loops else np.argsort(distances)[:k]
            adj_matrix[i, k_nearest] = 1.0
        
        # Make symmetric (undirected graph)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        adj_matrix = (adj_matrix > 0).astype(float)
        
        # Compute edge weights
        edge_weights = None
        if edge_weight == 'gaussian':
            if sigma is None:
                # Use mean distance to k-th neighbor as sigma
                k_distances = []
                for i in range(n):
                    distances = self.distance_matrix[i, :].copy()
                    if not self_loops:
                        distances[i] = np.inf
                    k_dist = np.sort(distances)[k]
                    k_distances.append(k_dist)
                sigma = np.mean(k_distances)
            
            edge_weights = np.exp(-(self.distance_matrix**2) / (2 * sigma**2))
            edge_weights = edge_weights * adj_matrix
            if self_loops:
                np.fill_diagonal(edge_weights, 1.0)
        elif edge_weight == 'distance':
            edge_weights = 1.0 / (self.distance_matrix + 1e-6)
            edge_weights = edge_weights * adj_matrix
            if self_loops:
                np.fill_diagonal(edge_weights, 1.0)
        elif edge_weight == 'binary':
            edge_weights = adj_matrix.copy()
        elif edge_weight == 'learnable':
            if sigma is None:
                k_distances = []
                for i in range(n):
                    distances = self.distance_matrix[i, :].copy()
                    if not self_loops:
                        distances[i] = np.inf
                    k_dist = np.sort(distances)[k]
                    k_distances.append(k_dist)
                sigma = np.mean(k_distances)
            edge_weights = np.exp(-(self.distance_matrix**2) / (2 * sigma**2))
            edge_weights = edge_weights * adj_matrix
            if self_loops:
                np.fill_diagonal(edge_weights, 1.0)
        else:
            raise ValueError(f"Unknown edge_weight type: {edge_weight}")
        
        return {
            'adj_matrix': adj_matrix,
            'edge_weights': edge_weights,
            'station_ids': self.station_ids.copy(),
            'distance_matrix': self.distance_matrix.copy(),
            'graph_type': 'knn',
            'graph_param': k,
            'edge_weight_type': edge_weight,
        }
    
    @staticmethod
    def save_graph(graph: Dict, path: Union[str, Path]):
        """Save graph structure to disk.
        
        Args:
            graph: Graph dictionary from build_radius_graph or build_knn_graph.
            path: Path to save the graph (directory or file).
        """
        path = Path(path)
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            graph_path = path / "graph.pkl"
        else:
            graph_path = path
            graph_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f)
    
    @staticmethod
    def load_graph(path: Union[str, Path]) -> Dict:
        """Load graph structure from disk.
        
        Args:
            path: Path to graph file or directory containing graph.pkl.
        
        Returns:
            Graph dictionary.
        """
        path = Path(path)
        if path.is_dir():
            graph_path = path / "graph.pkl"
        else:
            graph_path = path
        
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        return graph


def get_graph_cache_path(
    graph_type: str,
    graph_param: Union[int, float],
    cache_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Get cache path for graph structure.
    
    Args:
        graph_type: 'radius' or 'knn'.
        graph_param: Radius in km (for 'radius') or k (for 'knn').
        cache_dir: Cache directory (default: data/interim/graph).
    
    Returns:
        Path to cached graph file.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "interim" / "graph"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if graph_type == 'radius':
        filename = f"radius_{graph_param}km.pkl"
    elif graph_type == 'knn':
        filename = f"knn_{graph_param}.pkl"
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    return cache_dir / filename

