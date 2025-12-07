"""Graph building strategies (Refactored with strategy pattern).

This module provides strategy classes for building different types of graphs:
- DistanceGraphStrategy: Distance-based graphs (full connectivity with weights)
- RadiusGraphStrategy: Radius-based graphs (within radius)
- KNNGraphStrategy: k-nearest neighbor graphs

This refactoring decouples graph building logic from GraphBuilder,
allowing easier extension and testing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
from scipy.sparse import csr_matrix


class GraphBuildingStrategy(ABC):
    """Abstract base class for graph building strategies."""
    
    @abstractmethod
    def build(
        self,
        distance_matrix: np.ndarray,
        station_ids: np.ndarray,
        edge_weight: str = 'gaussian',
        sigma: Optional[float] = None,
        self_loops: bool = True
    ) -> Dict:
        """Build graph structure.
        
        Args:
            distance_matrix: Pairwise distance matrix (n x n).
            station_ids: Station ID array (n,).
            edge_weight: Type of edge weight ('gaussian', 'distance', 'binary', 'learnable').
            sigma: Standard deviation for Gaussian weights.
            self_loops: Whether to include self-loops.
        
        Returns:
            Graph dictionary with adj_matrix, edge_weights, etc.
        """
        pass
    
    @staticmethod
    def _compute_edge_weights(
        distance_matrix: np.ndarray,
        adj_matrix: np.ndarray,
        edge_weight: str,
        sigma: Optional[float] = None,
        self_loops: bool = True
    ) -> Optional[np.ndarray]:
        """Compute edge weights.
        
        Args:
            distance_matrix: Distance matrix.
            adj_matrix: Adjacency matrix (binary).
            edge_weight: Type of edge weight.
            sigma: Standard deviation for Gaussian (if needed).
            self_loops: Whether to include self-loops.
        
        Returns:
            Edge weight matrix or None.
        """
        if edge_weight == 'gaussian':
            if sigma is None:
                # Default sigma: mean of non-zero distances in adj_matrix
                non_zero_distances = distance_matrix[adj_matrix > 0]
                sigma = np.mean(non_zero_distances) if len(non_zero_distances) > 0 else 1.0
            
            edge_weights = np.exp(-(distance_matrix**2) / (2 * sigma**2))
            edge_weights = edge_weights * adj_matrix
            if self_loops:
                np.fill_diagonal(edge_weights, 1.0)
            return edge_weights
        
        elif edge_weight == 'distance':
            # Inverse distance (closer = stronger)
            edge_weights = 1.0 / (distance_matrix + 1e-6)
            edge_weights = edge_weights * adj_matrix
            if self_loops:
                np.fill_diagonal(edge_weights, 1.0)
            return edge_weights
        
        elif edge_weight == 'binary':
            return adj_matrix.copy()
        
        elif edge_weight == 'learnable':
            # Initialize with Gaussian weights, will be learned during training
            if sigma is None:
                non_zero_distances = distance_matrix[adj_matrix > 0]
                sigma = np.mean(non_zero_distances) if len(non_zero_distances) > 0 else 1.0
            
            edge_weights = np.exp(-(distance_matrix**2) / (2 * sigma**2))
            edge_weights = edge_weights * adj_matrix
            if self_loops:
                np.fill_diagonal(edge_weights, 1.0)
            return edge_weights
        
        else:
            raise ValueError(f"Unknown edge_weight type: {edge_weight}")


class DistanceGraphStrategy(GraphBuildingStrategy):
    """Strategy for distance-based graphs (full connectivity with weights)."""
    
    def build(
        self,
        distance_matrix: np.ndarray,
        station_ids: np.ndarray,
        edge_weight: str = 'gaussian',
        sigma: Optional[float] = None,
        self_loops: bool = True
    ) -> Dict:
        """Build distance-based graph (all nodes connected).
        
        Args:
            distance_matrix: Pairwise distance matrix.
            station_ids: Station ID array.
            edge_weight: Type of edge weight.
            sigma: Standard deviation for Gaussian weights.
            self_loops: Whether to include self-loops.
        
        Returns:
            Graph dictionary.
        """
        n = len(station_ids)
        adj_matrix = np.ones((n, n), dtype=float)
        
        if not self_loops:
            np.fill_diagonal(adj_matrix, 0)
        
        edge_weights = self._compute_edge_weights(
            distance_matrix, adj_matrix, edge_weight, sigma, self_loops
        )
        
        return {
            'adj_matrix': adj_matrix,
            'edge_weights': edge_weights,
            'station_ids': station_ids.copy(),
            'distance_matrix': distance_matrix.copy(),
            'graph_type': 'distance',
            'graph_param': None,
            'edge_weight_type': edge_weight,
        }


class RadiusGraphStrategy(GraphBuildingStrategy):
    """Strategy for radius-based graphs (within radius)."""
    
    def __init__(self, radius_km: float):
        """Initialize radius graph strategy.
        
        Args:
            radius_km: Maximum distance in km for edges.
        """
        self.radius_km = radius_km
    
    def build(
        self,
        distance_matrix: np.ndarray,
        station_ids: np.ndarray,
        edge_weight: str = 'gaussian',
        sigma: Optional[float] = None,
        self_loops: bool = True
    ) -> Dict:
        """Build radius-based graph.
        
        Args:
            distance_matrix: Pairwise distance matrix.
            station_ids: Station ID array.
            edge_weight: Type of edge weight.
            sigma: Standard deviation for Gaussian weights (default: radius_km / 3).
            self_loops: Whether to include self-loops.
        
        Returns:
            Graph dictionary.
        """
        n = len(station_ids)
        adj_matrix = (distance_matrix <= self.radius_km).astype(float)
        
        if not self_loops:
            np.fill_diagonal(adj_matrix, 0)
        
        # Default sigma for radius graphs
        if edge_weight == 'gaussian' and sigma is None:
            sigma = self.radius_km / 3.0
        
        edge_weights = self._compute_edge_weights(
            distance_matrix, adj_matrix, edge_weight, sigma, self_loops
        )
        
        return {
            'adj_matrix': adj_matrix,
            'edge_weights': edge_weights,
            'station_ids': station_ids.copy(),
            'distance_matrix': distance_matrix.copy(),
            'graph_type': 'radius',
            'graph_param': self.radius_km,
            'edge_weight_type': edge_weight,
        }


class KNNGraphStrategy(GraphBuildingStrategy):
    """Strategy for k-nearest neighbor graphs."""
    
    def __init__(self, k: int):
        """Initialize kNN graph strategy.
        
        Args:
            k: Number of nearest neighbors (excluding self).
        """
        self.k = k
    
    def build(
        self,
        distance_matrix: np.ndarray,
        station_ids: np.ndarray,
        edge_weight: str = 'gaussian',
        sigma: Optional[float] = None,
        self_loops: bool = True
    ) -> Dict:
        """Build k-nearest neighbor graph.
        
        Args:
            distance_matrix: Pairwise distance matrix.
            station_ids: Station ID array.
            edge_weight: Type of edge weight.
            sigma: Standard deviation for Gaussian weights (default: mean distance to k-th neighbor).
            self_loops: Whether to include self-loops.
        
        Returns:
            Graph dictionary.
        """
        n = len(station_ids)
        adj_matrix = np.zeros((n, n))
        
        # Find k nearest neighbors for each node
        for i in range(n):
            distances = distance_matrix[i, :].copy()
            if not self_loops:
                distances[i] = np.inf  # Exclude self
            
            # Get k+1 nearest (including self if self_loops=True)
            k_nearest = np.argsort(distances)[:self.k+1] if self_loops else np.argsort(distances)[:self.k]
            adj_matrix[i, k_nearest] = 1.0
        
        # Make symmetric (undirected graph)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        adj_matrix = (adj_matrix > 0).astype(float)
        
        # Default sigma for kNN graphs (mean distance to k-th neighbor)
        if edge_weight in ['gaussian', 'learnable'] and sigma is None:
            k_distances = []
            for i in range(n):
                distances = distance_matrix[i, :].copy()
                if not self_loops:
                    distances[i] = np.inf
                k_dist = np.sort(distances)[self.k]
                k_distances.append(k_dist)
            sigma = np.mean(k_distances) if k_distances else 1.0
        
        edge_weights = self._compute_edge_weights(
            distance_matrix, adj_matrix, edge_weight, sigma, self_loops
        )
        
        return {
            'adj_matrix': adj_matrix,
            'edge_weights': edge_weights,
            'station_ids': station_ids.copy(),
            'distance_matrix': distance_matrix.copy(),
            'graph_type': 'knn',
            'graph_param': self.k,
            'edge_weight_type': edge_weight,
        }


def create_strategy(graph_type: str, graph_param: Optional[float] = None) -> GraphBuildingStrategy:
    """Factory function to create graph building strategy.
    
    Args:
        graph_type: Type of graph ('distance', 'radius', 'knn').
        graph_param: Parameter for graph (radius_km for 'radius', k for 'knn').
    
    Returns:
        Graph building strategy instance.
    
    Raises:
        ValueError: If graph_type is unknown or graph_param is required but missing.
    """
    if graph_type == 'distance':
        return DistanceGraphStrategy()
    elif graph_type == 'radius':
        if graph_param is None:
            raise ValueError("graph_param (radius_km) required for 'radius' graph type")
        return RadiusGraphStrategy(radius_km=float(graph_param))
    elif graph_type == 'knn':
        if graph_param is None:
            raise ValueError("graph_param (k) required for 'knn' graph type")
        return KNNGraphStrategy(k=int(graph_param))
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}. Must be 'distance', 'radius', or 'knn'")

