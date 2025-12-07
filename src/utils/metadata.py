"""
Experiment metadata management.

⚠️ Circular Dependency Warning: This module must remain pure, only depends on standard library!

DO NOT import internal modules (src.training, src.evaluation, etc.),
to avoid circular dependencies. metadata must be the bottommost utility module.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime
from pathlib import Path
import json


@dataclass
class ExperimentMetadata:
    """Unified experiment metadata (required).
    
    ⚠️ All training runs must create and write this metadata to run_metadata.json.
    
    For legacy experiments / non-2×2+1 special runs:
    - Core fields (matrix_cell, track, etc.) can be None
    - Evaluation treats them uniformly as "non-standard experiments" (legacy runs)
    - Do not participate in matrix summary, but can participate in regular compare
    
    Type level allows Optional, but CLI level enforces required fields (new experiments).
    """
    matrix_cell: Optional[str] = None  # A/B/C/D/E (None = legacy run)
    track: Optional[str] = None  # raw/top175_features/... (None = legacy run)
    model_name: str = ""
    horizon_h: int = 0  # 3, 6, 12, 24
    radius_km: Optional[float] = None  # for C/D
    knn_k: Optional[int] = None  # for E
    training_scope: str = "full_training"
    created_at: Optional[str] = None  # ISO format timestamp
    
    def __post_init__(self):
        """Automatically add creation timestamp."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def is_legacy_run(self) -> bool:
        """Check if this is a legacy run (non-standard 2×2+1 experiment).
        
        Returns:
            True if matrix_cell or track is None
        """
        return self.matrix_cell is None or self.track is None
    
    def is_standard_run(self) -> bool:
        """Check if this is a standard 2×2+1 experiment.
        
        Returns:
            True if matrix_cell and track are both not None
        """
        return not self.is_legacy_run()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'matrix_cell': self.matrix_cell,
            'track': self.track,
            'model_name': self.model_name,
            'horizon_h': self.horizon_h,
            'radius_km': self.radius_km,
            'knn_k': self.knn_k,
            'training_scope': self.training_scope,
            'created_at': self.created_at,
            'is_legacy_run': self.is_legacy_run(),  # Convenient for evaluation
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentMetadata':
        """Create from dictionary."""
        # Remove helper fields if present
        d = {k: v for k, v in d.items() if k != 'is_legacy_run'}
        return cls(**d)
    
    def save(self, output_dir: Path) -> Path:
        """Save to run_metadata.json (required).
        
        Args:
            output_dir: Output directory
        
        Returns:
            Path to run_metadata.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "run_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        return metadata_path
    
    @classmethod
    def load(cls, metadata_path: Path) -> 'ExperimentMetadata':
        """Load from run_metadata.json.
        
        Args:
            metadata_path: Path to run_metadata.json
        
        Returns:
            ExperimentMetadata instance
        
        Raises:
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If JSON format is invalid
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return cls.from_dict(d)

