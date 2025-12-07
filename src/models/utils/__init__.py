"""Utility modules for model training and monitoring.

**Features:**
- ProgressLogger: Event-based interface with handlers and Runner path management
- TrainingHistory: Advanced APIs with schema injection and DataFrame output
- CheckpointManager: Runner path management, standard naming (best.ckpt, last.ckpt)
- Stateless plotting: Use `src.visualization.plots.plot_training_curves()`
- ConfigValidator: Located at `src.config.schema.validator`
"""

from .training_history import TrainingHistory
from .checkpoint_manager import CheckpointManager
from .progress_logger import ProgressLogger
from .graph_builder import GraphBuilder, get_graph_cache_path

# TrainingCurvePlotter is now stateless in src.visualization.plots
# ConfigValidator is now in src.config.schema.validator

# Import graph strategies (optional)
try:
    from .graph_strategies import (
        GraphBuildingStrategy,
        DistanceGraphStrategy,
        RadiusGraphStrategy,
        KNNGraphStrategy,
        create_strategy
    )
    GRAPH_STRATEGIES_AVAILABLE = True
except ImportError:
    GRAPH_STRATEGIES_AVAILABLE = False

__all__ = [
    "ProgressLogger",
    "TrainingHistory",
    "CheckpointManager",
    "GraphBuilder",
    "get_graph_cache_path",
]

# Add graph strategies if available
if GRAPH_STRATEGIES_AVAILABLE:
    __all__.extend([
        "GraphBuildingStrategy",
        "DistanceGraphStrategy",
        "RadiusGraphStrategy",
        "KNNGraphStrategy",
        "create_strategy",
    ])

