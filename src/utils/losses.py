"""Loss functions for imbalanced classification.

This module contains specialized loss functions for handling imbalanced datasets,
particularly useful for binary classification with extreme class imbalance (<1% positive).
"""

import logging
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples. It is particularly effective for datasets
    with extreme class imbalance (<1% positive samples).
    
    Paper: https://arxiv.org/abs/1708.02002
    
    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where:
        - p_t is the probability of the true class
        - alpha is the weighting factor for rare class (default: 0.25)
        - gamma is the focusing parameter (default: 2.0)
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25).
            Higher alpha gives more weight to positive class.
        gamma: Focusing parameter (default: 2.0).
            Higher gamma focuses more on hard examples.
        reduction: 'mean' or 'sum' (default: 'mean').
    
    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = model(x)  # Raw logits from model
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 0.25). Must be in [0, 1].
            gamma: Focusing parameter (default: 2.0). Higher gamma focuses more on hard examples. Must be >= 0.
            reduction: 'mean' or 'sum'.
        
        Raises:
            ValueError: If alpha not in [0, 1] or gamma < 0 or reduction not in ['mean', 'sum'].
        """
        super(FocalLoss, self).__init__()
        
        # Input validation
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Logits from model (batch_size,) or (batch_size, 1).
            targets: Binary targets (batch_size,) with values in [0, 1].
        
        Returns:
            Focal loss value (scalar if reduction='mean' or 'sum').
        
        Raises:
            ValueError: If inputs and targets have incompatible shapes.
        """
        # Ensure inputs are 1D (use squeeze(-1) to avoid removing batch dimension)
        if inputs.dim() > 1:
            inputs = inputs.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
        
        # Input validation
        if inputs.shape != targets.shape:
            raise ValueError(
                f"inputs and targets must have the same shape after squeezing. "
                f"Got {inputs.shape} and {targets.shape}"
            )
        
        if len(inputs) == 0:
            raise ValueError("Batch size cannot be zero")
        
        # Validate targets are binary (0 or 1) for correct p_t calculation
        unique_targets = torch.unique(targets)
        if not torch.all((unique_targets == 0) | (unique_targets == 1)):
            _logger.warning(
                f"Targets contain non-binary values: {unique_targets.tolist()}. "
                f"FocalLoss is designed for binary classification. Clipping targets to [0, 1]."
            )
            targets = torch.clamp(targets, 0.0, 1.0)
        
        # Compute binary cross entropy (this internally computes sigmoid, so no need to duplicate)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute probabilities and focal components with no_grad to avoid unnecessary gradients
        # (BCE already computes sigmoid internally, but we need probs for p_t calculation)
        with torch.no_grad():
            probs = torch.sigmoid(inputs)
            # Clamp probabilities to avoid numerical instability in p_t calculation
            eps = 1e-7
            probs = torch.clamp(probs, eps, 1.0 - eps)
            
            # Compute p_t: probability of true class
            # For binary classification, p_t = prob if target=1, else (1-prob) if target=0
            # This assumes targets are in [0, 1] (binary or soft labels)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            
            # Compute alpha_t: class weight
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            
            # Compute focal weight: (1 - p_t)^gamma
            focal_weight = (1 - p_t) ** self.gamma
        
        # Compute focal loss (focal_weight and alpha_t are detached, so gradients flow through bce)
        focal_loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted Binary Cross Entropy with Logits Loss.
    
    A wrapper around BCEWithLogitsLoss that supports class weighting.
    If pos_weight is None, it will be automatically computed from targets
    in each forward pass based on class imbalance ratio.
    
    Args:
        pos_weight: Weight for positive class. If None, computed from targets in forward().
        reduction: 'mean' or 'sum' (default: 'mean').
    
    Example:
        >>> criterion = WeightedBCEWithLogitsLoss(pos_weight=None)  # Auto-compute
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(self, pos_weight: torch.Tensor = None, reduction: str = 'mean'):
        """Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive class. If None, will be computed from targets.
            reduction: 'mean' or 'sum'.
        
        Raises:
            ValueError: If reduction not in ['mean', 'sum'] or pos_weight is invalid.
        """
        super(WeightedBCEWithLogitsLoss, self).__init__()
        
        # Input validation
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
        
        if pos_weight is not None and (not isinstance(pos_weight, torch.Tensor) or pos_weight.item() <= 0):
            raise ValueError(f"pos_weight must be a positive Tensor, got {pos_weight}")
        
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.
        
        Args:
            inputs: Logits from model (batch_size,) or (batch_size, 1).
            targets: Binary targets (batch_size,) with values in [0, 1].
        
        Returns:
            Weighted BCE loss value.
        
        Raises:
            ValueError: If inputs and targets have incompatible shapes.
        """
        # Ensure inputs are 1D (use squeeze(-1) to avoid removing batch dimension)
        if inputs.dim() > 1:
            inputs = inputs.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
        
        # Input validation
        if inputs.shape != targets.shape:
            raise ValueError(
                f"inputs and targets must have the same shape after squeezing. "
                f"Got {inputs.shape} and {targets.shape}"
            )
        
        if len(inputs) == 0:
            raise ValueError("Batch size cannot be zero")
        
        # Auto-compute pos_weight from targets if not provided
        pos_weight = self.pos_weight
        if pos_weight is None:
            with torch.no_grad():
                pos = (targets > 0.5).float().sum()
                neg = (targets <= 0.5).float().sum()
                # Avoid division by zero: if no positive samples, use weight of 1.0
                if pos > 0:
                    pos_weight = torch.tensor([neg / pos], device=inputs.device, dtype=inputs.dtype)
                else:
                    _logger.warning("No positive samples found in batch. Using pos_weight=1.0")
                    pos_weight = torch.tensor([1.0], device=inputs.device, dtype=inputs.dtype)
        
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pos_weight, reduction=self.reduction
        )

