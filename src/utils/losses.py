"""Loss functions for imbalanced classification.

This module contains specialized loss functions for handling imbalanced datasets,
particularly useful for binary classification with extreme class imbalance (<1% positive).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            alpha: Weighting factor for rare class (default: 0.25).
            gamma: Focusing parameter (default: 2.0). Higher gamma focuses more on hard examples.
            reduction: 'mean' or 'sum'.
        """
        super(FocalLoss, self).__init__()
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
        """
        # Ensure inputs are 1D
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute binary cross entropy
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t: probability of true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t: class weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted Binary Cross Entropy with Logits Loss.
    
    A wrapper around BCEWithLogitsLoss that automatically computes
    pos_weight based on class imbalance ratio.
    
    Args:
        pos_weight: Weight for positive class. If None, will be computed from targets.
        reduction: 'mean' or 'sum' (default: 'mean').
    """
    
    def __init__(self, pos_weight: torch.Tensor = None, reduction: str = 'mean'):
        """Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive class. If None, computed from targets.
            reduction: 'mean' or 'sum'.
        """
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.
        
        Args:
            inputs: Logits from model (batch_size,) or (batch_size, 1).
            targets: Binary targets (batch_size,) with values in [0, 1].
        
        Returns:
            Weighted BCE loss value.
        """
        # Ensure inputs are 1D
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction=self.reduction
        )

