# Loss Function Analysis for Frost Forecasting

## Current Loss Functions

### 1. LightGBM (Best Performing Model)

**Classification (Frost Prediction):**
- **Default Objective**: `binary` (not explicitly set in config)
- **Actual Loss Function**: Binary log-loss (cross-entropy)
  - Formula: `L = -[y * log(p) + (1-y) * log(1-p)]`
  - Where: `y` = true label (0 or 1), `p` = predicted probability

**Regression (Temperature Prediction):**
- **Default Objective**: `regression` (not explicitly set in config)  
- **Actual Loss Function**: L2 loss (MSE - Mean Squared Error)
  - Formula: `L = (y_pred - y_true)²`

**Current Configuration:**
- No explicit `objective` parameter set
- No `is_unbalance` parameter (handles class imbalance automatically)
- No `scale_pos_weight` parameter
- No class weights specified

### 2. Deep Learning Models (GRU, LSTM, TCN)

**Classification (Frost Prediction):**
- **Default**: `BCEWithLogitsLoss` with `pos_weight`
- **Alternative (Optional)**: `FocalLoss` (alpha=0.25, gamma=2.0) - currently disabled
- Formula (BCEWithLogitsLoss): `L = -[pos_weight * y * log(σ(z)) + (1-y) * log(1-σ(z))]`

**Regression (Temperature Prediction):**
- **Loss Function**: `MSELoss` (Mean Squared Error)

## Analysis: Are Current Loss Functions Good?

### ✅ **Strengths of Current Approach:**

1. **LightGBM's Default Binary Log-Loss:**
   - ✅ Good for probability calibration (well-calibrated probabilities)
   - ✅ LightGBM internally handles class imbalance better than other tree models
   - ✅ Works well with imbalanced data (0.87% positive rate)
   - ✅ Produces good ROC-AUC (0.9877-0.9972) and PR-AUC (0.4671-0.7242)

2. **Tree Models Handle Imbalance:**
   - LightGBM's gradient boosting naturally focuses on misclassified samples
   - Bootstrap sampling (subsample=0.8) helps balance the training signal
   - Feature importance helps identify important patterns

3. **Multi-task Learning Benefit:**
   - Temperature regression provides additional signal
   - Helps improve probability calibration (Brier Score < 0.005)

### ⚠️ **Potential Issues & Improvements:**

1. **No Explicit Class Weighting:**
   - **Current**: No `is_unbalance` or `scale_pos_weight` parameter
   - **Impact**: Model might not optimally focus on rare frost events
   - **Recommendation**: Add `is_unbalance=True` or calculate `scale_pos_weight`

2. **Cross-Entropy Loss Limitations:**
   - Equal penalty for false positives and false negatives
   - For frost forecasting, false negatives (missed frost) are much more costly
   - **Current result**: 67-85% recall, which is good but could be better

3. **No Cost-Sensitive Learning:**
   - Loss function doesn't reflect the actual cost structure:
     - Cost of FN (missed frost) >> Cost of FP (false alarm)
   - **Current approach**: Optimizes for balanced accuracy, not agricultural cost

## Recommendations

### Option 1: Add Class Weighting (Recommended - Easy to Implement)

```python
# In model_config.py, add to LightGBM classification config:
"is_unbalance": True,  # Automatic class weight balancing
# OR
"scale_pos_weight": ratio_of_negatives_to_positives,  # Manual weighting
```

**Expected Impact:**
- Better recall (more frost events captured)
- Slightly lower precision (more false alarms)
- Better alignment with agricultural needs

### Option 2: Use Focal Loss (Better for Extreme Imbalance)

LightGBM doesn't natively support Focal Loss, but you could:
- Use a custom objective function (advanced)
- Switch to deep learning models with Focal Loss enabled

**Focal Loss Formula:**
```
FL = -α(1-p)^γ * log(p)    for positive class
FL = -(1-α)p^γ * log(1-p)  for negative class
```

Where:
- α = 0.25 (weight for rare class)
- γ = 2.0 (focusing parameter)

### Option 3: Cost-Sensitive Learning (Best for Real Deployment)

Create a custom loss that reflects agricultural costs:

```python
# Custom loss: C_FP * FP + C_FN * FN
# Where C_FP = 10, C_FN = 1000 (for example)
```

This requires:
- Custom objective function for LightGBM
- Or use threshold optimization post-training (current approach with F2-score)

## Current Performance Assessment

**With Current Loss Functions:**
- ✅ ROC-AUC: 0.9877-0.9972 (Excellent discrimination)
- ✅ PR-AUC: 0.4671-0.7242 (Good for imbalanced data)
- ✅ Brier Score: < 0.005 (Excellent calibration)
- ✅ Recall: 69.7-84.8% with F2-optimized thresholds
- ✅ Temperature RMSE: 1.58-2.39°C (Excellent)

**Conclusion:**
The current loss functions work **reasonably well**, but there's room for improvement, especially for maximizing recall (minimizing missed frost events).

## Recommended Next Steps

1. **Quick Win**: Add `is_unbalance=True` to LightGBM config
2. **Better**: Calculate and set `scale_pos_weight` based on actual class ratio
3. **Best**: Implement cost-sensitive learning with domain-specific costs

---

**Note**: The current approach of optimizing thresholds post-training (using F2-score) partially addresses the cost-sensitivity, but incorporating it into the loss function during training could be more effective.

