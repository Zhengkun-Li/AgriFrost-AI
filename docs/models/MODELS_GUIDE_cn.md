# AgriFrost-AI Model Guide

<div align="center">

<img src="logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

## Frost Risk Forecasting Models

This document provides a comprehensive overview of all models implemented in the **AgriFrost-AI** project, including their key principles, advantages, disadvantages, and use cases.

## Table of Contents

1. [Machine Learning Models](#machine-learning-models)
2. [Deep Learning Models](#deep-learning-models)
3. [Graph Neural Network Models](#graph-neural-network-models)
4. [Traditional Models](#traditional-models)
5. [Model Selection Guide](#model-selection-guide)

---

## Machine Learning Models

### 1. LightGBM (Light Gradient Boosting Machine)

**Key Principle:**
- Gradient boosting framework using tree-based learning algorithms
- Uses histogram-based algorithm and leaf-wise tree growth strategy
- Handles categorical features automatically via optimal split finding

**Advantages:**
- ⚡ **Fast training**: Histogram-based algorithm significantly reduces training time
- 💾 **Memory efficient**: Lower memory consumption compared to XGBoost
- 📊 **Handles large datasets**: Efficient for datasets with millions of rows
- 🎯 **Good accuracy**: Competitive performance with careful tuning
- 🔢 **Built-in feature importance**: Provides feature importance scores
- 🔄 **Handles missing values**: Automatically handles missing data

**Disadvantages:**
- ⚠️ **Sensitive to overfitting**: May overfit on small datasets without proper regularization
- 🎲 **Parameter tuning**: Requires careful hyperparameter tuning for optimal performance
- 📈 **Limited interpretability**: Tree ensembles are less interpretable than linear models

**Use Cases:**
- Default choice for most forecasting tasks (Matrix Cells A, B, C, D)
- Best for tabular data with mixed feature types
- When training speed is important

---

### 2. XGBoost (Extreme Gradient Boosting)

**Key Principle:**
- Gradient boosting framework with level-wise tree growth
- Uses second-order gradients for more accurate optimization
- Implements parallel tree construction with regularization

**Advantages:**
- 🏆 **High accuracy**: Often achieves state-of-the-art performance
- 🛡️ **Regularization**: Built-in L1/L2 regularization reduces overfitting
- 🔧 **Flexibility**: Supports multiple objective functions and evaluation metrics
- 📦 **Robust**: Handles missing values and outliers well
- 🌐 **Cross-platform**: Widely used and well-documented

**Disadvantages:**
- ⏱️ **Slower training**: More computationally expensive than LightGBM
- 💾 **Memory usage**: Higher memory consumption
- 🎯 **Hyperparameter sensitivity**: Many parameters to tune

**Use Cases:**
- When maximum accuracy is the priority
- Medium to large datasets
- Competitions and benchmarks

---

### 3. CatBoost (Categorical Boosting)

**Key Principle:**
- Gradient boosting specifically designed for categorical features
- Uses ordered boosting to reduce overfitting
- Implements symmetric tree structure for fast inference

**Advantages:**
- 🏷️ **Categorical feature handling**: Excellent for datasets with many categorical variables
- 🛡️ **Less overfitting**: Ordered boosting algorithm reduces overfitting
- ⚙️ **Fewer hyperparameters**: Good default settings, requires less tuning
- 🚀 **Fast inference**: Optimized for prediction speed
- 📊 **Feature importance**: Provides detailed feature importance analysis

**Disadvantages:**
- ⏱️ **Training time**: Can be slower than LightGBM for large datasets
- 💾 **Memory**: Higher memory usage than LightGBM
- 🎯 **Limited flexibility**: Less customizable than XGBoost

**Use Cases:**
- Datasets with many categorical features
- When default parameters work well
- Production systems requiring fast inference

---

### 4. Random Forest

**Key Principle:**
- Ensemble of decision trees trained on random subsets of data (bootstrap sampling)
- Uses random feature selection for each split
- Predictions made by majority voting (classification) or averaging (regression)

**Advantages:**
- 🛡️ **Robust**: Resistant to overfitting due to ensemble averaging
- 🔢 **Feature importance**: Provides feature importance scores
- ⚡ **Fast training**: Can be parallelized easily
- 🎲 **Handles missing values**: Can handle missing data
- 📊 **Non-parametric**: No assumptions about data distribution

**Disadvantages:**
- 🐘 **Memory usage**: Stores all trees, requires significant memory
- ⏱️ **Inference time**: Slower prediction than gradient boosting
- 📈 **Less accurate**: Generally lower accuracy than gradient boosting methods
- 🔍 **Limited interpretability**: Individual tree interpretable, ensemble is not

**Use Cases:**
- Baseline model for comparison
- When interpretability (individual trees) is needed
- Small to medium datasets

---

### 5. ExtraTrees (Extremely Randomized Trees)

**Key Principle:**
- Similar to Random Forest but uses random thresholds for splits
- More randomization leads to lower variance but higher bias
- Faster training than Random Forest

**Advantages:**
- ⚡ **Very fast training**: Faster than Random Forest
- 🛡️ **Reduced overfitting**: Extra randomization helps prevent overfitting
- 📊 **Good for noisy data**: More robust to noisy datasets
- 🔧 **Fewer parameters**: Simpler to tune than gradient boosting

**Disadvantages:**
- 📉 **Lower accuracy**: Generally less accurate than gradient boosting
- 🔍 **Less interpretable**: Similar to Random Forest in interpretability
- 💾 **Memory**: Similar memory requirements to Random Forest

**Use Cases:**
- Quick baseline models
- Very noisy datasets
- When training speed is critical

---

### 6. Linear Regression

**Key Principle:**
- Models relationship between features and target using linear combination
- Minimizes sum of squared errors (least squares)
- Assumes linear relationship between features and target

**Advantages:**
- 📖 **Highly interpretable**: Coefficients directly indicate feature importance
- ⚡ **Fast**: Very fast training and prediction
- 💾 **Low memory**: Minimal memory requirements
- 📊 **Baseline**: Good baseline for comparison
- 🔍 **Feature analysis**: Easy to understand feature contributions

**Disadvantages:**
- 📉 **Limited capacity**: Cannot capture non-linear relationships
- 🎯 **Assumptions**: Assumes linearity and homoscedasticity
- 📈 **Outliers**: Sensitive to outliers
- 🔢 **Feature engineering**: Requires careful feature engineering

**Use Cases:**
- Baseline model
- Simple interpretable models
- When linear relationships are sufficient

---

### 7. Persistence Model

**Key Principle:**
- Simple baseline that predicts future values using current values
- For frost forecasting: predicts frost risk based on current temperature
- Assumes conditions remain constant (persistent)

**Advantages:**
- 🎯 **Simple**: Trivial to implement and understand
- ⚡ **Fast**: Instant prediction
- 📊 **Baseline**: Essential baseline for comparison
- 💾 **No memory**: No storage requirements

**Disadvantages:**
- 📉 **Poor accuracy**: Very limited predictive power
- ⏱️ **Time horizon**: Accuracy decreases rapidly with longer horizons
- 🎲 **No learning**: Cannot learn from historical patterns

**Use Cases:**
- Baseline comparison
- Short-term forecasts (1-3 hours)
- Benchmarking other models

---

### 8. Ensemble Model

**Key Principle:**
- Combines predictions from multiple models
- Uses weighted averaging or voting to make final predictions
- Leverages strengths of different models

**Advantages:**
- 🏆 **Higher accuracy**: Often outperforms individual models
- 🛡️ **Robust**: Reduces variance and model-specific errors
- 🔧 **Flexible**: Can combine different model types
- 📊 **Stable**: More stable predictions than single models

**Disadvantages:**
- ⏱️ **Training time**: Requires training multiple models
- 💾 **Memory**: Stores multiple models
- 🎯 **Complexity**: More complex to tune and maintain
- 📈 **Diminishing returns**: Improvements may be marginal

**Use Cases:**
- Final production models
- When maximum accuracy is needed
- Competitions and evaluations

---

## Deep Learning Models

### 1. LSTM (Long Short-Term Memory)

**Key Principle:**
- Recurrent neural network with memory cells that can store information for long periods
- Uses gates (forget, input, output) to control information flow
- Designed to capture long-term temporal dependencies

**Advantages:**
- ⏱️ **Temporal modeling**: Excellent at capturing temporal patterns and dependencies
- 🧠 **Long-term memory**: Can remember information for many time steps
- 📊 **Sequence data**: Naturally handles sequential/time-series data
- 🎯 **Non-linear**: Can capture complex non-linear relationships

**Disadvantages:**
- ⏱️ **Training time**: Slow training, especially on CPU
- 💾 **Memory**: High memory usage during training
- 🎯 **Hyperparameter tuning**: Many hyperparameters to tune
- 📈 **Gradient issues**: Can suffer from vanishing/exploding gradients

**Use Cases:**
- Matrix Cell A, B (single-station forecasting)
- When temporal patterns are important
- Time series with long-term dependencies

---

### 2. GRU (Gated Recurrent Unit)

**Key Principle:**
- Simplified version of LSTM with fewer parameters
- Uses only two gates (reset and update) instead of three
- Designed to be faster while maintaining similar performance

**Advantages:**
- ⚡ **Faster training**: Faster than LSTM due to simpler architecture
- 💾 **Lower memory**: Fewer parameters, less memory usage
- 🎯 **Similar performance**: Often achieves similar accuracy to LSTM
- 🔧 **Easier tuning**: Fewer hyperparameters to tune

**Disadvantages:**
- 📉 **Less capacity**: May have slightly less modeling capacity than LSTM
- ⏱️ **Still slow**: Still slower than tree-based models
- 📈 **Limited memory**: May struggle with very long sequences

**Use Cases:**
- Alternative to LSTM when speed is important
- Similar use cases to LSTM
- When computational resources are limited

---

### 3. LSTM Multitask

**Key Principle:**
- Extended LSTM that predicts multiple targets simultaneously
- Shares lower layers for common feature extraction
- Separate output heads for different tasks (e.g., temperature and frost risk)

**Advantages:**
- 🎯 **Multi-target**: Can predict multiple related targets simultaneously
- 📊 **Feature sharing**: Shared features may improve performance
- 🔄 **Efficient**: More efficient than training separate models
- 🧠 **Leverages relationships**: Can leverage relationships between targets

**Disadvantages:**
- 🎯 **Task balancing**: Requires careful balancing of different tasks
- 📈 **Complexity**: More complex architecture
- ⏱️ **Training time**: Longer training time
- 🔧 **Tuning**: More hyperparameters to tune

**Use Cases:**
- When predicting multiple related targets (temperature + frost risk)
- Matrix Cells with multiple outputs
- Tasks with shared underlying patterns

---

### 4. TCN (Temporal Convolutional Network)

**Key Principle:**
- Uses dilated convolutions to capture temporal patterns
- Causal convolutions ensure no future information leakage
- Residual connections for deeper networks

**Advantages:**
- ⚡ **Parallelizable**: Can process sequences in parallel (unlike RNNs)
- ⏱️ **Fast training**: Faster training than LSTM/GRU
- 🎯 **Long-range dependencies**: Dilated convolutions capture long-term patterns
- 💾 **Memory efficient**: More memory efficient than RNNs

**Disadvantages:**
- 📈 **Limited history**: Receptive field size limits historical context
- 🎯 **Hyperparameters**: Requires tuning dilation rates and kernel sizes
- 📊 **Less common**: Less commonly used than LSTM, fewer resources

**Use Cases:**
- Alternative to LSTM/GRU when speed is important
- When parallelization is needed
- Long sequences with temporal patterns

---

## Graph Neural Network Models

### 1. DCRNN (Diffusion Convolutional Recurrent Neural Network)

**Key Principle:**
- Combines graph convolution with RNN (typically GRU)
- Uses diffusion convolution to model spatial dependencies
- Captures both spatial and temporal patterns simultaneously

**Advantages:**
- 🌐 **Spatial-temporal**: Models both spatial and temporal dependencies
- 🔗 **Graph structure**: Leverages station relationships via graph
- 📊 **Multi-station**: Ideal for multi-station forecasting (Matrix Cells C, D, E)
- 🎯 **Spatial correlation**: Captures spatial correlations between stations

**Disadvantages:**
- ⏱️ **Training time**: Very slow training, computationally expensive
- 💾 **Memory**: High memory requirements
- 🔧 **Graph construction**: Requires careful graph construction
- 🎯 **Complex tuning**: Many hyperparameters to tune

**Use Cases:**
- Matrix Cell C, D (multi-station forecasting)
- When spatial relationships are important
- Networks of weather stations

---

### 2. GAT-LSTM (Graph Attention LSTM)

**Key Principle:**
- Uses graph attention mechanism to weight neighbor stations
- Combines attention-based graph convolution with LSTM
- Attention mechanism learns which stations are most relevant

**Advantages:**
- 🎯 **Attention mechanism**: Learns to focus on important stations
- 📊 **Interpretable**: Attention weights provide interpretability
- 🌐 **Spatial-temporal**: Captures both spatial and temporal patterns
- 🔗 **Adaptive**: Adaptively weights different stations

**Disadvantages:**
- ⏱️ **Training time**: Slow training due to attention computation
- 💾 **Memory**: Higher memory usage than simple graph models
- 🎯 **Complexity**: More complex than DCRNN
- 📈 **Tuning**: Requires careful tuning of attention mechanisms

**Use Cases:**
- Matrix Cell C, D, E (multi-station with attention)
- When understanding station importance is valuable
- Heterogeneous station networks

---

### 3. GraphWaveNet

**Key Principle:**
- Uses adaptive graph learning to learn station relationships automatically
- Combines graph convolution with dilated convolutions (from TCN)
- Can adapt graph structure during training

**Advantages:**
- 🔗 **Adaptive graphs**: Learns optimal graph structure automatically
- ⚡ **Fast**: Efficient graph convolutions
- 📊 **Flexible**: Can handle different graph structures
- 🎯 **No manual graph**: Doesn't require pre-defined graph

**Disadvantages:**
- 🎯 **Training complexity**: More complex training process
- 📈 **Hyperparameters**: Additional parameters for graph learning
- 💾 **Memory**: Still requires significant memory
- 📚 **Less mature**: Less widely used than DCRNN

**Use Cases:**
- Matrix Cell E (graph neural networks)
- When graph structure is uncertain
- When adaptive learning is needed

---

### 4. ST-GCN (Spatial-Temporal Graph Convolutional Network)

**Key Principle:**
- Uses spatial graph convolution for station relationships
- Uses temporal convolution for time dependencies
- Separates spatial and temporal modeling

**Advantages:**
- ⚡ **Efficient**: More efficient than RNN-based graph models
- 🔧 **Modular**: Separate spatial and temporal modules
- 📊 **Parallelizable**: Temporal convolutions can be parallelized
- 🎯 **Good performance**: Strong performance on spatial-temporal tasks

**Disadvantages:**
- 📈 **Limited temporal**: May have limited temporal modeling compared to RNNs
- 🎯 **Graph dependency**: Requires well-defined graph structure
- 💾 **Memory**: Still requires significant memory
- 🔧 **Tuning**: Requires tuning both spatial and temporal modules

**Use Cases:**
- Matrix Cell E (graph neural networks)
- When spatial and temporal patterns are separable
- Large station networks

---

## Traditional Models

### 1. Prophet

**Key Principle:**
- Decomposes time series into trend, seasonality, and holidays
- Uses additive or multiplicative models
- Designed for business forecasting with automatic seasonality detection

**Advantages:**
- 📊 **Interpretable**: Clear decomposition of components
- 🎯 **Robust**: Handles missing data and outliers well
- 🔧 **Automatic**: Automatic seasonality and holiday detection
- 📈 **Trend handling**: Good at capturing trends

**Disadvantages:**
- 📉 **Limited accuracy**: Generally lower accuracy than ML/DL models
- ⏱️ **Training time**: Can be slow for large datasets
- 🎯 **Feature engineering**: Limited ability to use external features
- 📊 **Assumptions**: Makes assumptions about data structure

**Use Cases:**
- Baseline comparison
- When interpretability is important
- Time series with clear seasonality

---

## Model Selection Guide

### By Matrix Cell

| Matrix Cell | Description | Recommended Models |
|-------------|-------------|-------------------|
| **A** | Raw, Single-station | LightGBM, LSTM, GRU |
| **B** | Feature-engineered, Single-station | LightGBM, XGBoost, LSTM Multitask |
| **C** | Raw, Multi-station | LightGBM, DCRNN, ST-GCN |
| **D** | Feature-engineered, Multi-station | LightGBM, XGBoost, GAT-LSTM |
| **E** | Graph Neural Networks | DCRNN, GAT-LSTM, GraphWaveNet, ST-GCN |

### By Forecast Horizon

- **Short-term (3h, 6h)**: LightGBM, LSTM, GRU
- **Medium-term (12h)**: LightGBM, XGBoost, LSTM Multitask
- **Long-term (24h)**: XGBoost, DCRNN, Ensemble models

### By Data Size

- **Small datasets (< 100K rows)**: Random Forest, XGBoost
- **Medium datasets (100K - 1M rows)**: LightGBM, XGBoost, LSTM
- **Large datasets (> 1M rows)**: LightGBM, TCN, Graph models

### By Requirements

- **Speed priority**: LightGBM, TCN, ExtraTrees
- **Accuracy priority**: XGBoost, Ensemble, DCRNN
- **Interpretability**: Linear Regression, Random Forest, Prophet
- **Multi-station**: DCRNN, GAT-LSTM, ST-GCN

---

## Summary

This project implements a comprehensive suite of models covering:

- **8 Machine Learning models**: From simple baselines to advanced gradient boosting
- **4 Deep Learning models**: Various RNN architectures and TCN
- **4 Graph Neural Network models**: For multi-station spatial-temporal forecasting
- **1 Traditional model**: Prophet for baseline comparison

Each model has its strengths and is suited for different scenarios. The choice of model should depend on:

1. **Matrix cell** (A, B, C, D, or E)
2. **Forecast horizon** (3h, 6h, 12h, 24h)
3. **Data characteristics** (size, features, station count)
4. **Requirements** (speed, accuracy, interpretability)

For most production use cases, **LightGBM** serves as a reliable default choice due to its balance of speed, accuracy, and ease of use. For multi-station scenarios requiring spatial modeling, **DCRNN** or **GAT-LSTM** are recommended.

