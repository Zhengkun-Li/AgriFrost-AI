# Graph Neural Network Models

This directory contains graph neural network models for spatial-temporal frost forecasting.

## Models

### 1. DCRNN (Diffusion Convolutional Recurrent Neural Network)
- **Status**: 🚧 In Progress
- **Description**: Combines diffusion convolution for spatial modeling with RNN for temporal modeling
- **Best for**: Temperature diffusion patterns, multi-horizon prediction

### 2. ST-GCN (Spatial-Temporal Graph Convolutional Network)
- **Status**: 🚧 In Progress
- **Description**: Classic spatial-temporal graph model with separate spatial and temporal convolutions
- **Best for**: Baseline comparison, stable performance

### 3. GAT-LSTM (Graph Attention Network + LSTM)
- **Status**: 🚧 In Progress
- **Description**: Graph attention mechanism for dynamic spatial relationships + LSTM for temporal modeling
- **Best for**: Complex spatial relationships (e.g., wind direction effects)

### 4. GraphWaveNet (Graph Convolution + WaveNet)
- **Status**: 🚧 In Progress
- **Description**: Graph convolution + dilated convolutions for long-term dependencies
- **Best for**: Long-horizon prediction (24h)

## Implementation Status

- [ ] Phase 1: Infrastructure (graph builder, base class)
- [ ] Phase 2: DCRNN
- [ ] Phase 3: ST-GCN
- [ ] Phase 4: GAT-LSTM
- [ ] Phase 5: GraphWaveNet
- [ ] Phase 6: Integration & Testing

## Dependencies

- `torch` >= 2.0.0
- `torch-geometric` >= 2.0.0
- `numpy` >= 1.20.0
- `pandas` >= 1.3.0

## Usage

See `docs/GRAPH_MODELS_IMPLEMENTATION_PLAN.md` for detailed implementation plan.

