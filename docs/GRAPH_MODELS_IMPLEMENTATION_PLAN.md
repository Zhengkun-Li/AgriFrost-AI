# 图神经网络模型实现计划 (Graph Neural Network Implementation Plan)

## 🎯 目标

实现 4 个图神经网络模型（方案 2）：
1. **DCRNN** (Diffusion Convolutional Recurrent Neural Network)
2. **ST-GCN** (Spatial-Temporal Graph Convolutional Network)
3. **GAT-LSTM** (Graph Attention Network + LSTM)
4. **GraphWaveNet** (Graph Convolution + WaveNet)

---

## 📁 目录结构

```
src/models/
├── graph/
│   ├── __init__.py
│   ├── base_graph_model.py      # 图模型基类（通用功能）
│   ├── graph_builder.py         # 图结构构建（距离矩阵、kNN等）
│   ├── dcrnn_model.py          # DCRNN 实现
│   ├── st_gcn_model.py         # ST-GCN 实现
│   ├── gat_lstm_model.py       # GAT-LSTM 实现
│   └── graphwavenet_model.py   # GraphWaveNet 实现
```

---

## 🔧 实现步骤

### Phase 1: 基础设施（1-2 天）

#### 1.1 创建图模型基类 (`base_graph_model.py`)
- [ ] 继承 `BaseModel`
- [ ] 实现图结构加载/保存
- [ ] 实现节点特征准备（Raw 变量 + 时间编码）
- [ ] 实现通用的训练逻辑（复用 LSTM 的训练流程）
- [ ] 实现 LOSO 评估支持

#### 1.2 图结构构建 (`graph_builder.py`)
- [ ] 基于距离的图（半径 R ∈ {25, 50, 75, 100} km）
- [ ] kNN 图（k ∈ {3, 5, 7}）
- [ ] 边权计算（距离衰减、高斯核、可学习）
- [ ] 图结构缓存（`data/interim/graph/{R|k}.pkl`）
- [ ] 支持风向门控（可选）

#### 1.3 数据准备
- [ ] 节点特征提取（Raw 变量 + 时间编码）
- [ ] 图数据加载器（Graph DataLoader）
- [ ] 时间序列图数据构建（支持多 horizon）

---

### Phase 2: DCRNN 实现（2-3 天）

#### 2.1 核心模块
- [ ] 扩散卷积层（Diffusion Convolution）
- [ ] RNN 层（LSTM/GRU）
- [ ] 多 horizon 预测头

#### 2.2 模型类
- [ ] `DCRNNModel` (PyTorch Module)
- [ ] `DCRNNForecastModel` (BaseModel wrapper)
- [ ] 实现 `fit`, `predict`, `predict_proba`

#### 2.3 训练集成
- [ ] 复用 LSTM 的训练逻辑（AMP, 早停, 校准等）
- [ ] 支持不平衡数据处理
- [ ] 支持概率校准

#### 2.4 测试与验证
- [ ] 单元测试
- [ ] 小规模数据测试
- [ ] 与 LSTM baseline 对比

---

### Phase 3: ST-GCN 实现（1-2 天）

#### 3.1 核心模块
- [ ] 空间图卷积层（Spatial Graph Convolution）
- [ ] 时间卷积层（Temporal Convolution）
- [ ] 残差连接

#### 3.2 模型类
- [ ] `STGCNModel` (PyTorch Module)
- [ ] `STGCNForecastModel` (BaseModel wrapper)
- [ ] 实现 `fit`, `predict`, `predict_proba`

#### 3.3 训练集成
- [ ] 复用训练逻辑
- [ ] 测试与验证

---

### Phase 4: GAT-LSTM 实现（2-3 天）

#### 4.1 核心模块
- [ ] 图注意力层（Graph Attention Layer）
- [ ] 多头注意力（Multi-head Attention）
- [ ] LSTM 层
- [ ] 注意力权重可视化（可选）

#### 4.2 模型类
- [ ] `GATLSTMModel` (PyTorch Module)
- [ ] `GATLSTMForecastModel` (BaseModel wrapper)
- [ ] 实现 `fit`, `predict`, `predict_proba`

#### 4.3 训练集成
- [ ] 复用训练逻辑
- [ ] 测试与验证

---

### Phase 5: GraphWaveNet 实现（3-4 天）

#### 5.1 核心模块
- [ ] 图卷积层（Graph Convolution）
- [ ] 扩张卷积层（Dilated Convolution）
- [ ] 残差连接
- [ ] 门控激活

#### 5.2 模型类
- [ ] `GraphWaveNetModel` (PyTorch Module)
- [ ] `GraphWaveNetForecastModel` (BaseModel wrapper)
- [ ] 实现 `fit`, `predict`, `predict_proba`

#### 5.3 训练集成
- [ ] 复用训练逻辑
- [ ] 内存优化（多尺度特征）
- [ ] 测试与验证

---

### Phase 6: 集成与测试（1-2 天）

#### 6.1 模型注册
- [ ] 更新 `src/training/model_config.py`
- [ ] 更新 `src/models/__init__.py`
- [ ] 更新 `src/models/graph/__init__.py`

#### 6.2 训练流程集成
- [ ] 更新 `src/training/model_trainer.py`（支持图模型）
- [ ] 更新训练脚本
- [ ] 测试完整训练流程

#### 6.3 文档与示例
- [ ] 更新 `MODEL_ROADMAP.md`
- [ ] 创建使用示例
- [ ] 更新 README

---

## 📋 技术细节

### 图结构定义

```python
# 图结构格式
graph = {
    'adj_matrix': np.ndarray,  # 邻接矩阵 (N, N)
    'edge_weights': np.ndarray,  # 边权 (可选)
    'node_features': np.ndarray,  # 节点特征 (N, F)
    'station_ids': np.ndarray,   # 站点 ID 映射
}
```

### 节点特征

```python
# Raw 变量 + 时间编码
node_features = [
    # 原始变量
    'Air Temp (C)',
    'Relative Humidity (%)',
    'Wind Speed (m/s)',
    'Wind Direction (deg)',
    'Solar Radiation (W/m²)',
    # 时间编码
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'month_sin', 'month_cos',
]
```

### 训练配置

```python
# 复用 LSTM 的训练配置
config = {
    'sequence_length': 24,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.0003,
    'batch_size': 32,
    'epochs': 120,
    'early_stopping': True,
    'patience': 20,
    'use_amp': True,
    'use_probability_calibration': True,
    # 图特定参数
    'graph_type': 'radius',  # 'radius' or 'knn'
    'graph_param': 50,  # R (km) or k
    'edge_weight': 'gaussian',  # 'gaussian', 'distance', 'learnable'
}
```

---

## 🔗 依赖库

### 必需
- `torch` >= 2.0.0
- `torch-geometric` >= 2.0.0 (图神经网络库)
- `numpy` >= 1.20.0
- `pandas` >= 1.3.0

### 可选
- `networkx` (图可视化)
- `matplotlib` (可视化)

---

## 📊 实现优先级

### 高优先级（必须）
1. ✅ **图结构构建** (`graph_builder.py`)
2. ✅ **图模型基类** (`base_graph_model.py`)
3. ✅ **DCRNN** (最适合霜冻预测)

### 中优先级（推荐）
4. ✅ **ST-GCN** (经典 baseline)
5. ✅ **GAT-LSTM** (注意力机制)

### 低优先级（时间允许）
6. ✅ **GraphWaveNet** (长期依赖)

---

## ⚠️ 注意事项

### 1. 数据格式
- 图模型需要特殊的数据格式（图结构 + 节点特征）
- 需要确保与现有数据管道兼容

### 2. 训练时间
- 图模型训练时间可能比序列模型长
- 需要优化图卷积计算（使用 GPU）

### 3. 内存占用
- 图结构需要额外内存
- GraphWaveNet 的多尺度特征可能占用大量内存

### 4. 超参数调优
- 图结构参数（R, k）需要敏感性实验
- 边权类型（距离衰减 vs 可学习）需要对比

---

## 📈 预期时间表

| Phase | 任务 | 预计时间 | 累计时间 |
|-------|------|----------|----------|
| Phase 1 | 基础设施 | 1-2 天 | 1-2 天 |
| Phase 2 | DCRNN | 2-3 天 | 3-5 天 |
| Phase 3 | ST-GCN | 1-2 天 | 4-7 天 |
| Phase 4 | GAT-LSTM | 2-3 天 | 6-10 天 |
| Phase 5 | GraphWaveNet | 3-4 天 | 9-14 天 |
| Phase 6 | 集成与测试 | 1-2 天 | **10-16 天** |

**总计**: 约 **2-3 周**（全职开发）

---

## 🎯 成功标准

### 功能完整性
- [ ] 4 个模型全部实现
- [ ] 与 BaseModel 接口兼容
- [ ] 支持完整训练流程
- [ ] 支持 LOSO 评估

### 性能要求
- [ ] 训练时间合理（每个 horizon < 2 小时）
- [ ] 内存占用可控（< 16 GB）
- [ ] GPU 利用率 > 50%

### 代码质量
- [ ] 代码结构清晰
- [ ] 文档完整
- [ ] 单元测试覆盖
- [ ] 与现有代码风格一致

---

## 📝 下一步行动

1. **立即开始**: Phase 1（基础设施）
2. **并行准备**: 研究 PyTorch Geometric 文档
3. **数据准备**: 准备图结构构建所需的数据（站点坐标、距离矩阵）

---

*创建时间: 2025-11-16*
*预计完成时间: 2025-12-01*

