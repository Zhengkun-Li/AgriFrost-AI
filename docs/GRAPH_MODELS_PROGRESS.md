# 图神经网络模型实现进度 (Graph Models Implementation Progress)

## ✅ Phase 1: 基础设施（已完成）

### 1. 图构建工具 (`src/models/utils/graph_builder.py`)
- ✅ 加载站点元数据
- ✅ 计算距离矩阵（Haversine 公式）
- ✅ 构建半径图（`build_radius_graph`）
- ✅ 构建 kNN 图（`build_knn_graph`）
- ✅ 支持多种边权类型（Gaussian, Distance, Binary, Learnable）
- ✅ 图结构保存/加载
- ✅ 缓存支持

**测试**: ✅ 通过（18 个站点，距离 13.22-458.41 km）

### 2. 图模型基类 (`src/models/graph/base_graph_model.py`)
- ✅ 继承 `BaseModel` 接口
- ✅ 图结构加载/保存
- ✅ 节点特征准备（Raw 变量 + 时间编码）
- ✅ 站点 ID 到节点索引映射
- ✅ 抽象方法定义

---

## 🚧 Phase 2: DCRNN 实现（进行中）

### 已完成
- ✅ `DiffusionConvolution` 层（扩散卷积）
- ✅ `DCRNNCell`（DCRNN 单元，结合扩散卷积和 GRU）
- ✅ `DCRNNModel`（完整的 DCRNN 模型）
- ✅ `DCRNNForecastModel`（BaseModel wrapper）
- ✅ 完整的训练逻辑（AMP, 早停, LR 调度）
- ✅ 不平衡数据处理（pos_weight）
- ✅ 概率校准支持
- ✅ 数据组织（按节点分组，创建序列）
- ✅ `predict` 和 `predict_proba` 方法

### 待优化
- [ ] 数据组织效率（当前按节点逐个处理，可优化为批量处理）
- [ ] 序列构建优化（预测时需要历史序列，当前实现较简单）
- [ ] 单元测试
- [ ] 小规模数据测试

### 代码结构
```python
src/models/graph/dcrnn_model.py
├── DiffusionConvolution      # 扩散卷积层
├── DCRNNCell                 # DCRNN 单元（扩散卷积 + GRU）
├── DCRNNModel                # DCRNN 模型（PyTorch Module）
├── GraphTimeSeriesDataset    # 图时间序列数据集（已定义但未使用）
└── DCRNNForecastModel        # DCRNN 预测模型（BaseModel wrapper）
```

---

## 📋 Phase 3-6: 待实现

### Phase 3: ST-GCN（待开始）
- [ ] 空间图卷积层
- [ ] 时间卷积层
- [ ] ST-GCN 模型
- [ ] ST-GCN Forecast Model

### Phase 4: GAT-LSTM（待开始）
- [ ] 图注意力层
- [ ] GAT-LSTM 模型
- [ ] GAT-LSTM Forecast Model

### Phase 5: GraphWaveNet（待开始）
- [ ] 图卷积层
- [ ] 扩张卷积层（WaveNet）
- [ ] GraphWaveNet 模型
- [ ] GraphWaveNet Forecast Model

### Phase 6: 集成与测试（待开始）
- [ ] 更新 `model_config.py`
- [ ] 更新 `model_trainer.py`
- [ ] 完整测试
- [ ] 文档更新

---

## 🎯 当前状态

### 已完成
- ✅ Phase 1: 基础设施（100%）
- 🚧 Phase 2: DCRNN（90% - 核心功能完成，待测试和优化）

### 下一步
1. **测试 DCRNN**：创建小规模测试验证基本功能
2. **优化数据组织**：改进预测时的数据组织方式
3. **继续实现**：Phase 3 (ST-GCN)

---

## 📝 注意事项

### DCRNN 实现说明

1. **数据组织**：
   - 训练时：按节点分组，创建时间序列
   - 预测时：需要为每个样本构建历史序列（当前实现较简单）

2. **图输入格式**：
   - 模型输入：`(batch_size, seq_len, num_nodes, n_features)`
   - 每个样本的特征被复制到所有节点，但只使用对应节点的输出

3. **优化建议**：
   - 预测时可以维护一个历史缓冲区，避免重复计算
   - 可以批量处理同一节点的多个样本

---

## 🔧 已知问题

1. **预测效率**：当前预测时逐个样本处理，效率较低
   - **解决方案**：维护历史缓冲区，批量处理

2. **序列构建**：预测时需要历史序列，当前实现假设数据已按时间排序
   - **解决方案**：在 `fit` 时保存每个节点的历史数据

---

*最后更新: 2025-11-16*
*当前进度: Phase 1 ✅ | Phase 2 🚧 90%*

