# 时空建模方案 (Spatial-Temporal Modeling)

## 🎯 问题分析

### 当前模型的局限性

1. **只考虑时间序列（Temporal）**
   - ✅ LSTM/GRU 捕捉时间依赖
   - ✅ 特征工程包含 lag features
   - ❌ **没有利用空间信息**

2. **单站点独立预测**
   - 每个站点单独训练和预测
   - 没有利用相邻站点的信息
   - 忽略了站点之间的空间相关性

3. **空间特征利用不足**
   - 虽然特征工程包含 GPS 坐标、距离等
   - 但只是作为**静态特征**加入
   - 没有**动态建模**站点间的空间依赖

### 为什么时空建模很重要？

1. **气象数据的空间相关性**
   - 相邻站点的温度、湿度往往相似
   - 天气系统具有空间连续性
   - 区域性天气模式（如冷锋、暖锋）

2. **数据优势**
   - 18 个站点，GPS 坐标固定
   - 站点分布覆盖加州不同区域
   - 可以构建空间图（Graph）或空间网格

3. **潜在性能提升**
   - 利用空间信息可以提升预测精度
   - 特别对于 LOSO 评估（测试站点可以利用训练站点的空间信息）
   - 可以捕捉区域性的霜冻模式

---

## 🚀 时空建模方案

### 方案 1: Graph Neural Networks (GNN) ⭐⭐⭐⭐⭐

#### 1.1 **GraphSAGE / GCN (Graph Convolutional Network)**

**架构**：
```
输入: 每个站点的时间序列特征 (T, F)
      + 站点空间图 (Graph with edges based on distance)
输出: 每个站点的预测值
```

**优势**：
- ✅ 显式建模站点间的空间关系
- ✅ 可以学习空间传播模式
- ✅ 适合不规则的空间分布（18个站点）
- ✅ 在气象预测中表现优秀

**实现思路**：
1. 构建空间图：基于距离或 k-NN 连接站点
2. 每个节点（站点）包含时间序列特征
3. GNN 层聚合邻居节点的信息
4. 结合 LSTM/GRU 处理时间依赖

**代码结构**：
```python
# GraphSAGE-LSTM 混合模型
class GraphSAGELSTMModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers):
        # GraphSAGE layers for spatial aggregation
        self.graphsage = GraphSAGE(...)
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(...)
        # Output layer
        self.fc = nn.Linear(...)
    
    def forward(self, x, edge_index):
        # x: (batch, nodes, time_steps, features)
        # 1. Spatial aggregation via GraphSAGE
        spatial_features = self.graphsage(x, edge_index)
        # 2. Temporal modeling via LSTM
        temporal_features = self.lstm(spatial_features)
        # 3. Prediction
        return self.fc(temporal_features)
```

**实现难度**：⭐⭐⭐⭐  
**预期时间**：6-8 小时  
**推荐库**：PyTorch Geometric (PyG)

---

#### 1.2 **Spatial-Temporal Graph Convolutional Network (ST-GCN)**

**架构**：
- 同时建模时间和空间依赖
- 交替使用图卷积（空间）和时间卷积（时间）

**优势**：
- ✅ 专门为时空数据设计
- ✅ 在交通预测、气象预测中表现优秀
- ✅ 可以捕捉复杂的时空模式

**实现难度**：⭐⭐⭐⭐⭐  
**预期时间**：8-10 小时

---

### 方案 2: ConvLSTM / PredRNN ⭐⭐⭐⭐

#### 2.1 **ConvLSTM (Convolutional LSTM)**

**架构**：
```
输入: 空间网格化的数据 (H, W, T, F)
      - 将18个站点插值到规则网格
输出: 每个网格点的预测值
```

**优势**：
- ✅ 结合 CNN 和 LSTM 的优势
- ✅ 可以捕捉空间局部模式
- ✅ 实现相对简单

**劣势**：
- ❌ 需要将不规则站点插值到网格（可能引入误差）
- ❌ 18个站点可能网格太小

**实现难度**：⭐⭐⭐  
**预期时间**：4-5 小时

---

### 方案 3: 多站点特征工程 ⭐⭐⭐

#### 3.1 **邻域特征聚合**

**思路**：
- 为每个站点计算邻域站点的统计特征
- 作为额外特征加入现有模型

**实现**：
```python
# 为每个站点计算：
# 1. 最近3个站点的平均温度
# 2. 50km内站点的温度标准差
# 3. 空间梯度（温度随距离的变化率）
```

**优势**：
- ✅ 实现简单，无需新模型架构
- ✅ 可以快速验证空间信息的价值
- ✅ 可以与现有模型（LightGBM, LSTM）结合

**劣势**：
- ❌ 静态特征，不能动态学习空间依赖
- ❌ 需要手动设计聚合方式

**实现难度**：⭐⭐  
**预期时间**：2-3 小时

---

### 方案 4: Attention-based Spatial-Temporal Model ⭐⭐⭐⭐

#### 4.1 **Spatial-Temporal Transformer**

**架构**：
- 使用 Transformer 的注意力机制
- 同时关注时间维度和空间维度

**优势**：
- ✅ 可以学习站点间的动态依赖
- ✅ 注意力权重可解释（哪些站点更重要）
- ✅ 适合长序列预测

**实现难度**：⭐⭐⭐⭐  
**预期时间**：6-8 小时

---

## 📊 推荐实施顺序

### 阶段 1: 快速验证（1-2 周）

1. **多站点特征工程** ⭐⭐⭐
   - 实现邻域特征聚合
   - 验证空间信息的价值
   - 与现有模型结合测试

2. **ConvLSTM** ⭐⭐⭐
   - 如果网格化效果良好
   - 快速实现和测试

### 阶段 2: 深度建模（2-3 周）

3. **GraphSAGE-LSTM** ⭐⭐⭐⭐⭐
   - 最推荐的方案
   - 显式建模空间关系
   - 适合不规则站点分布

4. **Spatial-Temporal Transformer** ⭐⭐⭐⭐
   - 如果 GraphSAGE 效果好，可以进一步优化

### 阶段 3: 高级模型（3-4 周）

5. **ST-GCN** ⭐⭐⭐⭐⭐
   - 最先进的时空模型
   - 如果前面方案效果好，可以尝试

---

## 🛠️ 技术实现细节

### 1. 空间图构建

#### 方法 1: 基于距离的图
```python
def build_distance_graph(stations_metadata, max_distance_km=100):
    """
    构建空间图：如果两个站点距离 < max_distance_km，则连接
    """
    graph = {}
    for i, station_i in enumerate(stations):
        neighbors = []
        for j, station_j in enumerate(stations):
            if i != j:
                dist = haversine_distance(
                    station_i['lat'], station_i['lon'],
                    station_j['lat'], station_j['lon']
                )
                if dist < max_distance_km:
                    neighbors.append((j, dist))
        graph[i] = neighbors
    return graph
```

#### 方法 2: k-NN 图
```python
def build_knn_graph(stations_metadata, k=5):
    """
    每个站点连接最近的 k 个站点
    """
    # 使用 sklearn.neighbors.KNeighborsTransformer
    pass
```

### 2. 数据格式转换

#### 当前格式（单站点）：
```python
# X: (N, F) - N个时间点，F个特征
# y: (N,) - N个时间点的目标值
```

#### 时空格式（多站点）：
```python
# X: (N, S, T, F)
#   - N: batch size
#   - S: 站点数量 (18)
#   - T: 时间步长 (sequence_length)
#   - F: 特征数量 (175)

# y: (N, S) - 每个站点的预测值
```

### 3. 训练策略

#### 选项 1: 多站点联合训练
- 所有站点一起训练
- 共享空间图结构
- 每个站点有独立的输出

#### 选项 2: LOSO 兼容
- 训练时：使用所有站点
- 测试时：只预测测试站点
- 可以利用训练站点的空间信息

---

## 📈 预期效果

### 性能提升预期

1. **短期预测（3h, 6h）**
   - 空间信息价值较小（局部变化快）
   - 预期提升：2-5%

2. **中期预测（12h, 24h）**
   - 空间信息价值较大（天气系统传播）
   - 预期提升：5-10%

3. **LOSO 评估**
   - 空间信息价值最大（可以利用其他站点）
   - 预期提升：10-15%

### 计算成本

- **训练时间**：增加 2-3 倍（需要处理多站点数据）
- **内存使用**：增加 3-5 倍（需要存储空间图）
- **推理时间**：增加 1.5-2 倍

---

## 🎯 下一步行动

### 立即开始（推荐）

1. **实现多站点特征工程**
   - 快速验证空间信息的价值
   - 无需新模型架构
   - 可以与现有模型结合

2. **准备 GraphSAGE-LSTM 实现**
   - 设计数据加载器（多站点格式）
   - 实现空间图构建
   - 实现模型架构

### 需要的信息

1. **站点空间分布**
   - ✅ 已有 GPS 坐标
   - ✅ 已有站点元数据
   - 需要：计算站点间距离矩阵

2. **数据格式**
   - 需要：将单站点数据转换为多站点格式
   - 需要：处理不同站点的时间对齐

---

## 📝 总结

**你的观察非常正确！** 当前模型确实只考虑了时间序列，没有充分利用空间信息。时空建模是一个**非常有价值的改进方向**，特别是对于：

1. ✅ 中期和长期预测（12h, 24h）
2. ✅ LOSO 评估（可以利用其他站点的信息）
3. ✅ 捕捉区域性天气模式

**推荐方案**：
- **短期**：多站点特征工程（快速验证）
- **中期**：GraphSAGE-LSTM（最佳平衡）
- **长期**：ST-GCN（最先进）

---

**最后更新**: 2025-11-15  
**维护者**: Zhengkun LI

