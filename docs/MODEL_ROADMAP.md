## 🌡 Frost Risk Forecast — 2×2+1 模型框架（统一评测与报告）

本项目的所有模型与实验，统一纳入如下框架，确保"输入形态（Raw vs Feature Engineering）× 空间范围（单站 vs 多站）"的公平可比性与可复现性。

> **框架说明**：核心是 2×2 矩阵（A/B/C/D），E 作为扩展（图神经网络端到端学习空间关系）。

- 矩阵定义：
  - **A**：Raw-only + 单站（原始变量，无特征工程）
  - **B**：Feature Engineering(175) + 单站（当前最强基线）
  - **C**：Raw-only + 多站（先做空间邻域聚合，再按 A 轨，不走 FE）
    - 先对邻站原始变量做聚合（mean/max/min/std/median、距离加权均值、gradient、range 等）
    - 使用与 A 相同的 Raw 模型（LightGBM/raw、LSTM/GRU/TCN raw 序列等）
    - 输入为"原始变量 + 空间聚合特征"，不走 FE 管线
  - **D**：Feature Engineering(175) + 多站（先做空间聚合，再走同一 FE 管线）
    - 在特征工程之前完成跨站拼接与空间聚合
    - 将"扩展后的原始变量"送入与 B 完全一致的 175 特征流水线
  - **E**：Raw-only + 多站（时空图神经网络，端到端学习）
    - 图结构：节点=站点；边=半径或 kNN；边权=距离衰减/可学
    - 节点特征：仅原始变量 + 时间编码，不走 FE
    - 模型族：ST-GCN / GC-LSTM / GAT / GraphWaveNet / DCRNN / TCN+GraphConv 等

- 适配现有与新增模型：
  - Raw 轨（A、C、E）：LightGBM/raw、LSTM/GRU/TCN raw 序列、空间聚合+Raw 模型、时空图（ST-GCN、GC-LSTM、GAT 等）
  - FE 轨（B、D）：LightGBM/XGBoost/CatBoost/RandomForest/Ensemble 等使用 175 特征管线

- 统一评测与落盘：
  - 切分：与现有 LOSO/时间窗设置保持一致
  - 指标：分类（ROC-AUC 等）、回归（MAE、R² 等）与当前一致
  - 目录：在 `experiments/<model>/<track>/<matrix_cell>/[full_training|...]/horizon_xxh/` 追加 `matrix_cell ∈ {A,B,C,D,E}`，多站实验（C、D、E）补充 `radius_km` 或 `knn_k`
  - 报告：新增“2×2 汇总”小节（曲线与表格），并输出“空间半径敏感性”曲线

### C（Raw + 多站：手工空间聚合）— 定义与落地
- **目标**：先做空间邻域聚合，再按 A 轨使用 Raw 模型，不走 FE。
- **空间聚合**：先对邻站原始变量做聚合（mean/max/min/std/median、距离加权均值、gradient（邻域均值减目标站）、range（max-min）等），生成聚合特征。
- **邻域构建**：基于 `cimis_station_metadata` 计算距离，按半径 R（如 25/50/75/100 km）或 kNN（如 1/3/5）建图；边权可用距离衰减或高斯核。
- **时间对齐与缺失**：邻站同一时间戳对齐，缺测可 forward-fill/rolling-mean，并产出缺失掩码特征；无邻居回退为单站并显式标记。
- **模型**：使用与 A 相同的 Raw 模型（LightGBM/raw、LSTM/GRU/TCN raw 序列等），输入为"原始变量 + 空间聚合特征"。
- **泄漏防控**：所有聚合与 rolling/lag 仅使用 t-1 及之前；LOSO 下测试站不参与训练期统计量拟合。
- **与 D 的区别**：D 在空间聚合后走 FE 管线（175 特征），C 在空间聚合后直接使用 Raw 模型（不走 FE）。
- **与 E 的区别**：C 用手工聚合特征，E 用图神经网络端到端学习空间关系。

### D（FE + 多站：空间聚合后走 FE 管线）— 正确逻辑（关键点）
- **目标**：在特征工程之前完成跨站拼接与空间聚合，再将"扩展后的原始变量"送入与 B 完全一致的 175 特征流水线，保证可比性。
- **邻域构建**：基于 `cimis_station_metadata` 计算距离，按半径 R（如 25/50/75/100 km）或 kNN（如 1/3/5）建图；边权可用距离衰减或高斯核。
- **时间对齐与缺失**：邻站同一时间戳对齐，缺测可 forward-fill/rolling-mean，并产出缺失掩码特征；无邻居回退为单站并显式标记。
- **空间聚合（在 FE 之前）**：对每个原始变量做 mean/max/min/std/median、距离加权均值、gradient（邻域均值减目标站）、range（max-min）；可选按风向门控上风向邻站。
- **泄漏防控**：所有聚合与 rolling/lag 仅使用 t-1 及之前；LOSO 下测试站不参与训练期统计量拟合。
- **FE 复用**：进入与 B 相同的 175 特征管线（参数/列名规范一致）。

### E（Raw + 多站：时空图神经网络）— 定义与落地
- **图结构**：节点=站点；边=半径或 kNN；边权=距离衰减/可学；可选风向门控。
- **节点特征**：仅原始变量 + 时间编码（hour_sin/cos、日/季节），不走 FE。
- **模型族**：ST-GCN / GC-LSTM / GraphWaveNet / DCRNN / GAT / TCN+GraphConv 等。
- **训练与评测**：与 B/C/D 同步的时间窗与指标；将 R 或 k 作为敏感性实验的超参维度。
- **实现建议**：新增 `src/models/graph/` 并与 `BaseModel` 接口对齐；图结构缓存到 `data/interim/graph/{R|k}.pkl`。
- **与 C 的区别**：C 用手工聚合特征，E 用图神经网络端到端学习空间关系。

### 空间半径敏感性实验（适用于 C、D、E）
- 设定：R ∈ {25,50,75,100} km 或 k ∈ {1,3,5}
- 报告：AUC/MAE/R² vs R（或 k）曲线；同时报告邻居覆盖率、缺失率对性能的影响
- 结论：给出最优空间影响范围与稳健区间

### 模型到矩阵单元的映射（示例指引）
- **A**：Raw-ML Baseline（LightGBM raw）、Raw-LSTM/GRU/TCN Baseline
- **B**：LightGBM（Top 175）、XGBoost/CatBoost/RandomForest（175）、Ensemble（基于 175）
- **C**：空间聚合 + LightGBM/raw、空间聚合 + LSTM/GRU/TCN（手工聚合，Raw 模型）
- **D**：LightGBM（Spatial FE 版 175）、其他基于 175 的树模型/集成（多站 FE）
- **E**：ST-GCN / GC-LSTM / GAT / GraphWaveNet / DCRNN / Raw-序列-图模型（端到端学习）

> 执行建议：在训练配置与结果落盘中增加字段 `matrix_cell`，并在现有报告中新增"2×2+1 汇总"与"空间半径敏感性"页，确保所有现有/新增模型均以该框架为主线进行对比与呈现。

# 模型开发路线图 (Model Development Roadmap)

## 📊 当前已实现的模型

### ✅ 已完成 (11 个模型)

#### 树模型 (5 个)
1. **LightGBM** - 梯度提升树模型，性能优秀
2. **XGBoost** - 梯度提升树模型，与 LightGBM 互补
3. **CatBoost** - 自动处理类别特征的梯度提升模型
4. **Random Forest** - 基准模型，稳定可解释
5. **Ensemble (Mean)** - 简单平均集成 LightGBM + XGBoost + CatBoost

#### 深度学习模型 (6 个)
6. **LSTM** - 长短期记忆网络，捕捉时间依赖
7. **LSTM Multi-task** - 多任务学习，同时预测温度和霜冻概率
8. **GRU** - 门控循环单元，比 LSTM 更轻量级
9. **TCN** - 时序卷积网络，使用扩张卷积捕捉长期依赖
10. **Prophet** - Facebook 时间序列模型，擅长趋势和季节性

### 🚧 开发中 (4 个图神经网络模型)

#### 图神经网络模型 (E 类别)
11. **DCRNN** - 扩散卷积循环网络（Phase 2，预计 2-3 天）
12. **ST-GCN** - 时空图卷积网络（Phase 3，预计 1-2 天）
13. **GAT-LSTM** - 图注意力网络 + LSTM（Phase 4，预计 2-3 天）
14. **GraphWaveNet** - 图卷积 + WaveNet（Phase 5，预计 3-4 天）

> **实现计划**: 详见 `docs/GRAPH_MODELS_IMPLEMENTATION_PLAN.md`
> **预计完成时间**: 10-16 天（2-3 周）

---

## 🧾 实验落盘与命名规范（2×2+1 框架统一）

为确保所有现有/新增模型能在框架下可比、可复现、可汇总，统一以下规范：

- 目录层级（示例）：
  - `experiments/{model_name}/{track}/{matrix_cell}/{training_scope}/horizon_{H}h/`
    - `model_name`：如 `lightgbm`、`xgboost`、`catboost`、`random_forest`、`lstm`、`st_gcn` 等
    - `track`：`top175_features`（FE 轨）或 `raw`（原始轨）
    - `matrix_cell`：`A`/`B`/`C`/`D`
    - `training_scope`：`full_training` 或其他既有设定
    - `H`：预测视窗，如 3/6/12/24
  - 对多站（C/D）实验，追加空间参数子目录或文件前缀：
    - 半径：`radius_{R}km/`
    - 或 kNN：`knn_{k}/`

- 文件命名（建议）：
  - `predictions.json` / `metrics.json` / `config.json`
  - 可选总览：`run_summary.json`（含关键信息与文件指针）

- 结果 CSV 追加字段（统一汇总时使用）：
  - `matrix_cell` ∈ {A,B,C,D,E}
  - `track` ∈ {raw, top175_features}
  - 对 C/D/E：`radius_km` 或 `knn_k`（二者选其一）
  - 其他保持与现有 `model_comparison.csv` 一致

> 说明：既有 `top175_features` 目录结构保持不变，仅在中间层新增 `matrix_cell` 层级，便于后续报告自动按四象限汇总与对比。

### 统一结果 CSV 字段规范（补充）
- 基础字段：`model_name`、`horizon_h`、`split`（如 LOSO）、
  `metric_type`（classification/regression）、`roc_auc`、`mae`、`r2`、`timestamp` 等（沿用现有规范）
- 新增字段：
  - `matrix_cell`（A/B/C/D/E）
  - `track`（raw / top175_features）
  - `radius_km`（数值或 NA）
  - `knn_k`（数值或 NA）
  - `notes`（可选，记录风向门控等实验条件）

### 目录结构示例
```
experiments/
  lightgbm/
    raw/
      A/
        full_training/
          horizon_12h/
            predictions.json
            metrics.json
            config.json
      C/
        full_training/
          radius_50km/
            horizon_12h/
              predictions.json
              metrics.json
              config.json
    top175_features/
      B/
        full_training/
          horizon_12h/
            predictions.json
            metrics.json
            config.json
      D/
        full_training/
          radius_50km/
            horizon_12h/
              predictions.json
              metrics.json
              config.json
  lstm/
    raw/
      A/
        full_training/
          horizon_6h/
            predictions.json
            metrics.json
            config.json
  st_gcn/
    raw/
      E/
        full_training/
          knn_3/
            horizon_24h/
              predictions.json
              metrics.json
              config.json
```

### 运行清单（Checklists）
- **A（Raw+单站）**：
  - 输入：原始变量 + 时间编码；无 FE
  - 模型：Raw-ML 或 Raw-序列（LSTM/GRU/TCN）
  - 落盘：`track=raw`，`matrix_cell=A`
- **B（175FE+单站）**：
  - 输入：单站 175 特征管线输出
  - 模型：树模型/集成（LightGBM 等）
  - 落盘：`track=top175_features`，`matrix_cell=B`
- **C（Raw+多站：手工空间聚合）**：
  - 先做空间邻域与聚合（R 或 kNN），生成聚合特征
  - 模型：Raw 模型（LightGBM/raw、LSTM/GRU/TCN 等），输入为"原始变量 + 空间聚合特征"
  - 泄漏防控：仅用 t-1 及之前；LOSO 不污染统计量
  - 落盘：`track=raw`，`matrix_cell=C`，记录 `radius_km` 或 `knn_k`
- **D（175FE+多站）**：
  - 先做空间邻域与聚合（R 或 kNN），再走 175 FE 管线
  - 泄漏防控：仅用 t-1 及之前；LOSO 不污染统计量
  - 落盘：`track=top175_features`，`matrix_cell=D`，记录 `radius_km` 或 `knn_k`
- **E（Raw+多站：时空图神经网络）**：
  - 输入：原始变量 + 时间编码；图结构（R/kNN）
  - 模型：ST-GCN/GC-LSTM/GAT/GraphWaveNet/DCRNN 等（端到端学习空间关系）
  - 落盘：`track=raw`，`matrix_cell=E`，记录 `radius_km` 或 `knn_k`

---

### 🛠 运行示例（2×2+1 框架下的评估与可视化）

- 评估并落盘（示例：D，多站 FE，R=50km，12h 视窗目录仅示意）：

```bash
python scripts/evaluate/evaluate_model.py /path/to/model_dir \
  --output experiments/lightgbm/top175_features/D/full_training/radius_50km/horizon_12h \
  --matrix-cell D \
  --track top175_features \
  --radius-km 50
```

- 2×2+1 汇总热力图（输入为多个评估输出目录）：

```bash
python scripts/evaluate/plot_matrix_summary.py \
  experiments/lightgbm/raw/A/full_training/horizon_12h \
  experiments/lightgbm/top175_features/B/full_training/horizon_12h \
  experiments/lightgbm/raw/C/full_training/radius_50km/horizon_12h \
  experiments/lightgbm/top175_features/D/full_training/radius_50km/horizon_12h \
  experiments/st_gcn/raw/E/full_training/knn_3/horizon_12h \
  --output experiments/summaries/matrix \
  --metrics mae r2 roc_auc
```

- 空间半径/邻居数敏感性（对 C/D/E 汇总曲线）：

```bash
python scripts/evaluate/plot_spatial_sensitivity.py \
  experiments \
  --output experiments/summaries/sensitivity \
  --metrics mae r2 roc_auc
```

> 注：脚本会读取 `evaluation_metrics.json` 与 `run_metadata.json` 中的 `matrix_cell/track/radius_km/knn_k` 字段以完成自动汇总。

---

## 🚀 模型实现状态

### ✅ 已完成
- 树模型：LightGBM, XGBoost, CatBoost, Random Forest, Ensemble
- 深度学习：LSTM, LSTM Multi-task, GRU, TCN, Prophet

### 🚧 开发中（图神经网络 - E 类别）
- **Phase 1**: 基础设施（graph_builder.py, base_graph_model.py）
- **Phase 2**: DCRNN（扩散卷积循环网络）
- **Phase 3**: ST-GCN（时空图卷积网络）
- **Phase 4**: GAT-LSTM（图注意力网络 + LSTM）
- **Phase 5**: GraphWaveNet（图卷积 + WaveNet）
- **Phase 6**: 集成与测试

详见：`docs/GRAPH_MODELS_IMPLEMENTATION_PLAN.md`

---

## 🚀 建议添加的模型（未来扩展）

### 优先级 1: 其他深度学习模型 (高价值)

#### 1. **GRU (Gated Recurrent Unit)**
- **优势**：
  - 比 LSTM 更轻量级，训练更快
  - 参数更少，不易过拟合
  - 在某些任务上与 LSTM 性能相当
- **适用场景**：与 LSTM 对比，评估轻量级 RNN 的效果
- **实现难度**：⭐⭐ (与 LSTM 类似)
- **预期时间**：2-3 小时

#### 2. **CNN-LSTM (Hybrid Model)**
- **优势**：
  - CNN 提取局部特征模式
  - LSTM 捕捉长期时间依赖
  - 结合两种架构的优势
- **适用场景**：捕捉多尺度时间模式（短期波动 + 长期趋势）
- **实现难度**：⭐⭐⭐ (需要设计 CNN 层)
- **预期时间**：3-4 小时

#### 3. **TCN (Temporal Convolutional Network)**
- **优势**：
  - 基于卷积，训练速度快
  - 并行计算效率高
  - 可以捕捉长期依赖（通过扩张卷积）
- **适用场景**：需要快速训练和推理的场景
- **实现难度**：⭐⭐⭐ (需要实现扩张卷积)
- **预期时间**：4-5 小时

### 优先级 2: Transformer 模型 (前沿技术)

#### 4. **Time Series Transformer**
- **优势**：
  - 注意力机制捕捉全局依赖
  - 在长序列预测中表现优秀
  - 可解释性强（注意力权重可视化）
- **适用场景**：长序列预测，需要捕捉全局时间依赖
- **实现难度**：⭐⭐⭐⭐ (需要实现位置编码、多头注意力)
- **预期时间**：6-8 小时

#### 5. **Informer / Autoformer**
- **优势**：
  - 专门为长序列时间序列设计
  - 计算效率高（ProbSparse 注意力）
  - 在多个时间序列基准测试中表现优秀
- **适用场景**：超长序列预测（>100 时间步）
- **实现难度**：⭐⭐⭐⭐⭐ (复杂架构)
- **预期时间**：8-10 小时

### 优先级 3: 高级集成方法

#### 6. **Stacking Ensemble**
- **优势**：
  - 使用元学习器学习如何组合基模型
  - 通常比简单平均效果更好
  - 可以学习不同模型的互补性
- **适用场景**：在已有多个强模型基础上进一步提升
- **实现难度**：⭐⭐⭐ (需要实现交叉验证和元学习器)
- **预期时间**：3-4 小时

#### 7. **Weighted Ensemble (学习权重)**
- **优势**：
  - 根据验证集性能自动学习权重
  - 比固定权重更灵活
- **适用场景**：优化当前 Ensemble 模型
- **实现难度**：⭐⭐ (相对简单)
- **预期时间**：1-2 小时

### 优先级 4: 传统时间序列模型

#### 8. **NeuralProphet**
- **优势**：
  - Prophet 的神经网络版本
  - 结合 Prophet 的可解释性和神经网络的灵活性
  - 自动学习非线性模式
- **适用场景**：需要 Prophet 风格的可解释性 + 更好的性能
- **实现难度**：⭐⭐⭐ (需要安装 neuralprophet 库)
- **预期时间**：2-3 小时

#### 9. **ARIMA / SARIMA** (可选)
- **优势**：
  - 经典统计模型，理论基础扎实
  - 可解释性强
- **劣势**：
  - 不适合多特征场景（当前有 175 个特征）
  - 需要手动选择参数
- **适用场景**：作为基准对比，或用于单变量时间序列
- **实现难度**：⭐⭐ (使用 statsmodels)
- **预期时间**：2-3 小时
- **建议**：优先级较低，除非需要统计基准

---

## 📋 推荐实施顺序

### 阶段 1: 快速提升 (1-2 周)
1. ✅ **GRU** - 快速实现，与 LSTM 对比
2. ✅ **Weighted Ensemble** - 优化当前集成模型
3. ✅ **CNN-LSTM** - 混合架构，可能带来性能提升

### 阶段 2: 深度优化 (2-3 周)
4. ✅ **TCN** - 高效的时间序列模型
5. ✅ **Stacking Ensemble** - 高级集成方法
6. ✅ **NeuralProphet** - Prophet 的改进版

### 阶段 3: 前沿探索 (3-4 周)
7. ✅ **Time Series Transformer** - 探索注意力机制
8. ✅ **Informer/Autoformer** - 超长序列预测（如果数据支持）

---

## 🎯 模型选择建议

### 基于当前任务特点：

1. **数据特点**：
   - 18 个站点，多站点数据
   - 175 个特征（多特征）
   - 时间序列（小时级数据）
   - 需要预测多个时间窗口（3h, 6h, 12h, 24h）

2. **推荐模型**：
   - ✅ **GRU** - 轻量级，快速验证
   - ✅ **CNN-LSTM** - 多尺度特征提取
   - ✅ **TCN** - 高效训练
   - ✅ **Stacking Ensemble** - 提升集成效果
   - ✅ **Time Series Transformer** - 如果序列长度足够

3. **不推荐**：
   - ❌ **ARIMA/SARIMA** - 不适合多特征场景
   - ❌ **纯 CNN** - 缺少时间依赖建模

---

## 📝 实施注意事项

### 1. 模型架构设计
- 保持与现有 `BaseModel` 接口一致
- 支持 `checkpoint_dir` 和 `log_file` 参数
- 实现统一的训练工具（history, checkpoint, logging）

### 2. 数据预处理
- 确保新模型能处理现有的特征工程输出
- 对于序列模型，复用 `TimeSeriesDataset`
- 对于 Transformer，可能需要实现新的数据加载器

### 3. 训练配置
- 在 `src/training/model_config.py` 中添加新模型配置
- 支持资源感知配置（根据内存调整参数）
- 实现早停、学习率调度等优化

### 4. 评估和对比
- 使用相同的评估指标
- 支持 LOSO 评估
- 生成对比报告

---

## 🔄 持续改进

1. **模型对比**：定期对比所有模型性能
2. **超参数优化**：使用 Optuna 或 Hyperopt 进行自动调参
3. **模型融合**：探索更复杂的集成策略
4. **可解释性**：添加 SHAP 值、注意力可视化等

---

**最后更新**: 2025-11-15  
**维护者**: Zhengkun LI

