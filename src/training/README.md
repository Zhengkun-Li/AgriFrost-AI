# Training Module

训练模块 (`src/training`) 负责模型训练、评估和推理的整个流程。

## 📁 模块结构

```
src/training/
├── __init__.py              # 模块导出
├── pipeline_runner.py       # 配置驱动的训练/评估/推理运行器
├── model_trainer.py         # 模型训练逻辑
├── model_config.py          # 模型配置管理
├── data_preparation.py      # 数据准备（使用 DataPipeline）
└── loso_evaluator.py        # Leave-One-Station-Out 评估
```

## 🔧 核心组件

### 1. PipelineRunner (`pipeline_runner.py`)

配置驱动的执行器，支持训练、评估和推理：

- **TrainingRunner**: 执行完整的训练流程
- **EvaluationRunner**: 执行评估任务（LOSO、直接评估等）
- **InferenceRunner**: 执行推理任务

**关键特性**:
- ✅ 统一的配置系统（YAML + CLI 覆盖）
- ✅ **严格的列校验**（DATE_COL, STATION_ID_COL, TEMP_COL, 特征列）
- ✅ **Track 推断改进**（明确的 RAW_CELLS vs FE_CELLS，支持 E track）
- ✅ **CLI override 逻辑改进**（避免误判，如 'AINet' 中的 'A'）
- ✅ **Feature engineering enable 逻辑**（CLI 优先于 track 推断）
- ✅ 输入验证（DataFrame 空值检查、列检查）
- ✅ 标准化的日志记录

### 2. ModelTrainer (`model_trainer.py`)

模型训练核心逻辑：

- **train_models_for_horizon**: 为特定 horizon 训练模型
- **train_frost_model**: 训练霜冻分类模型
- **train_temp_model**: 训练温度回归模型
- **train_multitask_model**: 训练多任务模型（LSTM）

**关键特性**:
- ✅ 支持多种模型类型（树模型、深度学习、图神经网络）
- ✅ LOSO 兼容的训练流程
- ✅ **GPU 内存管理**（每个 horizon 后自动清理 GPU 缓存）
- ✅ **增强的验证逻辑**（分别检查 X, y_frost, y_temp，形状匹配验证）
- ✅ 内存优化（减少不必要的复制）
- ✅ 输入验证和错误处理

### 3. DataPreparation (`data_preparation.py`)

数据准备工具函数：

- **load_and_prepare_data**: 使用 DataPipeline 加载和准备数据
- **create_frost_labels**: 创建霜冻标签
- **prepare_features_and_targets**: 准备特征和目标变量

**关键特性**:
- ✅ 完全集成 DataPipeline
- ✅ **严格的输入验证**（特征列存在性检查，目标列存在性检查，使用 KeyError）
- ✅ **改进的错误消息**（显示可用列列表）
- ✅ 内存优化（减少不必要的 `.copy()`）
- ✅ 支持 track-aware 特征选择

### 4. LOSOEvaluator (`loso_evaluator.py`)

Leave-One-Station-Out 交叉验证评估：

- **perform_loso_evaluation**: 执行完整的 LOSO 评估
- **calculate_loso_summary**: 计算汇总统计

**关键特性**:
- ✅ **严格的时间泄漏防护**（时间顺序验证，站点隔离验证）
- ✅ **严格的时间排序**（train 和 test 都按 date + hour 排序）
- ✅ **GPU 内存清理**（每个 horizon 后自动清理）
- ✅ **改进的输出目录结构**（包含 track 信息，便于 2×2+1 框架汇总）
- ✅ 内存优化（按需加载、及时释放）
- ✅ 检查点和恢复功能
- ✅ 错误处理和恢复
- ✅ 文件验证和格式检查

### 5. ModelConfig (`model_config.py`)

模型配置管理：

- **get_model_params**: 获取模型参数
- **get_model_class**: 获取模型类
- **get_model_config**: 获取完整配置
- **get_resource_aware_config**: 资源感知配置

**关键特性**:
- ✅ **完整的图模型支持**（dcrnn, st_gcn, gat_lstm, graphwavenet）
- ✅ 图模型特定参数（graph_type, graph_param, edge_weight）
- ✅ 资源感知配置（根据可用内存自动调整参数）
- ✅ LOSO 模式优化（更小的配置以节省内存）

## ✅ 代码质量改进

### 已完成

1. **日志标准化** ✅
   - 所有 `print()` 语句已替换为 `logging`
   - 统一的日志级别（info/warning/error/debug）
   - 约 100+ 处改进

2. **错误处理改进** ✅
   - 使用具体异常类型（`ValueError`, `KeyError`, `FileNotFoundError`, `pd.errors.EmptyDataError`, `pd.errors.ParserError`）
   - 清晰的错误消息
   - 区分可恢复错误和意外错误

3. **输入验证** ✅
   - **严格列校验**（DATE_COL, STATION_ID_COL, TEMP_COL 强制检查，使用 KeyError）
   - DataFrame 空值检查
   - 必需列存在性检查
   - 文件存在性和格式验证
   - **特征和目标验证**（分别检查 X, y_frost, y_temp，形状匹配）

4. **泄漏防护** ✅
   - **LOSO 时间泄漏防护**（时间顺序验证，站点隔离验证）
   - **严格时间排序**（所有 splits 按 date + hour 排序）

5. **GPU 内存管理** ✅
   - **每个 horizon 后自动清理 GPU 缓存**（`torch.cuda.empty_cache()`）
   - 支持多 horizon 训练而不爆显存

6. **内存优化** ✅
   - 移除不必要的 `.copy()` 调用
   - 使用 DataFrame views 代替 copies
   - 优化数据类型转换（inplace）

7. **配置和逻辑改进** ✅
   - **Track 推断重写**（明确的 RAW_CELLS = {A, C, E}, FE_CELLS = {B, D}）
   - **CLI override 逻辑改进**（避免路径误判）
   - **Feature engineering enable 逻辑**（CLI 优先）

## 📝 使用示例

### 训练模型

```python
from src.training.pipeline_runner import load_training_config, TrainingRunner
from pathlib import Path

# 加载配置
config = load_training_config(
    config_path=None,  # 使用默认配置
    project_root=Path("."),
    cli_overrides={
        "model": "lightgbm",
        "output_dir": "experiments/lightgbm/B/test",
        "matrix_cell": "B",
        "horizons": [3, 6, 12, 24],
        "loso": True,
    }
)

# 执行训练
runner = TrainingRunner(config, project_root=Path("."))
exit_code = runner.run()
```

### LOSO 评估

```python
from src.training.loso_evaluator import perform_loso_evaluation

results = perform_loso_evaluation(
    labeled_path=Path("experiments/lightgbm/B/test/labeled_data.parquet"),
    horizons=[3, 6, 12, 24],
    output_dir=Path("experiments/lightgbm/B/test"),
    model_type="lightgbm",
    frost_threshold=0.0,
    resume=True,  # 支持恢复
    save_models=False,
)
```

## 🔍 数据流程

```
DataPipeline (数据加载/清洗/特征工程)
    ↓
Label Generation (标签生成)
    ↓
Feature Preparation (特征准备)
    ↓
Model Training (模型训练)
    ↓
LOSO Evaluation (LOSO 评估)
```

## ⚠️ 注意事项

1. **数据验证**: 所有数据准备步骤都包含输入验证，确保数据完整性
2. **内存管理**: LOSO 评估使用按需加载策略，适合大数据集
3. **错误恢复**: LOSO 评估支持检查点和恢复，避免重复计算
4. **日志记录**: 所有操作都记录日志，便于调试和监控

## 🚀 性能优化

- **内存优化**: 减少不必要的 DataFrame 复制
- **按需加载**: LOSO 评估按站点加载数据
- **类型优化**: 自动优化数据类型以减少内存使用
- **缓存机制**: EvaluationRunner 和 InferenceRunner 使用数据集缓存

## 📊 状态

**模块状态**: ✅ **生产就绪**

**最后更新**: 2025-11-19

所有关键问题已修复：
- ✅ 日志标准化
- ✅ 错误处理
- ✅ 输入验证
- ✅ 内存优化
- ✅ 严格列校验
- ✅ 时间泄漏防护
- ✅ GPU 内存管理

