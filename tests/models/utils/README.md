# Model Utils Tests

单元测试、集成测试和性能测试套件，用于验证 `src/models/utils/` 模块的所有工具类。

## 📁 测试文件结构

```
tests/models/utils/
├── __init__.py
├── test_progress_logger.py      # ProgressLogger 单元测试
├── test_training_history.py     # TrainingHistory 单元测试
├── test_checkpoint_manager.py   # CheckpointManager 单元测试
├── test_config_validator.py     # ConfigValidator 单元测试
├── test_graph_builder.py        # GraphBuilder 单元测试
├── test_integration.py          # 集成测试
└── test_performance.py          # 性能基准测试
```

## 🧪 测试内容

### 单元测试

1. **test_progress_logger.py**
   - 日志文件创建和写入
   - 日志轮转功能（`max_log_size_mb`）
   - Flush 优化机制（`flush_interval`）
   - tqdm 集成和配置
   - `log_epoch` 字段对齐

2. **test_training_history.py**
   - 统一 metrics 列表
   - `epoch_time` 标准字段
   - duration 精度（使用 epoch_times 累加）
   - 加载时的 metrics 过滤
   - 保存/加载一致性

3. **test_checkpoint_manager.py**
   - GPU/CPU 兼容性（自动转换到 CPU）
   - best-k checkpoint 管理（`keep_top_k`）
   - resume 训练功能
   - checkpoint metadata 暴露
   - 定期保存和最佳模型保存

4. **test_config_validator.py**
   - 2×2+1 框架规则验证
   - matrix_cell A/B/E 禁止 radius_km 和 knn_k
   - matrix_cell C/D 要求 radius_km
   - matrix_cell E 要求 knn_k
   - horizon_h 和 track 验证

5. **test_graph_builder.py**
   - 图构建（radius 和 knn）
   - 图缓存机制
   - 增强的缓存验证（station_ids 和 station_coords hash）
   - metadata 导出到 `run_metadata.json`

### 集成测试

**test_integration.py**
- ProgressLogger + TrainingHistory 字段统一
- GraphBuilder metadata 导出
- ConfigValidator 2×2+1 规则验证
- CheckpointManager resume 训练
- 完整训练工作流模拟

### 性能测试

**test_performance.py**
- Flush 优化性能（减少 10x flush 操作）
- 日志轮转性能
- duration 精度测试

## 🚀 运行测试

### 使用 pytest（推荐）

```bash
# 运行所有测试
python -m pytest tests/models/utils/ -v

# 运行特定测试文件
python -m pytest tests/models/utils/test_progress_logger.py -v

# 运行特定测试用例
python -m pytest tests/models/utils/test_progress_logger.py::TestProgressLogger::test_init -v

# 运行性能测试
python -m pytest tests/models/utils/test_performance.py -v -s
```

### 直接运行 Python（绕过 pytest 插件问题）

如果遇到 ROS 插件冲突，可以直接运行 Python 代码验证：

```bash
python << 'EOF'
from src.models.utils.progress_logger import ProgressLogger
from src.models.utils.training_history import TrainingHistory
from src.models.utils.checkpoint_manager import CheckpointManager
from src.models.utils.config_validator import ConfigValidator
print("✅ 所有工具类导入成功")
EOF
```

## ⚠️ 注意事项

### pytest 插件冲突

如果遇到以下错误：
```
PluginValidationError: unknown hook 'pytest_launch_collect_makemodule' in plugin
```

这是因为系统中安装了 ROS 的 pytest 插件，与 pytest 9.x 不兼容。

**解决方案**：
1. `pytest.ini` 已配置禁用相关插件
2. 或使用 `-p no:launch_testing_ros_pytest_entrypoint` 参数
3. 或直接运行 Python 代码验证功能

### 测试环境要求

- Python 3.8+
- pytest
- torch（用于 CheckpointManager 测试）
- numpy（用于数值计算测试）

所有依赖应在虚拟环境中安装：
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 📊 测试覆盖

- ✅ ProgressLogger: 日志轮转、flush优化、tqdm集成
- ✅ TrainingHistory: 字段统一、duration精度、加载过滤
- ✅ CheckpointManager: GPU/CPU兼容、best-k管理、resume训练
- ✅ ConfigValidator: 2×2+1框架规则验证
- ✅ GraphBuilder: 图构建、缓存、metadata导出
- ✅ 集成测试: 工具间协作和完整工作流
- ✅ 性能测试: flush优化、日志轮转、duration精度

## 🔗 相关文档

- [`src/models/utils/`](../../../src/models/utils/) - 工具类源代码
- [`docs/MODEL_TRAINING_UTILITIES.md`](../../../docs/MODEL_TRAINING_UTILITIES.md) - 详细使用文档
- [`examples/training_with_tools.py`](../../../examples/training_with_tools.py) - 端到端使用示例

