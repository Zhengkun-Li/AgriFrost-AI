# 深度学习模型快速测试指南

## 概述

`test_deep_models_quick.py` 是一个快速测试脚本，用于验证深度学习模型（LSTM, GRU, TCN）是否正常工作。

## 特点

- ✅ **内存友好**: 仅使用 5000 个样本
- ✅ **快速验证**: 每个模型训练 5 个 epoch（约 2-5 分钟）
- ✅ **完整测试**: 测试分类和回归任务
- ✅ **不影响其他训练**: 不会干扰正在运行的 Random Forest 训练

## 使用方法

```bash
# 进入项目目录
cd /home/zhengkun-li/frost-risk-forecast-challenge

# 运行测试
python scripts/test/test_deep_models_quick.py
```

## 测试内容

脚本会测试以下模型：
1. **LSTM** - Long Short-Term Memory 网络
2. **GRU** - Gated Recurrent Unit 网络
3. **TCN** - Temporal Convolutional Network

每个模型都会测试：
- 分类任务（霜冻预测）
- 回归任务（温度预测）

## 配置参数

可以在脚本中修改以下参数：

```python
models_to_test = ['lstm', 'gru', 'tcn']  # 要测试的模型列表
horizon = 3                               # 预测时间窗口（小时）
max_samples = 5000                        # 使用的样本数量
```

## 预期输出

测试成功时会显示：
```
✅ LSTM test PASSED!
✅ GRU test PASSED!
✅ TCN test PASSED!

TEST SUMMARY
============
LSTM: ✅ PASSED
GRU: ✅ PASSED
TCN: ✅ PASSED

Total: 3/3 models passed
```

## 注意事项

1. **内存使用**: 测试使用少量数据，但深度学习模型仍需要一些内存。如果系统内存紧张，可以减少 `max_samples`。

2. **GPU 支持**: 如果系统有 GPU，PyTorch 会自动使用。可以在脚本中设置 `CUDA_VISIBLE_DEVICES` 来控制使用哪个 GPU。

3. **训练时间**: 每个模型约需要 2-5 分钟，取决于硬件配置。

4. **依赖**: 确保已安装 PyTorch:
   ```bash
   pip install torch
   ```

## 故障排除

**问题**: ImportError: PyTorch is required
**解决**: 安装 PyTorch: `pip install torch`

**问题**: CUDA out of memory
**解决**: 减少 `max_samples` 或 `batch_size` 参数

**问题**: 测试失败但错误信息不明确
**解决**: 检查日志输出，通常会有详细的错误信息

## 下一步

测试通过后，可以使用完整的训练流程训练深度学习模型：

```bash
python -m src.cli train single \
    --model-name lstm \
    --matrix-cell A \
    --track raw \
    --horizon-h 3 \
    --output-dir experiments/lstm/raw/A/full_training/horizon_3h
```

