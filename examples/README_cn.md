# Examples

Complete usage examples for the frost risk forecasting project.

## 📁 Examples

### `training_with_tools.py`

Complete end-to-end example demonstrating how to use all improved training tools:

- **ProgressLogger**: Logging with rotation and flush optimization
- **TrainingHistory**: History tracking with unified fields
- **CheckpointManager**: GPU/CPU compatible checkpoints with best-k saving
- **ConfigValidator**: 2×2+1 framework validation
- **GraphBuilder**: Graph construction with metadata export

**Features demonstrated:**
- ✅ Configuration validation (2×2+1 framework)
- ✅ Tool initialization and setup
- ✅ Experiment metadata creation
- ✅ Graph building with metadata export
- ✅ Complete training loop with unified fields
- ✅ Checkpoint saving (periodic + best-k)
- ✅ Training history saving
- ✅ Training curve plotting
- ✅ Resume training from checkpoint

**Prerequisites:**

确保已创建并激活虚拟环境：

```bash
# 创建虚拟环境（如果还没有）
python3 -m venv .venv

# 激活虚拟环境
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate.bat

# 安装依赖（如果还没有）
pip install -r requirements.txt
```

**Usage:**
```bash
cd /home/zhengkun-li/frost-risk-forecast-challenge

# 确保虚拟环境已激活（命令提示符前应显示 (.venv)）
source .venv/bin/activate  # Linux/macOS

# 运行示例
python examples/training_with_tools.py
```

## 📝 Additional Examples

For more examples, see:
- [`docs/QUICK_START.md`](../docs/QUICK_START.md) - Quick start tutorial
- [`notebooks/tutorial.ipynb`](../notebooks/tutorial.ipynb) - Interactive Jupyter notebook
- [`docs/MODEL_TRAINING_UTILITIES.md`](../docs/MODEL_TRAINING_UTILITIES.md) - Detailed tool documentation
