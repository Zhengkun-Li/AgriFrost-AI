# Scripts Directory

⚠️ **本目录已简化。所有主要功能请使用新的 CLI 命令：`python -m src.cli ...`**

## 新项目结构

本项目已完全迁移到统一的 CLI 接口。`scripts/` 目录仅保留：

1. **工具脚本** (`tools/`) - 独立的工具脚本，如获取元数据等
2. **测试脚本** (`test/`) - 项目测试脚本
3. **Shell 脚本** - 训练相关的 shell 脚本（如果仍在使用）

## 使用新的 CLI

所有主要功能已迁移到 `src/cli`：

```bash
# 训练
python -m src.cli train single --model-name lightgbm --matrix-cell B --track top175_features --horizon-h 12
python -m src.cli train matrix --config config/pipeline/matrix_experiments.yaml

# 评估
python -m src.cli evaluate model --model-dir experiments/model_dir --config config/evaluation.yaml
python -m src.cli evaluate compare --model-dirs dir1 dir2 --output-dir comparison/
python -m src.cli evaluate matrix --experiments-dir experiments/ --output-dir matrix/

# 推理
python -m src.cli inference predict --model-dir experiments/model --input data.csv --output pred.csv

# 分析
python -m src.cli analysis full --data-path data.csv --output-dir analysis/
python -m src.cli analysis compare-sets --feature-sets '[{"name": "raw", "path": "data/raw.csv"}]' --output-dir comparison/
```

## 获取帮助

```bash
# 查看所有命令
python -m src.cli --help

# 查看特定命令的帮助
python -m src.cli train --help
python -m src.cli evaluate --help
python -m src.cli inference --help
python -m src.cli analysis --help
```

## 目录结构

```
scripts/
├── README.md              # 本文件
├── MIGRATION.md           # 详细的迁移指南（如果从旧版本升级）
├── tools/                 # 独立工具脚本
│   ├── fetch_station_metadata.py
│   ├── generate_station_map.py
│   ├── run_full_pipeline.py
│   └── select_features.py
└── test/                  # 测试脚本
    └── test_graph_builder.py
```

## 为什么迁移？

1. **统一接口**：所有命令使用相同的模式 `python -m src.cli <command> <subcommand>`
2. **更好的帮助**：使用 `--help` 查看详细用法
3. **类型安全**：完整的类型检查和验证
4. **可扩展性**：易于添加新命令和功能
5. **生产就绪**：完善的错误处理和资源管理
6. **代码复用**：消除重复代码，统一实现

## 旧脚本已完全移除

所有旧的脚本包装器已完全移除：

- ❌ `scripts/train/` - 已删除（使用 `python -m src.cli train ...`）
- ❌ `scripts/evaluate/` - 已删除（使用 `python -m src.cli evaluate ...`）
- ❌ `scripts/inference/` - 已删除（使用 `python -m src.cli inference ...`）
- ❌ `scripts/analysis/` - 已删除（使用 `python -m src.cli analysis ...`）

所有功能都已集成到 `src/cli` 中。**这是一个全新的项目结构，无需向后兼容。**

## 工具脚本

`tools/` 目录下的脚本是独立工具，可能需要单独运行：

### `fetch_station_metadata.py`

获取 CIMIS 站点元数据（包括经纬度、名称等）。

**Usage:**
```bash
python scripts/tools/fetch_station_metadata.py
```

**Output:**
- `data/external/cimis_station_metadata.json`
- `data/external/cimis_station_metadata.csv`

### `generate_station_map.py`

生成交互式站点分布地图，显示所有 18 个 CIMIS 站点位置。

**Usage:**
```bash
# 先获取站点元数据（如果需要）
python scripts/tools/fetch_station_metadata.py

# 生成地图
python scripts/tools/generate_station_map.py
```

**Requirements:**
- Station metadata must be available (run `fetch_station_metadata.py` first)
- Optional: Mapbox token in `config/settings.json` for better map tiles

**Output:**
- `docs/figures/station_distribution_map.html` - Interactive HTML map

### `run_full_pipeline.py`

运行完整流水线（特定用途）

### `select_features.py`

特征选择工具

---

**Note**: 这些脚本可以根据需要使用，它们不依赖于新的 CLI 结构。
