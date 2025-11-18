## Experiments Summary (Matrix A/B/C/D/E)

说明：当前所有已训练结果均归档到 `experiments/B/`（矩阵单元 B：Feature Engineering(175) + 单站）。下表列出各模型的训练与 LOSO 状态。

### 状态定义
- full_training: 标准时间切分训练（3h/6h/12h/24h）
- loso: Leave-One-Station-Out 评估（按站点落盘）
- track: 数据轨迹（raw 或 top175_features）
- matrix_cell: A/B/C/D/E（详见 docs/MODEL_ROADMAP.md）

### 总览

| Model          | matrix_cell | track            | full_training | loso | Path Hint |
|----------------|-------------|------------------|---------------|------|-----------|
| LightGBM       | B           | top175_features  | ✅             | ✅    | `experiments/B/lightgbm/top175_features/` |
| XGBoost        | B           | top175_features  | ✅             | ✅    | `experiments/B/xgboost/top175_features/` |
| CatBoost       | B           | top175_features  | ✅             | ✅    | `experiments/B/top175_features/`（CatBoost 结果） |
| Random Forest  | B           | top175_features  | ✅             | ✅    | `experiments/B/random_forest/top175_features/` |
| Ensemble (Mean)| B           | top175_features  | ✅             | ✅    | `experiments/B/ensemble/top175_features/` |
| LSTM           | B           | top175_features  | ⏳/未确认       | ⛔    | `experiments/B/lstm/top175_features/` |
| LSTM Multitask | B           | top175_features  | ⛔             | ⛔    | `experiments/B/lstm_multitask/top175_features/` |
| Prophet        | B           | top175_features  | ⛔             | ⛔    | `experiments/B/prophet/top175_features/` |

注：
- CatBoost 的结果目前集中在 `experiments/B/top175_features/` 下（含 `catboost_training.log` 等），后续可迁移至 `experiments/B/catboost/top175_features/` 以完全一致化。
- LSTM 在顶层 `experiments/lstm/top175_features/` 有历史日志，但 `experiments/B/lstm/top175_features/` 目录下当前未检测到落盘结果，标记为“未确认”。

### 详细路径示例

- LightGBM
  - full_training: `experiments/B/lightgbm/top175_features/full_training/horizon_{3h|6h|12h|24h}/`
  - loso: `experiments/B/lightgbm/top175_features/loso/station_{id}/horizon_{H}h/`
- XGBoost
  - full_training: `experiments/B/xgboost/top175_features/full_training/`
  - loso: `experiments/B/xgboost/top175_features/loso/`
- CatBoost
  - full_training: `experiments/B/top175_features/full_training/`（含 CatBoost 训练日志与产物）
  - loso: `experiments/B/top175_features/loso/`
- Random Forest
  - full_training: `experiments/B/random_forest/top175_features/full_training/`
  - loso: `experiments/B/random_forest/top175_features/loso/`
- Ensemble (Mean)
  - full_training: `experiments/B/ensemble/top175_features/full_training/`
  - loso: `experiments/B/ensemble/top175_features/loso/`
- LSTM / LSTM Multitask / Prophet
  - 目前 `experiments/B/{lstm|lstm_multitask|prophet}/top175_features/` 下未检测到标准落盘（待训练/迁移）。

### 下一步建议
- 目录一致化：将 CatBoost 结果迁移到 `experiments/B/catboost/top175_features/`，保持与其他模型一致的层级。
- LSTM/LSTM Multitask/Prophet：按统一管线启动 full_training 与 LOSO，并产出 `training.log`（简要）与 `training_detailed.log`（详细，本地保留）。
- 在评测输出中补充 `matrix_cell=B` 与 `track=top175_features` 字段，便于 2×2 汇总脚本自动聚合。

