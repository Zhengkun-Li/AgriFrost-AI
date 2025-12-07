# AgriFrost-AI 数据文档

<div align="center">

<img src="logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**最后更新**: 2025-11-12

本文档详细说明 **AgriFrost-AI** 项目的数据格式、QC 处理、变量使用情况和数据分析报告。

## 📋 目录

1. [数据概览](#数据概览)
2. [字段说明](#字段说明)
3. [QC 处理](#qc-处理)
4. [变量使用情况](#变量使用情况)
5. [数据分析报告](#数据分析报告)

---

## 数据概览

### 数据来源

- **来源**：F3 Innovate Frost Risk Forecast Challenge 官方仓库（2025）
- **位置**：`data/raw/frost-risk-forecast-challenge/`
- **格式**：CSV 文件

### 数据规模

- **时间跨度**: 2010-09-28 至 2025-09-28
- **时间分辨率**: 逐小时
- **站点数量**: 18个
- **总记录数**: 约236万行

### 数据结构

```
data/raw/frost-risk-forecast-challenge/
├── cimis_all_stations.csv.gz    # 全站合并文件
└── stations/                     # 逐站CSV文件
    ├── 2-FivePoints.csv
    ├── 7-Firebaugh_Telles.csv
    └── ... (共18个站点)
```

---

## 字段说明

### 字段结构概览

| 字段 | 说明 | 备注 |
| --- | --- | --- |
| `Stn Id` | 站点编号 | 可视为分类变量 |
| `Stn Name` | 站点名称 | 与 `Stn Id` 一一对应 |
| `CIMIS Region` | CIMIS 官方区域分类 | 分析区域差异时可用 |
| `Date` | 日期（yyyy-mm-dd） | 已转为 datetime |
| `Hour (PST)` | 当地时（PST，离散编码） | 0000 / 0100 … 2300 |
| `Jul` | 年内日序 | 可用于季节性特征 |
| `ETo (mm)` | 参考蒸散量 | 数值型 |
| `Precip (mm)` | 降水量 | 数值型 |
| `Sol Rad (W/sq.m)` | 太阳辐射 | 含哨兵值 -6999 |
| `Vap Pres (kPa)` | 水汽压 | 数值型 |
| `Air Temp (C)` | 气温 | 霜冻主目标之一 |
| `Rel Hum (%)` | 相对湿度 | 存在极端负值，需处理 |
| `Dew Point (C)` | 露点 | 数值型 |
| `Wind Speed (m/s)` | 风速 | ≥0.2 m/s |
| `Wind Dir (0-360)` | 风向 | 0–360 |
| `Soil Temp (C)` | 土壤温度 | 含哨兵值 -6999 |
| `qc` ~ `qc.9` | 各物理量对应的质量控制标记 | 空白、`Y`、`M`、`R`、`Q`、`S`、`P` 等编码 |

> 说明：带有 `qc` 前缀的字段用于指示同一行中前一个物理量的质量等级，例如 `qc.4` 对应 `Air Temp (C)`。

### 变量物理意义

这些字段用于监测和预测霜冻风险的气象特征，能帮助模型刻画地表能量与水分平衡、空气状态及地面冷却条件：

- **ETo (mm)**：参考蒸散量，描述地表水分蒸发能力。高蒸散意味着地面散热增强，晚间更容易降温到霜冻阈值。
- **Precip (mm)**：降水量，最近的降水会提高土壤和空气湿度，减缓夜间降温；干燥条件则加剧霜冻风险。
- **Sol Rad (W/sq.m)**：太阳辐射，白天的辐射累积决定了夜间可散失的热量；云层减少会让夜间辐射降温更强。
- **Vap Pres (kPa)**：水汽压，反映空气中水汽含量；水汽越少，夜间长波辐射冷却越强，霜冻概率更高。
- **Air Temp (C)**：气温，为霜冻风险的核心指标，低于作物临界温度即出现霜害。
- **Rel Hum (%)**：相对湿度，影响露点与凝霜；极端低湿度时温度更易快速下降。
- **Dew Point (C)**：露点，标志空气达到饱和时的温度。霜冻往往发生在气温接近或低于露点的时段。
- **Wind Speed (m/s)**：风速，弱风或静风条件下易形成辐射霜冻；适度搅动可缓解地表冷却。
- **Wind Dir (0-360)**：风向，有助于识别冷空气入侵路径及局地环流模式。
- **Soil Temp (C)**：土壤温度，地表热储量的直观指标。较高的土温可在夜间向上供热，减轻霜冻。

---

## QC 处理

### QC 标记含义

根据 CIMIS 标准，QC 标记的含义如下：

| QC 标记 | 含义 | 处理方式 | 说明 |
|---------|------|----------|------|
| **空白** | 通过全部质检 | ✅ **保留** | 高可信度数据 |
| **Y** | Moderate outlier, accepted | ✅ **保留** | 轻度偏离但已接受 |
| **Q** | Questionable | ❌ **标记为缺失** | 可疑值，默认剔除 |
| **P** | Provisional | ❌ **标记为缺失** | 临时/插补值，默认剔除 |
| **M** | Missing data | ❌ **标记为缺失** | 缺失数据 |
| **R** | Rejected | ❌ **标记为缺失** | 严重超出阈值，拒绝 |
| **S** | Severe outlier | ❌ **标记为缺失** | 极端离群值 |

### 处理流程

系统使用 `DataCleaner` 类来处理质量控制（QC）标记：

1. **QC 过滤**: 根据 QC 标记过滤低质量数据
   - 自动检测所有 QC 列（以 `qc` 开头的列）
   - 为每个变量找到对应的 QC 列
   - 根据 QC 标记决定是否保留数据

2. **哨兵值处理**: 将 `-6999`, `-9999` 等哨兵值替换为 `NaN`

3. **缺失值插补**: 使用前向填充（按站点分组）

### 使用示例

```python
from src.data.cleaners import DataCleaner

cleaner = DataCleaner()
df_cleaned = cleaner.clean_pipeline(df)
```

---

## 变量使用情况

### 当前使用情况

| 变量名 | 使用情况 | 说明 |
|--------|----------|------|
| `Air Temp (C)` | ✅ 已使用 | 滞后、滚动、派生特征 |
| `Dew Point (C)` | ✅ 已使用 | 滞后、滚动、派生特征 |
| `Rel Hum (%)` | ✅ 已使用 | 滞后、滚动、派生特征 |
| `Wind Speed (m/s)` | ✅ 已使用 | 滞后、滚动、派生特征 |
| `Wind Dir (0-360)` | ✅ 已使用 | 周期性编码、滞后特征 |
| `Sol Rad (W/sq.m)` | ✅ 已使用 | 滞后、滚动、派生特征 |
| `Soil Temp (C)` | ✅ 已使用 | 滞后、滚动特征 |
| `ETo (mm)` | ✅ 已使用 | 滞后、滚动特征 |
| `Precip (mm)` | ✅ 已使用 | 滞后、滚动特征 |
| `Vap Pres (kPa)` | ✅ 已使用 | 滞后、滚动特征 |

### 变量使用率

- **之前**: 31% (5/16)
- **现在**: 81% (13/16)
- **改进**: +50%

---

## 数据分析报告

### 数据概览统计

- **总记录数**：2,367,360
- **字段数量**：26
- **站点数量**：18
- **时间范围**：2010-09-28T00:00:00 至 2025-09-28T00:00:00
- **气温分布（°C）**：均值 17.10｜p1 0.20｜p50 16.20｜p99 37.30

### 核心洞察

- **缺失率最高的观测量**：
  - ETo (mm) 2.73%
  - Soil Temp (C) 2.48%
  - Air Temp (C) 0.84%
  - Dew Point (C) 0.82%
  - Rel Hum (%) 0.82%

- **缺失最严重的站点**：Coalinga (2.50%)，Oakdale (2.33%)，Panoche (2.29%)

- **异常标记贡献度**：
  - dew_point_extreme: 2299
  - humidity_out_of_range: 20
  - sol_rad_sentinel: 18
  - soil_temp_sentinel: 3
  - air_temp_extreme: 2

- **异常最集中的站点**：Parlier (648)，Modesto (470)，Auburn (253)

- **极端气温站点**：
  - 最低：Modesto (-53.4 °C)
  - 最高：Stratford (48.0 °C)

- **平均湿度对比**：
  - 最干：Coalinga (46.2%)
  - 最湿：Modesto (70.0%)

### 输出文件

数据分析脚本生成以下文件：
- `data/processed/station_overview.csv`
- `data/processed/missing_by_station.csv`
- `data/processed/anomalies_by_station.csv`
- `data/processed/distribution_by_station.csv`

### 图形化摘要

**Note**: Visualization figures can be generated using the provided tools:

1. **Station Distribution Map** - Generate interactive map of CIMIS stations:
   ```bash
   python scripts/tools/generate_station_map.py
   ```
   This creates `docs/figures/station_distribution_map.html` with an interactive map showing all 18 CIMIS station locations.

2. **Data Analysis Plots** - Generate analysis plots using CLI:
   ```bash
   python -m src.cli analysis full --config config/analysis.yaml
   ```

---

## 📚 相关文档

- **[USER_GUIDE.md](USER_GUIDE.md)**: 用户指南
- **[FEATURE_GUIDE.md](FEATURE_GUIDE.md)**: 特征工程完整指南
- **[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md)**: 训练和评估文档
