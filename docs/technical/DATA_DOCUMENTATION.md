# AgriFrost-AI Data Documentation

<div align="center">

<img src="logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**Last Updated**: 2025-12-06

This document provides detailed information about the data format, QC processing, variable usage, and data analysis reports for the **AgriFrost-AI** project.

## 📋 Table of Contents

1. [Data Overview](#data-overview)
2. [Field Descriptions](#field-descriptions)
3. [QC Processing](#qc-processing)
4. [Variable Usage](#variable-usage)
5. [Data Analysis Reports](#data-analysis-reports)

---

## Data Overview

### Data Source

- **Source**: F3 Innovate Frost Risk Forecast Challenge official repository (2025)
- **Location**: `data/raw/frost-risk-forecast-challenge/`
- **Format**: CSV files

### Data Scale

- **Time Span**: 2010-09-28 to 2025-09-28
- **Temporal Resolution**: Hourly
- **Number of Stations**: 18
- **Total Records**: Approximately 2.36 million rows

### Data Structure

```
data/raw/frost-risk-forecast-challenge/
├── cimis_all_stations.csv.gz    # Combined file for all stations
└── stations/                     # Per-station CSV files
    ├── 2-FivePoints.csv
    ├── 7-Firebaugh_Telles.csv
    └── ... (18 stations total)
```

---

## Field Descriptions

### Field Structure Overview

| Field | Description | Notes |
|-------|-------------|-------|
| `Stn Id` | Station ID | Can be treated as categorical variable |
| `Stn Name` | Station name | One-to-one correspondence with `Stn Id` |
| `CIMIS Region` | CIMIS official regional classification | Can be used for regional difference analysis |
| `Date` | Date (yyyy-mm-dd) | Converted to datetime |
| `Hour (PST)` | Local time (PST, discrete encoding) | 0000 / 0100 … 2300 |
| `Jul` | Day of year | Can be used for seasonal features |
| `ETo (mm)` | Reference evapotranspiration | Numeric |
| `Precip (mm)` | Precipitation | Numeric |
| `Sol Rad (W/sq.m)` | Solar radiation | Contains sentinel value -6999 |
| `Vap Pres (kPa)` | Vapor pressure | Numeric |
| `Air Temp (C)` | Air temperature | One of the main frost targets |
| `Rel Hum (%)` | Relative humidity | Contains extreme negative values, needs processing |
| `Dew Point (C)` | Dew point | Numeric |
| `Wind Speed (m/s)` | Wind speed | ≥0.2 m/s |
| `Wind Dir (0-360)` | Wind direction | 0–360 |
| `Soil Temp (C)` | Soil temperature | Contains sentinel value -6999 |
| `qc` ~ `qc.9` | Quality control flags for each physical variable | Encodings: blank, `Y`, `M`, `R`, `Q`, `S`, `P` |

> Note: Fields with `qc` prefix indicate the quality level of the previous physical variable in the same row, e.g., `qc.4` corresponds to `Air Temp (C)`.

### Physical Meaning of Variables

These fields are meteorological features used to monitor and predict frost risk, helping models characterize surface energy and water balance, air state, and ground cooling conditions:

- **ETo (mm)**: Reference evapotranspiration, describes surface water evaporation capacity. High evapotranspiration means enhanced surface heat dissipation, making it easier to cool to frost threshold at night.
- **Precip (mm)**: Precipitation, recent precipitation increases soil and air humidity, slowing nighttime cooling; dry conditions increase frost risk.
- **Sol Rad (W/sq.m)**: Solar radiation, daytime radiation accumulation determines heat that can be lost at night; reduced cloud cover makes nighttime radiative cooling stronger.
- **Vap Pres (kPa)**: Vapor pressure, reflects water vapor content in air; less water vapor means stronger longwave radiative cooling at night, higher frost probability.
- **Air Temp (C)**: Air temperature, core indicator of frost risk, frost damage occurs when below crop critical temperature.
- **Rel Hum (%)**: Relative humidity, affects dew point and frost formation; extreme low humidity makes temperature drop faster.
- **Dew Point (C)**: Dew point, temperature at which air reaches saturation. Frost often occurs when air temperature approaches or falls below dew point.
- **Wind Speed (m/s)**: Wind speed, weak wind or calm conditions favor radiative frost formation; moderate mixing can alleviate surface cooling.
- **Wind Dir (0-360)**: Wind direction, helps identify cold air intrusion paths and local circulation patterns.
- **Soil Temp (C)**: Soil temperature, intuitive indicator of surface heat storage. Higher soil temperature can supply heat upward at night, reducing frost.

---

## QC Processing

### QC Flag Meanings

According to CIMIS standards, QC flag meanings are as follows:

| QC Flag | Meaning | Processing | Description |
|---------|---------|------------|-------------|
| **Blank** | Passed all quality checks | ✅ **Keep** | High confidence data |
| **Y** | Moderate outlier, accepted | ✅ **Keep** | Slightly deviant but accepted |
| **Q** | Questionable | ❌ **Mark as missing** | Questionable value, default removal |
| **P** | Provisional | ❌ **Mark as missing** | Temporary/imputed value, default removal |
| **M** | Missing data | ❌ **Mark as missing** | Missing data |
| **R** | Rejected | ❌ **Mark as missing** | Severely exceeds threshold, rejected |
| **S** | Severe outlier | ❌ **Mark as missing** | Extreme outlier |

### Processing Workflow

The system uses the `DataCleaner` class to process quality control (QC) flags:

1. **QC Filtering**: Filter low-quality data based on QC flags
   - Automatically detect all QC columns (columns starting with `qc`)
   - Find corresponding QC column for each variable
   - Decide whether to keep data based on QC flags

2. **Sentinel Value Handling**: Replace sentinel values like `-6999`, `-9999` with `NaN`

3. **Missing Value Imputation**: Use forward fill (grouped by station)

### Usage Example

```python
from src.data.cleaners import DataCleaner

cleaner = DataCleaner()
df_cleaned = cleaner.clean_pipeline(df)
```

---

## Variable Usage

### Current Usage

| Variable Name | Usage Status | Description |
|---------------|--------------|-------------|
| `Air Temp (C)` | ✅ Used | Lag, rolling, derived features |
| `Dew Point (C)` | ✅ Used | Lag, rolling, derived features |
| `Rel Hum (%)` | ✅ Used | Lag, rolling, derived features |
| `Wind Speed (m/s)` | ✅ Used | Lag, rolling, derived features |
| `Wind Dir (0-360)` | ✅ Used | Periodic encoding, lag features |
| `Sol Rad (W/sq.m)` | ✅ Used | Lag, rolling, derived features |
| `Soil Temp (C)` | ✅ Used | Lag, rolling features |
| `ETo (mm)` | ✅ Used | Lag, rolling features |
| `Precip (mm)` | ✅ Used | Lag, rolling features |
| `Vap Pres (kPa)` | ✅ Used | Lag, rolling features |

### Variable Usage Rate

- **Before**: 31% (5/16)
- **Now**: 81% (13/16)
- **Improvement**: +50%

---

## Data Analysis Reports

### Data Overview Statistics

- **Total Records**: 2,367,360
- **Number of Fields**: 26
- **Number of Stations**: 18
- **Time Range**: 2010-09-28T00:00:00 to 2025-09-28T00:00:00
- **Temperature Distribution (°C)**: Mean 17.10 | p1 0.20 | p50 16.20 | p99 37.30

### Key Insights

- **Observations with Highest Missing Rates**:
  - ETo (mm) 2.73%
  - Soil Temp (C) 2.48%
  - Air Temp (C) 0.84%
  - Dew Point (C) 0.82%
  - Rel Hum (%) 0.82%

- **Stations with Most Missing Data**: Coalinga (2.50%), Oakdale (2.33%), Panoche (2.29%)

- **Anomaly Flag Contributions**:
  - dew_point_extreme: 2299
  - humidity_out_of_range: 20
  - sol_rad_sentinel: 18
  - soil_temp_sentinel: 3
  - air_temp_extreme: 2

- **Stations with Most Anomalies**: Parlier (648), Modesto (470), Auburn (253)

- **Extreme Temperature Stations**:
  - Lowest: Modesto (-53.4 °C)
  - Highest: Stratford (48.0 °C)

- **Average Humidity Comparison**:
  - Driest: Coalinga (46.2%)
  - Wettest: Modesto (70.0%)

### Output Files

Data analysis scripts generate the following files:
- `data/processed/station_overview.csv`
- `data/processed/missing_by_station.csv`
- `data/processed/anomalies_by_station.csv`
- `data/processed/distribution_by_station.csv`

### Visualization Summary

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

## 📚 Related Documentation

- **[User Guide](../guides/USER_GUIDE.md)**: User guide
- **[Feature Guide](../features/FEATURE_GUIDE.md)**: Complete feature engineering guide
- **[Training Guide](../training/TRAINING_GUIDE.md)**: Training and evaluation documentation

---

**Last Updated**: 2025-12-06  
**Document Version**: 1.0
