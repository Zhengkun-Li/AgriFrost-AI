# Complete Feature Importance Analysis - 3h Horizon
**Total Features**: 298
**Total Importance**: 9940.00

## Top 20 Most Important Features

| Rank | Category | Feature Name | Importance |
|------|----------|--------------|------------|
| 1 | Time | Hour (PST) | 147.00 |
| 2 | Radiation | daily_solar_radiation | 132.00 |
| 3 | Time | hour_cos | 122.00 |
| 4 | Lag | Sol Rad (W/sq.m)_lag_6 | 119.00 |
| 5 | Lag | Vap Pres (kPa)_lag_6 | 117.00 |
| 6 | Time | day_of_week | 111.00 |
| 7 | Wind | wind_dir_sin | 107.00 |
| 8 | Derived | wind_chill | 101.00 |
| 9 | Lag | Rel Hum (%)_lag_24 | 101.00 |
| 10 | Time | month_sin | 97.00 |
| 11 | Rolling | Rel Hum (%)_rolling_6h_max | 96.00 |
| 12 | Time | hour_sin | 95.00 |
| 13 | Lag | Sol Rad (W/sq.m)_lag_3 | 95.00 |
| 14 | Wind | wind_dir_cos | 94.00 |
| 15 | Rolling | Vap Pres (kPa)_rolling_24h_sum | 94.00 |
| 16 | Time | month | 93.00 |
| 17 | Rolling | Vap Pres (kPa)_rolling_24h_std | 92.00 |
| 18 | QC | qc.3 | 91.00 |
| 19 | Lag | Sol Rad (W/sq.m)_lag_12 | 88.00 |
| 20 | Lag | Precip (mm)_lag_6 | 87.00 |

## Category Statistics

| Category | Count | Total Importance | Mean Importance | Max Importance | Min Importance |
|----------|-------|------------------|-----------------|----------------|----------------|
| Rolling | 180 | 4985.00 | 27.69 | 96.00 | 0.00 |
| Lag | 51 | 2303.00 | 45.16 | 119.00 | 0.00 |
| Time | 10 | 862.00 | 86.20 | 147.00 | 14.00 |
| Station | 14 | 406.00 | 29.00 | 69.00 | 0.00 |
| Derived | 14 | 361.00 | 25.79 | 101.00 | 1.00 |
| QC | 10 | 293.00 | 29.30 | 91.00 | 0.00 |
| Wind | 6 | 218.00 | 36.33 | 107.00 | 0.00 |
| Other | 5 | 165.00 | 33.00 | 57.00 | 1.00 |
| Radiation | 2 | 141.00 | 70.50 | 132.00 | 9.00 |
| Humidity | 2 | 105.00 | 52.50 | 59.00 | 46.00 |
| Temperature | 3 | 92.00 | 30.67 | 55.00 | 0.00 |
| Precipitation | 1 | 9.00 | 9.00 | 9.00 | 9.00 |

## Top Features by Category

### Derived (14 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 8 | wind_chill | 101.00 |
| 26 | temp_change_rate | 76.00 |
| 44 | sol_rad_change_rate | 63.00 |
| 70 | radiation_temp_interaction | 53.00 |
| 190 | heat_index | 16.00 |
| 212 | wind_speed_change_rate | 12.00 |
| 232 | wind_dir_temp_interaction | 9.00 |
| 236 | dew_point_proximity | 8.00 |
| 241 | humidity_change_rate | 7.00 |
| 247 | temp_trend | 6.00 |

### Humidity (2 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 59 | Vap Pres (kPa) | 59.00 |
| 84 | Dew Point (C) | 46.00 |

### Lag (51 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 4 | Sol Rad (W/sq.m)_lag_6 | 119.00 |
| 5 | Vap Pres (kPa)_lag_6 | 117.00 |
| 9 | Rel Hum (%)_lag_24 | 101.00 |
| 13 | Sol Rad (W/sq.m)_lag_3 | 95.00 |
| 19 | Sol Rad (W/sq.m)_lag_12 | 88.00 |
| 20 | Precip (mm)_lag_6 | 87.00 |
| 21 | Sol Rad (W/sq.m)_lag_1 | 86.00 |
| 28 | Rel Hum (%)_lag_12 | 75.00 |
| 32 | Dew Point (C)_lag_24 | 72.00 |
| 33 | Air Temp (C)_lag_12 | 72.00 |

### Other (5 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 62 | is_night | 57.00 |
| 72 | season | 52.00 |
| 132 | nighttime_cooling_rate | 31.00 |
| 158 | Rel Hum (%) | 24.00 |
| 277 | cooling_acceleration | 1.00 |

### Precipitation (1 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 231 | Precip (mm) | 9.00 |

### QC (10 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 18 | qc.3 | 91.00 |
| 24 | qc.9 | 80.00 |
| 76 | qc.2 | 49.00 |
| 116 | qc | 36.00 |
| 182 | qc.1 | 18.00 |
| 195 | qc.6 | 15.00 |
| 266 | qc.8 | 3.00 |
| 274 | qc.5 | 1.00 |
| 285 | qc.7 | 0.00 |
| 298 | qc.4 | 0.00 |

### Radiation (2 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 2 | daily_solar_radiation | 132.00 |
| 230 | Sol Rad (W/sq.m) | 9.00 |

### Rolling (180 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 11 | Rel Hum (%)_rolling_6h_max | 96.00 |
| 15 | Vap Pres (kPa)_rolling_24h_sum | 94.00 |
| 17 | Vap Pres (kPa)_rolling_24h_std | 92.00 |
| 22 | Soil Temp (C)_rolling_6h_max | 83.00 |
| 23 | Dew Point (C)_rolling_6h_max | 81.00 |
| 25 | Soil Temp (C)_rolling_3h_max | 79.00 |
| 29 | Air Temp (C)_rolling_12h_max | 74.00 |
| 30 | Precip (mm)_rolling_24h_max | 73.00 |
| 31 | Dew Point (C)_rolling_3h_max | 72.00 |
| 36 | Sol Rad (W/sq.m)_rolling_24h_max | 69.00 |

### Station (14 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 37 | latitude_cos | 69.00 |
| 43 | longitude_sin | 64.00 |
| 51 | distance_to_coast_approx | 62.00 |
| 66 | station_density | 55.00 |
| 81 | longitude_cos | 47.00 |
| 125 | distance_to_nearest_station | 34.00 |
| 162 | latitude_sin | 23.00 |
| 184 | elevation_m | 17.00 |
| 206 | station_id_encoded | 13.00 |
| 242 | elevation_ft | 7.00 |

### Temperature (3 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 65 | Soil Temp (C) | 55.00 |
| 112 | Air Temp (C) | 37.00 |
| 290 | temp_decline_rate | 0.00 |

### Time (10 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 1 | Hour (PST) | 147.00 |
| 3 | hour_cos | 122.00 |
| 6 | day_of_week | 111.00 |
| 10 | month_sin | 97.00 |
| 12 | hour_sin | 95.00 |
| 16 | month | 93.00 |
| 27 | hour | 76.00 |
| 34 | day_of_year | 71.00 |
| 113 | month_cos | 36.00 |
| 204 | Jul | 14.00 |

### Wind (6 features)

| Rank | Feature Name | Importance |
|------|--------------|------------|
| 7 | wind_dir_sin | 107.00 |
| 14 | wind_dir_cos | 94.00 |
| 191 | calm_wind_duration | 16.00 |
| 278 | wind_dir_category | 1.00 |
| 286 | Wind Speed (m/s) | 0.00 |
| 287 | Wind Dir (0-360) | 0.00 |


## Features with Zero Importance (18 features)

### Rolling (11 features)

- Precip (mm)_rolling_24h_mean
- Precip (mm)_rolling_12h_mean
- Sol Rad (W/sq.m)_rolling_6h_mean
- Precip (mm)_rolling_3h_mean
- Vap Pres (kPa)_rolling_24h_mean
- ETo (mm)_rolling_24h_mean
- ETo (mm)_rolling_6h_mean
- ETo (mm)_rolling_3h_mean
- Precip (mm)_rolling_6h_mean
- Vap Pres (kPa)_rolling_3h_mean
- Sol Rad (W/sq.m)_rolling_3h_mean

### QC (2 features)

- qc.7
- qc.4

### Wind (2 features)

- Wind Speed (m/s)
- Wind Dir (0-360)

### Lag (1 features)

- temp_decline_rate_lag_1

### Station (1 features)

- latitude

### Temperature (1 features)

- temp_decline_rate


## Features with Low Importance (<10) (72 features)

| Category | Count | Percentage |
|----------|-------|------------|
| Derived | 8 | 57.1% |
| Lag | 6 | 11.8% |
| Other | 1 | 20.0% |
| Precipitation | 1 | 100.0% |
| QC | 4 | 40.0% |
| Radiation | 1 | 50.0% |
| Rolling | 42 | 23.3% |
| Station | 5 | 35.7% |
| Temperature | 1 | 33.3% |
| Wind | 3 | 50.0% |
