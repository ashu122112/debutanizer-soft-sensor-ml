# Debutanizer Soft Sensor - Stage 1 Data Summary

## Overview
- **Total Rows:** 2394 (Confirmed exactly 2394 rows in chronological time order)
- **Total Columns:** 8 (7 process input sensors `u1` to `u7` plus target `C4`)
- **Missing Values:** 0 missing entries
- **Duplicate Rows:** 0 duplicate rows
- **Value Range:** Min = 0.0000, Max = 1.0000 (Confirmed strictly bounded within [0, 1])
- **Proprietary Notice:** The original refinery units, sensor calibrations, and physical variable definitions were not publicly disclosed by the original authors (Fortuna et al.) to preserve commercial confidentiality.

## Column Data Types & Missing Counts
| Column | Data Type | Missing Count | Min | Max |
|---|---|---|---|---|
| `u1` | float64 | 0 | 0.0000 | 1.0000 |
| `u2` | float64 | 0 | 0.0000 | 1.0000 |
| `u3` | float64 | 0 | 0.0000 | 1.0000 |
| `u4` | float64 | 0 | 0.0000 | 1.0000 |
| `u5` | float64 | 0 | 0.0000 | 1.0000 |
| `u6` | float64 | 0 | 0.0000 | 1.0000 |
| `u7` | float64 | 0 | 0.0000 | 1.0000 |
| `C4` | float64 | 0 | 0.0000 | 1.0000 |

## Descriptive Statistics (describe())
| Metric | u1 | u2 | u3 | u4 | u5 | u6 | u7 | C4 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **count** | 2394.0000 | 2394.0000 | 2394.0000 | 2394.0000 | 2394.0000 | 2394.0000 | 2394.0000 | 2394.0000 |
| **mean** | 0.2839 | 0.6678 | 0.5985 | 0.4359 | 0.6581 | 0.6207 | 0.5894 | 0.2678 |
| **std** | 0.1031 | 0.0577 | 0.1986 | 0.1382 | 0.1271 | 0.1651 | 0.1733 | 0.1589 |
| **min** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **25%** | 0.2240 | 0.6538 | 0.5004 | 0.3634 | 0.6087 | 0.4818 | 0.4450 | 0.1692 |
| **50%** | 0.2640 | 0.6719 | 0.6243 | 0.3963 | 0.6946 | 0.6396 | 0.6010 | 0.2469 |
| **75%** | 0.3256 | 0.6909 | 0.7428 | 0.5181 | 0.7301 | 0.7550 | 0.7414 | 0.3181 |
| **max** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 3-Line Viva Explanation (Stage 1)
> 1. The dataset contains 2,394 sequential operational time samples of a refinery debutanizer column with zero missing values.
> 2. All 7 process variables (u1-u7) and the bottom butane target (C4) are normalized between 0 and 1 by the plant owners for commercial confidentiality.
> 3. Preserving strict chronological ordering without random shuffling is crucial because process measurements are time-dependent and have dynamic memory.