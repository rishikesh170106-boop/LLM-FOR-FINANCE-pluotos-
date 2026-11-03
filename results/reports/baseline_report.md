# Phase 5: Baseline Benchmarking Report

### Benchmark Comparison Matrix

| Model | Split | Accuracy | Macro-F1 | MCC | Prediction Dist (DOWN / UP) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Majority Class** | Validation | 44.58% | 30.83% | 0.0 | 0.0% / 100.0% |
| **Majority Class** | Test | 51.29% | 33.9% | 0.0 | 0.0% / 100.0% |
| **Random Forest (Lagged)** | Validation | **53.39%** | **53.1%** | **0.1006** | 36.75% / 63.25% |
| **Random Forest (Lagged)** | Test | **52.53%** | **52.09%** | **0.0469** | 41.72% / 58.28% |

### Top Technical Features by Importance

| Rank | Feature Indicator | Gini Importance |
| :---: | :--- | :---: |
| 1 | `lag_1_return` | 0.0825 |
| 2 | `lag_5_return` | 0.0556 |
| 3 | `roll_momentum_5d` | 0.0510 |
| 4 | `lag_2_return` | 0.0495 |
| 5 | `roll_return_mean_5d` | 0.0488 |
| 6 | `lagged_rsi_14` | 0.0461 |
| 7 | `lag_2_log_volume` | 0.0438 |
| 8 | `lag_3_log_volume` | 0.0436 |
| 9 | `lag_4_return` | 0.0396 |
| 10 | `lag_2_low_ret` | 0.0372 |
