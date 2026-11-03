# StockNet Target Label & Split Statistics (Phase 3)

### Target Logic
- **DOWN (0):** Return $\le -0.005$
- **UP (1):** Return $> +0.0055$
- **Excluded:** Return $\in (-0.005, 0.0055]$

### Split Distribution Comparison with Reference Values

| Dataset Split | Date Range | Current Observed Count | Prior Reference Count | Match Status |
| :--- | :--- | :--- | :--- | :---: |
| **Training** | `2014-01-02` to `2015-07-31` | **20,348** | 20,348 | **EXACT MATCH** |
| **Validation** | `2015-08-03` to `2015-09-30` | **2,555** | 2,555 | **EXACT MATCH** |
| **Test** | `2015-10-01` to `2015-12-31` | **3,720** | 3,720 | **EXACT MATCH** |
| **Total Labeled** | `2014-01-02` to `2015-12-31` | **26,623** | 26,623 | **EXACT MATCH** |

### Class Balance
- Overall DOWN (0): 13,255 (49.79%)
- Overall UP (1): 13,368 (50.21%)
