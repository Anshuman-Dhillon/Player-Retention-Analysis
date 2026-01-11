# Gaming Player Retention - Causal Analysis

## Overview
Analyzes whether early game achievements cause players to stay engaged, or if engaged players simply achieve more. Uses causal inference methods to separate correlation from causation.

### Dataset

- Source: Kaggle - "Predict Online Gaming Behavior Dataset"
- Size: 40,034 player records
- Features: Age, playtime, achievements, session frequency, engagement level

### Methods

- Causal Inference: DoWhy (propensity score weighting, refutation tests)
- Machine Learning: EconML Causal Forest, Random Forest regressors
- Data Processing: pandas, NumPy
- Visualization: Matplotlib, Seaborn

### Key Findings

- Observed correlation: 1.9 percentage point difference in retention between high/low achievement players
- Causal effect: 1.6 percentage points after controlling for confounders
- Interpretation: ~75% of the relationship is due to player characteristics (selection bias), not the achievements themselves

### Technical Details

- Automated confounder detection and validation
- Multicollinearity removal (correlation > 0.95)
- Feature standardization for numerical stability
- k-NN fallback when Causal Forest fails
- Propensity score weighting with refutation tests
