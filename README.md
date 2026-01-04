# COCOMO Analysis - Software Effort Estimation

This repository contains Jupyter notebooks for COCOMO (Constructive Cost Model) analysis to predict software development effort using the COCOMO81 dataset.

## Notebooks

### 1. Original Analysis
- **File**: [Cocomo_Analysis_Experiments.ipynb](Cocomo_Analysis_Experiments.ipynb)
- Basic implementation using Random Forest and Decision Tree
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImrulMir15/cocomo-analysis/blob/main/Cocomo_Analysis_Experiments.ipynb)

### 2. Improved Analysis (Recommended)
- **File**: [Improved_Cocomo_Analysis.ipynb](Improved_Cocomo_Analysis.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImrulMir15/cocomo-analysis/blob/main/Improved_Cocomo_Analysis.ipynb)

## Key Improvements in the New Notebook

| Aspect | Original | Improved |
|--------|----------|----------|
| Target Transformation | None | Log transformation for skewed data |
| Feature Scaling | None | StandardScaler for sensitive models |
| Models Compared | 2-3 | 10 different algorithms |
| Cross-Validation | Basic | Robust 5-fold CV |
| Feature Analysis | Limited | Comprehensive importance analysis |
| Visualization | Basic | Comprehensive comparison plots |

### Improvements Summary

1. **Log Transformation**: Applied to target variable (effort) to handle the highly skewed distribution, improving model predictions significantly.

2. **Feature Scaling**: StandardScaler applied for models sensitive to feature scales (SVR, KNN, linear models).

3. **Multiple Models Compared**:
   - Tree-based: Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Decision Tree
   - Linear: Ridge, Lasso, ElasticNet
   - Distance-based: K-Nearest Neighbors, SVR

4. **Robust Cross-Validation**: 5-fold cross-validation for all models to get reliable performance estimates.

5. **Comprehensive Metrics**:
   - MMRE (Mean Magnitude of Relative Error)
   - MdMRE (Median MRE)
   - Pred(25) - Percentage within 25% of actual
   - R² Score
   - RMSE

6. **Feature Importance Analysis**: Identifies key effort drivers (LOC is the most important predictor).

## Dataset

- **File**: `cocomo811.arff`
- **Source**: PROMISE Software Engineering Repository
- **Instances**: 63 software projects
- **Features**: 15 COCOMO effort multipliers + 1 size metric (LOC)
- **Target**: Actual development effort

## Requirements

```
scipy
pandas
numpy
matplotlib
scikit-learn
```

## Usage

1. Clone the repository
2. Install requirements: `pip install scipy pandas numpy matplotlib scikit-learn`
3. Open the notebook in Jupyter or Google Colab
4. Run all cells to see the analysis results