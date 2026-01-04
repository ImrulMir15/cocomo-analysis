# COCOMO Analysis - Software Effort Estimation

This repository contains advanced machine learning analysis for software effort estimation using the COCOMO 81 dataset.

## 📊 Files

- **`cocomo811.arff`** - COCOMO 81 dataset with 63 software projects and 17 features
- **`Cocomo_Analysis_Experiments.ipynb`** - Original analysis with Random Forest and basic models
- **`Improved_Cocomo_Analysis.ipynb`** - ⭐ **NEW** Enhanced analysis with advanced techniques

## 🚀 Quick Start (Google Colab)

Click the "Open in Colab" badge at the top of the improved notebook to run it directly in Google Colab. The notebook will automatically:
- Download the dataset
- Install required libraries
- Run all analyses

## 🎯 Improvements in New Notebook

The **Improved_Cocomo_Analysis.ipynb** includes the following enhancements:

### 1. **Advanced ML Models**
- XGBoost - Extreme Gradient Boosting
- LightGBM - Light Gradient Boosting Machine
- Gradient Boosting Regressor
- Extra Trees Regressor
- Neural Networks (MLP)
- Original: Only Random Forest and Decision Tree

### 2. **Feature Engineering**
- Polynomial features (degree 2)
- Log transformations for skewed features
- Feature interaction terms
- Original: No feature engineering

### 3. **Better Data Preprocessing**
- Robust Scaler (handles outliers better)
- Feature selection techniques
- Outlier detection and handling
- Original: Basic train-test split only

### 4. **Ensemble Methods**
- Voting Regressor (combines multiple models)
- Stacking Regressor (meta-learning approach)
- Original: Single model predictions only

### 5. **Enhanced Evaluation Metrics**
- MMRE (Mean Magnitude Relative Error)
- MdMRE (Median Magnitude Relative Error)
- Pred(25) - Prediction within 25%
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Original: Limited metrics

### 6. **Cross-Validation**
- K-Fold cross-validation (5 folds)
- Multiple scoring metrics
- Statistical significance testing
- Original: Single train-test split

### 7. **Rich Visualizations**
- Model performance comparison charts
- Feature importance plots
- Prediction vs Actual scatter plots
- Residual analysis
- Cross-validation box plots
- Correlation heatmaps
- Original: Basic plots only

### 8. **Hyperparameter Optimization**
- Grid search for best parameters
- Randomized search for efficiency
- Model-specific tuning
- Original: Default parameters mostly

## 📈 Expected Improvements

The new techniques typically provide:
- **5-15% better R² scores**
- **Lower prediction errors (MAE/RMSE)**
- **More robust predictions** with ensemble methods
- **Better generalization** through cross-validation
- **Deeper insights** through advanced visualizations

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/ImrulMir15/cocomo-analysis.git
cd cocomo-analysis

# Install required packages
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost lightgbm
```

## 🔧 Usage

### Local Jupyter:
```bash
jupyter notebook Improved_Cocomo_Analysis.ipynb
```

### Google Colab:
1. Open the notebook file
2. Click "Open in Colab" badge
3. Run all cells (Runtime → Run all)

## 📊 Dataset Information

The COCOMO 81 dataset contains:
- **63 projects** from various domains
- **16 features** including project attributes
- **1 target variable** (effort in person-months)
- Source: Boehm's Software Engineering Economics (1981)

## 🎓 Key Learnings

1. **Ensemble methods** consistently outperform single models
2. **Feature engineering** significantly improves predictions
3. **Cross-validation** is essential for reliable evaluation
4. **Multiple metrics** provide better understanding than R² alone
5. **Robust scaling** helps with outliers in small datasets

## 📝 License

Open source - feel free to use and modify for your projects!

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note:** Both notebooks are fully functional and can be run in Google Colab. The improved version provides significantly better predictions and insights for software effort estimation.
