# 💧 Water Potability Classification

Classifies water as **potable** or **not potable** using physicochemical properties.  
Handles missing data, class imbalance, scaling, and hyperparameter tuning.

## Dataset
- **File:** `water_potability.csv`
- **Source:** [Kaggle](https://www.kaggle.com/datasets/rithikkotha/water-portability-dataset)
- **Target:** `Potability` (0 = Not Potable, 1 = Potable)

## Requirements
```bash
pip install numpy pandas scikit-learn imbalanced-learn tabulate
````

## ▶ Run

```bash
python3 water_potability.py
```

## Key Features

* **Class Imbalance:** No Resampling, OverSampling, UnderSampling, SMOTE
* **Missing Values:** Mean, Random Sample, Hot Deck (custom imputers)
* **Scaling:** MinMaxScaler, StandardScaler (log), PowerTransformer
* **Models:** Dummy, Logistic Regression, KNN, Random Forest
* **Optimization:** GridSearchCV (F1 Score focus)

## Output

* Accuracy, F1, Precision, Recall
* Confusion matrices
* Best hyperparameters
