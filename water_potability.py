

"""
This script implements a machine learning pipeline for classifying water potability 
based on physicochemical features.

The code addresses several data challenges in stages:
1. Class Imbalance : Evaluates different resampling techniques
    (no resampling, oversampling, undersampling, SMOTE) using baseline and advanced classifiers.
2. Missing Values  : Tests multiple imputation strategies 
   (mean, random sampling, and hot deck) to fill missing data and assess model performance.
3. Feature Scaling : Compares performance impact of various scaling approaches 
   (MinMax, StandardScaler with log transform, PowerTransformer) to handle feature disparity.
4. Hyperparameters : Uses grid search with cross-validation to fine-tune 
   logistic regression, k-nearest neighbors, and random forest models for optimal F1 score.

Includes:
- Custom imputers (`RandomSampleImputer`, `HotDeckImputer`)
- Modularized evaluation and metric reporting
- Confusion matrix visualization
- Easily extensible classifier configuration

Target Variable:
- `Potability` (0 = Not potable, 1 = Potable)

Input:
- A CSV file (`water_potability.csv`) with water quality measurements.

Output:
- Tabulated performance metrics and confusion matrices for model comparison.
"""

import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    StandardScaler,
)
from tabulate import tabulate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# === Constants ===
DATA_PATH = "./water_potability.csv"

# Predefined classifiers with parameters tuned for initial experimentation
CLASSIFIERS = {
    "Dummy classifier (baseline)": DummyClassifier(
        strategy="stratified", random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='lbfgs',
        max_iter=2000,
        random_state=42,
        class_weight='balanced',
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=3,
        p=1,
        weights='distance',
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=150,
        min_samples_split=2,
        max_depth=None,
        random_state=42,
        class_weight='balanced',
    ),
}


# === Custom Imputers ===
class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """Impute missing values by random sampling from existing column values."""

    def fit(self, X, y=None):
        self.random_state = 42
        return self

    def transform(self, X):
        Xc = X.copy()
        rng = np.random.RandomState(self.random_state)
        for col in X.columns:
            if X[col].isna().sum() > 0:
                vals = X[col].dropna().values
                Xc.loc[Xc[col].isna(), col] = rng.choice(
                    vals, size=X[col].isna().sum(), replace=True
                )
        return Xc


class HotDeckImputer(BaseEstimator, TransformerMixin):
    """Impute missing values using the most similar complete row."""

    def fit(self, X, y=None):
        self.complete = X.dropna()
        return self

    def transform(self, X):
        Xc = X.copy()
        for idx in Xc[Xc.isna().any(axis=1)].index:
            missing = Xc.columns[Xc.loc[idx].isna()]
            donor = self.complete.sample(1, random_state=idx).iloc[0]
            for col in missing:
                Xc.at[idx, col] = donor[col]
        return Xc


# === Utility Functions ===
def print_metrics_table(title, metrics):
    """Print evaluation metrics in a formatted table."""
    print(f"\n=== {title} ===")
    df = pd.DataFrame.from_dict(metrics, orient="index")
    print(tabulate(df, headers="keys", tablefmt="grid"))


def evaluate_models(X_train, X_test, y_train, y_test, classifiers):
    """Train and evaluate models, returning performance metrics and confusion matrices."""
    results = {
        metric: {} for metric in [
            "Accuracy (test)",
            "F1 Score (Class 1)",
            "Precision (Class 1)",
            "Recall (Class 1)"
        ]
    }
    conf_matrices = {}

    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Capture evaluation metrics
        results["Accuracy (test)"][clf_name] = round(
            accuracy_score(y_test, y_pred) * 100, 2
        )
        results["F1 Score (Class 1)"][clf_name] = round(
            f1_score(y_test, y_pred, pos_label=1), 4
        )
        results["Precision (Class 1)"][clf_name] = round(
            precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4
        )
        results["Recall (Class 1)"][clf_name] = round(
            recall_score(y_test, y_pred, pos_label=1), 4
        )

        # Save confusion matrix
        conf_matrices[clf_name] = confusion_matrix(y_test, y_pred)

    return results, conf_matrices


# === Challenge 1: Class Imbalance ===
def run_challenge_1(df):
    """Evaluate the impact of different resampling strategies on class imbalance."""
    print("\n======================== Challenge 1: Class imbalance =========================")

    # Split and impute data
    X, y = df.drop("Potability", axis=1), df["Potability"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Show actual distribution in training set
    y_train_counts = pd.Series(y_train).value_counts()
    print(f"\nOriginal training set class counts:\n{y_train_counts.to_string()}")

    # Prepare samplers
    ros = RandomOverSampler(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    smote = SMOTE(random_state=42)

    # Apply each sampler to training data
    X_ros, y_ros = ros.fit_resample(X_train, y_train)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    # Report sampled class distributions
    print(f"\nAfter Oversampling:\n{pd.Series(y_ros).value_counts().to_string()}")
    print(f"\nAfter Undersampling:\n{pd.Series(y_rus).value_counts().to_string()}")
    print(f"\nAfter SMOTE:\n{pd.Series(y_smote).value_counts().to_string()}")

    # Report number of synthetic examples created by SMOTE
    original_positives = pd.Series(y_train).value_counts()[1]
    smote_positives = pd.Series(y_smote).value_counts()[1]
    synthetic_samples = smote_positives - original_positives
    print(f"\nSynthetic samples generated by SMOTE: {synthetic_samples}")

    # Define resampling strategies for full evaluation
    resampling_strategies = {
        "No Resampling": None,
        "Oversampling": ros,
        "Undersampling": rus,
        "SMOTE": smote,
    }

    metrics_all, conf_all = {}, {}

    # Evaluate each resampling strategy
    for name, sampler in resampling_strategies.items():
        X_res, y_res = (
            sampler.fit_resample(X_train, y_train)
            if sampler else (X_train, y_train)
        )
        metrics, confs = evaluate_models(X_res, X_test, y_res, y_test, CLASSIFIERS)

        for key in metrics:
            metrics_all.setdefault(key, {})[name] = metrics[key]
        conf_all[name] = confs

    for key in metrics_all:
        print_metrics_table(key, metrics_all[key])

#if required, Print confusion matrices for analaysis purpose
#    print("\n================ Confusion Matrices =====================")
#    for strat, confs in conf_all.items():
#        print(f"\n--- {strat} ---")
#        for clf, cm in confs.items():
#         


# === Challenge 2: Handling Missing Values ===
def run_challenge_2(df):
    """Assess performance under various missing value imputation methods."""
    print("\n========================= Challenge 2: Missing Values ==========================")

    imputers = {
        "Mean Imputation": SimpleImputer(strategy="mean"),
        "Random Sample Imputation": RandomSampleImputer(),
        "Hot Deck Imputation": HotDeckImputer(),
    }

    scaler = StandardScaler()
    undersampler = RandomUnderSampler(random_state=42)

    metrics_all, conf_all = {}, {}
    X, y = df.drop("Potability", axis=1), df["Potability"]

    for name, imp in imputers.items():
        # Split, impute, and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=42
        )
        X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X.columns)
        X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X.columns)

        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        X_train_bal, y_train_bal = undersampler.fit_resample(X_train_scaled, y_train)
        metrics, confs = evaluate_models(
            X_train_bal, X_test_scaled, y_train_bal, y_test, CLASSIFIERS
        )

        for key in metrics:
            metrics_all.setdefault(key, {})[name] = metrics[key]
        conf_all[name] = confs

    for key in metrics_all:
        print_metrics_table(key, metrics_all[key])

#if required, Print confusion matrices for analaysis purpose
#    print("\n================== Confusion Matrices ====================")
#    for imp, confs in conf_all.items():
#        print(f"\n--- {imp} ---")
#        for clf, cm in confs.items():
#            print(f"\nConfusion Matrix for {clf}:\n{cm}")


# === Challenge 3: Feature Scaling & Disparities ===
def run_challenge_3(df):
    """Evaluate the effect of various feature scaling techniques on model performance."""
    print("\n================ Challenge 3: Feature Scaling & Unit Disparities ================")

    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    X = df_imputed.drop("Potability", axis=1)
    y = df_imputed["Potability"]

    # Different scaling strategies
    scaling_strategies = {
        "No Scaling": X.copy(),
        "MinMaxScaler": pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns),
        "PowerTransformer (Yeo-Johnson)": pd.DataFrame(
            PowerTransformer(method='yeo-johnson').fit_transform(X), columns=X.columns
        ),
    }

    # Apply log1p to 'Solids' then standardize
    X_log = X.copy()
    X_log["Solids"] = np.log1p(X_log["Solids"])
    scaling_strategies["Log1p+StandardScaler"] = pd.DataFrame(
        StandardScaler().fit_transform(X_log), columns=X.columns
    )

    metrics_all, conf_all = {}, {}

    for strat_name, X_scaled in scaling_strategies.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.1, stratify=y, random_state=42
        )
        rus = RandomUnderSampler(random_state=42)
        X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
        metrics, confs = evaluate_models(
            X_train_bal, X_test, y_train_bal, y_test, CLASSIFIERS
        )

        for key in metrics:
            metrics_all.setdefault(key, {})[strat_name] = metrics[key]
        conf_all[strat_name] = confs

    for key in metrics_all:
        print_metrics_table(key, metrics_all[key])
# if required, Print confusion matrices for analaysis purpose
#    print("\n================== Confusion Matrices ====================")
#    for strat, confs in conf_all.items():
#        print(f"\n--- {strat} ---")
#        for clf, cm in confs.items():
#            print(f"\nConfusion Matrix for {clf}:\n{cm}")


# === Hyperparameters ===
def run_hyperparameter_tuning(df):
    """Perform hyperparameter tuning using GridSearchCV for multiple models."""
    print("\n========================= Hyperparameters ==========================")

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scorer = make_scorer(f1_score, pos_label=1)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=3000, random_state=42)
    log_params = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2'],
    }
    log_search = GridSearchCV(log_reg, log_params, cv=5, scoring=scorer)
    log_search.fit(X_train_imputed, y_train)

    # K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn_params = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
    }
    knn_search = GridSearchCV(knn, knn_params, cv=5, scoring=scorer)
    knn_search.fit(X_train_imputed, y_train)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 4, 6],
    }
    rf_search = GridSearchCV(rf, rf_params, cv=5, scoring=scorer)
    rf_search.fit(X_train_imputed, y_train)

    # Output best parameters
    print("\nBest Hyperparameters")
    print("Logistic Regression:", log_search.best_params_)
    print("KNN:", knn_search.best_params_)
    print("Random Forest:", rf_search.best_params_)


if __name__ == "__main__":
    # Load dataset and run challenges
    df = pd.read_csv(DATA_PATH)
    run_challenge_1(df)
    run_challenge_2(df)
    run_challenge_3(df)
    run_hyperparameter_tuning(df)
