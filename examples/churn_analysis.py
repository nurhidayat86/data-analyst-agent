---
description: Root-cause analysis for customer churn or payment defaults using feature importance
use_case: Understand why customers churn or defaults occur, identify key driving factors
parameters:
  - df: input DataFrame
  - target_col: column name for the target (churn/defaults, binary)
  - n_top_features: number of top features to display (default 15)
tags:
  - churn
  - root-cause
  - feature-importance
  - classification
  - defaults
  - payment
---

"""Root-cause analysis for churn or payment defaults."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from adk.tools.chart_generator import (
    feature_importance,
    roc_curve,
    bar,
    histogram,
)

# --- Configuration ---
TARGET_COL = None  # Auto-detect binary columns if None
N_TOP_FEATURES = 15

# --- Auto-detect target column ---
if TARGET_COL is None:
    # Look for binary columns that might be the target
    binary_cols = df.select_dtypes(include=[np.number]).columns[
        df.select_dtypes(include=[np.number]).nunique() == 2
    ]
    if len(binary_cols) > 0:
        TARGET_COL = binary_cols[0]
    else:
        TARGET_COL = df.select_dtypes(include=[np.number]).columns[0]

# --- Prepare data ---
target = df[TARGET_COL]
feature_cols = [c for c in df.columns if c != TARGET_COL]
feature_cols = [c for c in feature_cols if df[c].dtype in [np.number, 'int64', 'float64', 'int32', 'float32']]

# Handle categorical targets
if target.nunique() > 2:
    # Multi-class: treat as binary (most common vs others)
    most_common = target.mode()[0]
    target = (target == most_common).astype(int)

df_model = df[feature_cols + [TARGET_COL]].dropna()
X = df_model[feature_cols]
y = target

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Fill any remaining NaN
X = X.fillna(0)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# --- Feature importance ---
importances = rf.feature_importances_
feature_names = X.columns.tolist()
idx = np.argsort(importances)[::-1][:N_TOP_FEATURES]
top_features = [feature_names[i] for i in idx]
top_importances = [float(importances[i]) for i in idx]

# --- Charts ---
chart1 = feature_importance(
    top_features,
    top_importances,
    title=f'Top {N_TOP_FEATURES} Features Driving {TARGET_COL}',
)

if len(np.unique(y)) == 2:
    chart2 = roc_curve(y_test, y_prob, title='ROC Curve')
else:
    chart2 = None

# --- Results ---
# Calculate per-feature impact
feature_impact = {}
for feat in top_features[:5]:
    col_idx = feature_names.index(feat)
    mean_val = float(X[feat].mean())
    impact_score = float(importances[col_idx]) * mean_val
    feature_impact[feat] = round(impact_score, 4)

results = {
    "summary": (
        f"Random Forest analysis identified the top drivers of {TARGET_COL}. "
        f"Model accuracy: {accuracy_score(y_test, y_pred):.3f}, "
        f"AUC: {roc_auc_score(y_test, y_prob):.3f}. "
        f"Top feature: {top_features[0]} (importance: {top_importances[0]:.4f})."
    ),
    "findings": [
        f"Top driver: {top_features[0]} with importance {top_importances[0]:.4f}",
        f"Model accuracy: {accuracy_score(y_test, y_pred):.3f}",
        f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}",
        f"Training samples: {len(X_train)}, Test samples: {len(X_test)}",
        f"Number of features analyzed: {len(feature_names)}",
    ],
    "charts": [chart1, chart2] if chart2 else [chart1],
    "details": (
        f"Target column: {TARGET_COL}\n"
        f"Features used: {len(feature_names)}\n"
        f"Model: Random Forest (100 trees, balanced class weights)\n\n"
        f"Classification Report:\n{classification_report(y_test, y_pred)}\n\n"
        f"Top {N_TOP_FEATURES} Features:\n" + "\n".join(
            f"  {i+1}. {top_features[i]}: {top_importances[i]:.4f}"
            for i in range(min(N_TOP_FEATURES, len(top_features)))
        )
    ),
}
