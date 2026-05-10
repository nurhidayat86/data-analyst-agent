---
description: Default analysis — basic statistics, distributions, correlations, and missing values
use_case: Quick overview of any dataset, ideal for initial data exploration
parameters:
  - df: input DataFrame
tags:
  - exploratory
  - statistics
  - distributions
  - correlations
  - missing-values
  - summary
---

"""Default exploratory data analysis — stats, distributions, correlations."""

import numpy as np
import pandas as pd

from adk.tools.chart_generator import (
    histogram,
    heatmap,
    bar,
    boxplot,
)

# --- Configuration ---
NUMERIC_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
CATEGORICAL_COLS = df.select_dtypes(include=['object', 'category']).columns.tolist()
N_TOP_BARS = 10

# --- Missing values ---
null_counts = df.isna().sum()
null_pct = (df.isna().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'null_count': null_counts,
    'null_percentage': null_pct,
}, index=df.columns).sort_values('null_percentage', ascending=False)

# --- Basic statistics ---
numeric_stats = df[NUMERIC_COLS].describe().round(4)

# --- Correlation (top pairs) ---
if len(NUMERIC_COLS) > 1:
    corr_matrix = df[NUMERIC_COLS].corr()
    # Find top correlations
    top_corr_pairs = []
    for i in range(len(NUMERIC_COLS)):
        for j in range(i + 1, len(NUMERIC_COLS)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:
                top_corr_pairs.append({
                    'feature_1': NUMERIC_COLS[i],
                    'feature_2': NUMERIC_COLS[j],
                    'correlation': round(corr_val, 4),
                })
    top_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
else:
    top_corr_pairs = []

# --- Charts ---
charts = []

# Histograms for top 3 numeric columns
for col in NUMERIC_COLS[:3]:
    charts.append(histogram(df, x=col, title=f'Distribution of {col}'))

# Correlation heatmap
if len(NUMERIC_COLS) >= 2:
    charts.append(heatmap(df, title='Correlation Heatmap'))

# Bar chart for top categorical column
if CATEGORICAL_COLS:
    top_cat = CATEGORICAL_COLS[0]
    charts.append(bar(
        df,
        x=top_cat,
        title=f'Count by {top_cat}',
    ))

# --- Results ---
total_nulls = int(df.isna().sum().sum())
total_cells = df.shape[0] * df.shape[1]
completeness = round((1 - total_nulls / total_cells) * 100, 2)

results = {
    "summary": (
        f"Dataset overview: {df.shape[0]} rows x {df.shape[1]} columns. "
        f"{len(NUMERIC_COLS)} numeric, {len(CATEGORICAL_COLS)} categorical. "
        f"Data completeness: {completeness}%. "
        f"{total_nulls} missing values found across {int((missing_df['null_percentage'] > 0).sum())} columns."
    ),
    "findings": [
        f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"Numeric columns: {len(NUMERIC_COLS)}, Categorical columns: {len(CATEGORICAL_COLS)}",
        f"Data completeness: {completeness}%",
        f"Missing values: {total_nulls} total ({int((missing_df['null_percentage'] > 0).sum())} columns affected)",
        f"Strongest correlation: {top_corr_pairs[0]['feature_1']} <-> {top_corr_pairs[0]['feature_2']} "
        f"(r={top_corr_pairs[0]['correlation']})" if top_corr_pairs else "No strong correlations found (|r| > 0.3)",
    ],
    "charts": charts,
    "details": (
        f"=== Missing Values ===\n{missing_df.to_string()}\n\n"
        f"=== Numeric Statistics ===\n{numeric_stats.to_string()}\n\n"
        f"=== Top Correlations (|r| > 0.3) ===\n" + "\n".join(
            f"  {p['feature_1']} <-> {p['feature_2']}: {p['correlation']}"
            for p in top_corr_pairs[:10]
        ) if top_corr_pairs else "  No strong correlations."
    ),
}
