---
description: Customer segmentation using k-means clustering based on chosen variables
use_case: Group customers into meaningful segments for targeted marketing
parameters:
  - df: input DataFrame
  - n_clusters: number of clusters (default 3-5)
  - features: list of feature columns to cluster on
  - normalize: whether to normalize features before clustering
tags:
  - clustering
  - k-means
  - segmentation
  - customer
  - unsupervised
---

"""Customer segmentation analysis using k-means clustering."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from adk.tools.chart_generator import (
    scatter,
    boxplot,
    bar,
    violin_plot,
)

# --- Configuration ---
N_CLUSTERS = 4
FEATURES = [col for col in df.select_dtypes(include=[np.number]).columns
            if col not in ['cluster']]  # Exclude cluster column if exists
NORMALIZE = True

# --- Analysis ---
# Select features and handle missing values
analysis_features = [f for f in FEATURES if f in df.columns]
df_clean = df[analysis_features].dropna()

if NORMALIZE and len(analysis_features) > 1:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[analysis_features])
else:
    X_scaled = df_clean[analysis_features].values

# --- Elbow method to find optimal clusters ---
inertias = []
k_range = range(2, 11)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)

# --- K-means clustering ---
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# --- Cluster profiling ---
cluster_profiles = df_clean.groupby('cluster')[analysis_features].agg(['mean', 'std']).round(4)
cluster_sizes = df_clean['cluster'].value_counts().sort_index()

# --- Charts ---
# Scatter plot (first two features)
if len(analysis_features) >= 2:
    chart1 = scatter(
        df_clean,
        x=analysis_features[0],
        y=analysis_features[1],
        color='cluster',
        title=f'Customer Clusters (k={N_CLUSTERS})',
    )
    chart2 = boxplot(
        df_clean,
        x='cluster',
        y=analysis_features[0],
        title=f'{analysis_features[0]} by Cluster',
    )
else:
    chart1 = bar(
        df_clean,
        x='cluster',
        title='Cluster Sizes',
    )
    chart2 = None

# --- Results ---
cluster_descriptions = []
for c in sorted(df_clean['cluster'].unique()):
    profile = df_clean[df_clean['cluster'] == c]
    desc = (f"Cluster {c} (n={len(profile)}, {len(profile)/len(df_clean)*100:.1f}%): "
            "distinct group of customers")
    cluster_descriptions.append(desc)

results = {
    "summary": (
        f"K-means clustering identified {N_CLUSTERS} customer segments from "
        f"{len(analysis_features)} features across {len(df_clean)} records. "
        f"Cluster sizes range from {cluster_sizes.min()} to {cluster_sizes.max()} customers."
    ),
    "findings": [
        f"{N_CLUSTERS} clusters found with balanced to imbalanced distribution",
        f"Largest cluster: {cluster_sizes.max()} customers ({cluster_sizes.max()/len(df_clean)*100:.1f}%)",
        f"Smallest cluster: {cluster_sizes.min()} customers ({cluster_sizes.min()/len(df_clean)*100:.1f}%)",
        *cluster_descriptions,
    ],
    "charts": [chart1, chart2] if chart2 else [chart1],
    "details": (
        f"Features used: {', '.join(analysis_features)}\n"
        f"Normalization: {'Yes' if NORMALIZE else 'No'}\n\n"
        f"Cluster sizes:\n{cluster_sizes.to_string()}\n\n"
        f"Cluster profiles (mean values):\n{cluster_profiles.to_string()}"
    ),
}
