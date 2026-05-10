"""Chart generator — creates Plotly charts for data analysis.

Provides a library of common chart types used in banking data analysis:
scatter plots, histograms, boxplots, correlation heatmaps, bar charts,
ROC curves, and feature importance plots.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

OUTPUT_DIR = Path("output/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chart type registry for tool discovery
CHART_REGISTRY = {
    "scatter": {
        "description": "Scatter plot for bivariate relationships",
        "params": ["x", "y", "color", "title"],
        "use_case": "Exploring relationships between two numeric variables",
    },
    "histogram": {
        "description": "Histogram for univariate distribution",
        "params": ["x", "color", "title"],
        "use_case": "Distribution of a numeric variable",
    },
    "bar": {
        "description": "Bar chart for categorical counts or sums",
        "params": ["x", "y", "color", "title"],
        "use_case": "Comparing values across categories",
    },
    "boxplot": {
        "description": "Box plot for distribution by category",
        "params": ["x", "y", "color", "title"],
        "use_case": "Comparing distributions across groups",
    },
    "heatmap": {
        "description": "Correlation heatmap",
        "params": ["df", "title"],
        "use_case": "Correlation matrix of numeric variables",
    },
    "roc": {
        "description": "ROC curve for classification",
        "params": ["y_true", "y_prob", "title"],
        "use_case": "Model classification performance",
    },
    "feature_importance": {
        "description": "Feature importance bar chart",
        "params": ["features", "importances", "title"],
        "use_case": "Feature importance from tree models",
    },
    "pie": {
        "description": "Pie chart for proportions",
        "params": ["labels", "values", "title"],
        "use_case": "Proportional breakdown of categories",
    },
    "violin": {
        "description": "Violin plot for distribution density",
        "params": ["x", "y", "color", "title"],
        "use_case": "Distribution shape by category",
    },
}


def _save_plot(fig: Any, chart_type: str, **kwargs) -> Path:
    """Save a Plotly figure to a file.

    Args:
        fig: Plotly figure object.
        chart_type: Type of chart (for filename).
        **kwargs: Additional kwargs for filename uniqueness.

    Returns:
        Path to the saved HTML file.
    """
    # Generate unique filename based on chart type and kwargs
    key = f"{chart_type}_{hashlib.md5(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()[:8]}"
    output_path = OUTPUT_DIR / f"{key}.html"
    fig.write_html(str(output_path))
    return output_path


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str = "Scatter Plot",
) -> Path:
    """Create a scatter plot.

    Args:
        df: DataFrame containing the data.
        x: Column name for x-axis.
        y: Column name for y-axis.
        color: Column name for coloring (optional).
        title: Plot title.

    Returns:
        Path to saved HTML file.
    """
    fig = px.scatter(df, x=x, y=y, color=color, title=title)
    return _save_plot(fig, "scatter", x=x, y=y, color=color, title=title)


def histogram(
    df: pd.DataFrame,
    x: str,
    color: str | None = None,
    title: str = "Histogram",
    nbins: int = 30,
    color_palette: str | None = None,
) -> Path:
    """Create a histogram.

    Args:
        df: DataFrame containing the data.
        x: Column name for the histogram.
        color: Column name for grouping (optional).
        title: Plot title.
        nbins: Number of bins.

    Returns:
        Path to saved HTML file.
    """
    fig = px.histogram(df, x=x, color=color, title=title, nbins=nbins)
    return _save_plot(fig, "histogram", x=x, title=title, nbins=nbins)


def bar(
    df: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    title: str = "Bar Chart",
    agg_func: str = "count",
    color_palette: str | None = None,  # accepted for compatibility, not used
) -> Path:
    """Create a bar chart.

    Args:
        df: DataFrame containing the data.
        x: Column name for x-axis (optional; if None, uses first column index for crosstab-style DataFrames).
        y: Column name for y-axis values (optional, uses count if None).
        color: Column name for grouping (optional).
        title: Plot title.
        agg_func: Aggregation function (sum, mean, count, etc.).
        color_palette: Accepted for compatibility but not used (Plotly uses its own defaults).

    Returns:
        Path to saved HTML file.
    """
    if x is None:
        # Crosstab-style DataFrame: use index as x
        x_data = df.reset_index()
        fig = px.bar(x_data, x=x_data.columns[0], color=color, title=title)
    elif y is None:
        fig = px.bar(df, x=x, color=color, title=title)
    else:
        fig = px.bar(
            df, x=x, y=y, color=color, title=title,
            aggregate_function=agg_func,
        )
    return _save_plot(fig, "bar", x=x, y=y, title=title)


def boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str = "Box Plot",
    color_palette: str | None = None,  # accepted for compatibility, not used
) -> Path:
    """Create a box plot.

    Args:
        df: DataFrame containing the data.
        x: Column name for x-axis (categorical).
        y: Column name for y-axis (numeric).
        color: Column name for grouping (optional).
        title: Plot title.
        color_palette: Accepted for compatibility but not used.

    Returns:
        Path to saved HTML file.
    """
    fig = px.box(df, x=x, y=y, color=color, title=title)
    return _save_plot(fig, "boxplot", x=x, y=y, title=title)


def heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
    columns: list[str] | None = None,
) -> Path:
    """Create a correlation heatmap.

    Args:
        df: DataFrame with numeric columns.
        title: Plot title.
        columns: Subset of columns to include (optional).

    Returns:
        Path to saved HTML file.
    """
    if columns:
        corr_df = df[columns].select_dtypes(include="number")
    else:
        corr_df = df.select_dtypes(include="number")

    corr = corr_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title=title,
        range_color=[-1, 1],
    )
    return _save_plot(fig, "heatmap", title=title)


def roc_curve(
    y_true: list[int] | pd.Series,
    y_prob: list[float] | pd.Series,
    title: str = "ROC Curve",
) -> Path:
    """Create a ROC curve.

    Args:
        y_true: True labels (0 or 1).
        y_prob: Predicted probabilities.
        title: Plot title.

    Returns:
        Path to saved HTML file.
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC (AUC = {roc_auc:.3f})",
        line=dict(width=2, color="blue"),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(width=1, dash="dash", color="gray"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.7, y=0.3),
    )
    return _save_plot(fig, "roc", title=title)


def feature_importance(
    features: list[str],
    importances: list[float],
    title: str = "Feature Importance",
    top_n: int = 15,
) -> Path:
    """Create a feature importance chart.

    Args:
        features: List of feature names.
        importances: List of importance values.
        title: Plot title.
        top_n: Number of top features to show.

    Returns:
        Path to saved HTML file.
    """
    # Sort by importance
    paired = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    top_pairs = paired[:top_n]
    top_features = [p[0] for p in top_pairs]
    top_values = [p[1] for p in top_pairs]

    fig = go.Figure(go.Bar(
        x=top_values,
        y=top_features,
        orientation="h",
        marker_color="steelblue",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, len(top_features) * 30),
    )
    return _save_plot(fig, "feature_importance", title=title)


def violin_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str = "Violin Plot",
) -> Path:
    """Create a violin plot.

    Args:
        df: DataFrame containing the data.
        x: Column name for x-axis (categorical).
        y: Column name for y-axis (numeric).
        color: Column name for grouping (optional).
        title: Plot title.

    Returns:
        Path to saved HTML file.
    """
    fig = px.violin(df, x=x, y=y, color=color, title=title, box=True)
    return _save_plot(fig, "violin", x=x, y=y, title=title)


def pie_chart(
    df: pd.DataFrame,
    labels: str | None = None,
    values: str | None = None,
    x: str | None = None,  # alias for labels
    y: str | None = None,  # alias for values
    title: str = "Pie Chart",
    color_palette: str | None = None,  # accepted but not used (px.pie has its own defaults)
) -> Path:
    """Create a pie chart.

    Args:
        df: DataFrame containing the data.
        labels: Column name for labels (or use ``x`` as alias).
        values: Column name for values (or use ``y`` as alias).
        x: Alias for ``labels``.
        y: Alias for ``values``.
        title: Plot title.
        color_palette: Accepted for compatibility but not used (Plotly uses its own defaults).

    Returns:
        Path to saved HTML file.
    """
    labels = labels or x
    values = values or y
    if labels is None:
        raise ValueError("pie_chart requires 'labels' (or 'x') parameter.")
    if values is not None:
        pie_data = df.groupby(labels)[values].sum().reset_index()
    else:
        # Auto-count when no values column provided
        pie_data = df.groupby(labels).size().reset_index(name="_count")
        values = "_count"
    fig = px.pie(
        pie_data,
        values=values,
        names=labels,
        title=title,
    )
    fig = px.pie(
        pie_data,
        values=values,
        names=labels,
        title=title,
    )
    return _save_plot(fig, "pie", labels=labels, values=values, title=title)


def get_registry() -> dict[str, Any]:
    """Return the chart type registry for tool discovery."""
    return CHART_REGISTRY
