"""Schema discovery — auto-detect column names, types, and distributions from uploaded data.

The agent needs to know the data schema before generating analysis code.
This module extracts schema metadata from pandas DataFrames and returns
a structured description the LLM can use for prompt injection.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def discover_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Discover schema metadata from a DataFrame.

    Args:
        df: pandas DataFrame to analyze.

    Returns:
        Dict with column names, types, distributions, and summary stats.
    """
    schema = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {},
        "sample_values": {},
        "null_counts": {},
        "null_percentages": {},
    }

    for col in df.columns:
        col_data = df[col]
        col_info: dict[str, Any] = {
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isna().sum()),
            "null_percentage": round(float(col_data.isna().sum()) / len(df) * 100, 2),
            "unique_count": int(col_data.nunique()),
        }

        if pd.api.types.is_numeric_dtype(col_data):
            col_info["stats"] = {
                "mean": round(float(col_data.mean()), 4),
                "std": round(float(col_data.std()), 4),
                "min": round(float(col_data.min()), 4),
                "max": round(float(col_data.max()), 4),
                "median": round(float(col_data.median()), 4),
                "quartiles": [
                    round(float(q), 4) for q in col_data.quantile([0.25, 0.5, 0.75]).values
                ],
            }
            col_info["type"] = "numeric"
        elif pd.api.types.is_bool_dtype(col_data):
            col_info["type"] = "boolean"
            col_info["value_counts"] = col_data.value_counts().to_dict()
        else:
            col_info["type"] = "categorical"
            top_values = col_data.value_counts().head(10)
            col_info["value_counts"] = {
                str(k): int(v) for k, v in top_values.items()
            }

        # Store sample values
        schema["sample_values"][col] = col_data.head(3).tolist()
        schema["null_counts"][col] = col_info["null_count"]
        schema["null_percentages"][col] = col_info["null_percentage"]
        schema["columns"][col] = col_info

    return schema


def schema_to_prompt(schema: dict[str, Any]) -> str:
    """Convert schema dict to a prompt-friendly string description.

    Args:
        schema: Schema dict from discover_schema().

    Returns:
        Formatted string for LLM prompt injection.
    """
    lines = [
        "## DATA SCHEMA",
        f"Shape: {schema['shape']['rows']} rows x {schema['shape']['columns']} columns",
        "",
        "### Columns",
    ]

    for col, info in schema["columns"].items():
        lines.append(f"- **{col}** ({info['type']}, {info['dtype']})")
        lines.append(f"  - Nulls: {info['null_percentage']}%")
        lines.append(f"  - Unique values: {info['unique_count']}")

        if info["type"] == "numeric" and "stats" in info:
            s = info["stats"]
            lines.append(f"  - Stats: mean={s['mean']}, std={s['std']}, "
                         f"min={s['min']}, max={s['max']}, median={s['median']}")

        if info["type"] == "categorical" and "value_counts" in info:
            top = info["value_counts"]
            top_items = ", ".join(f"{k}={v}" for k, v in list(top.items())[:5])
            lines.append(f"  - Top values: {top_items}")

    lines.append("")
    lines.append("### Sample Rows")
    return "\n".join(lines)


def save_schema(schema: dict[str, Any], output_dir: str = "data") -> Path:
    """Save schema to a JSON file.

    Args:
        schema: Schema dict.
        output_dir: Directory to save to.

    Returns:
        Path to saved file.
    """
    out_path = Path(output_dir) / "schema.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2, default=str)
    return out_path


def load_schema(input_dir: str = "data") -> dict[str, Any]:
    """Load schema from saved JSON file.

    Args:
        input_dir: Directory containing schema.json.

    Returns:
        Schema dict.
    """
    schema_path = Path(input_dir) / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found at {schema_path}. Upload data first.")
    with open(schema_path, "r") as f:
        return json.load(f)
