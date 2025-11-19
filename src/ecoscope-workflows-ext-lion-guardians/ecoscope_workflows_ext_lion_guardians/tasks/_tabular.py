import pandas as pd
from typing import Union, List, Dict
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame


@task
def add_totals_row(
    df: AnyDataFrame, label_col: Union[str, List[str], None] = None, label: str = "Total"
) -> AnyDataFrame:
    totals = df.select_dtypes(include="number").sum(numeric_only=True)

    # Create empty row
    total_row = pd.Series({col: None for col in df.columns})
    total_row[totals.index] = totals.values

    # Handle label columns
    if isinstance(label_col, str):
        if label_col in df.columns:
            total_row[label_col] = label
    elif isinstance(label_col, list):
        for col in label_col:
            if col in df.columns:
                total_row[col] = label

    # Combine and reset index
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)


@task
def rename_columns(df: AnyDataFrame, rename_cols: Dict[str, str]) -> AnyDataFrame:
    if not isinstance(rename_cols, dict):
        raise TypeError(f"rename_cols must be a dictionary, got {type(rename_cols).__name__}")

    if not rename_cols:
        raise ValueError("rename_cols dictionary cannot be empty")

    current_columns = set(df.columns)
    missing_cols = set(rename_cols.keys()) - current_columns
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    new_column_names = set(rename_cols.values())
    if len(new_column_names) != len(rename_cols):
        raise ValueError("Duplicate values found in new column names")

    # Perform the rename operation
    df_renamed = df.rename(rename_cols)

    return df_renamed


@task
def extract_date_parts(
    df: AnyDataFrame, date_column: str, parts: List[str] = ["month", "day", "month_name"]
) -> AnyDataFrame:
    """
    Extract date parts from a date column.

    Parameters:
    -----------
    df : DataFrame (pandas or polars)
    date_column : str - Name of date column
    parts : List[str] - Options: 'month', 'day', 'month_name', 'year', 'day_name', 'quarter', 'week'

    Returns: DataFrame with additional date part columns
    """
    # Basic validation
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found")

    df_result = df.copy()

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df_result[date_column]):
        df_result[date_column] = pd.to_datetime(df_result[date_column])

    # Extract parts
    part_map = {
        "month": lambda: df_result[date_column].dt.month,
        "day": lambda: df_result[date_column].dt.day,
        "month_name": lambda: df_result[date_column].dt.month_name(),
        "year": lambda: df_result[date_column].dt.year,
        "day_name": lambda: df_result[date_column].dt.day_name(),
        "quarter": lambda: df_result[date_column].dt.quarter,
        "week": lambda: df_result[date_column].dt.isocalendar().week,
    }

    for part in parts:
        if part in part_map:
            df_result[f"{part}"] = part_map[part]()

    return df_result
