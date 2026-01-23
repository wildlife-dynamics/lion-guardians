import pandas as pd
from typing import List
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame


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
