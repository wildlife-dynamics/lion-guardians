import pandas as pd
from typing import Union, List
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
