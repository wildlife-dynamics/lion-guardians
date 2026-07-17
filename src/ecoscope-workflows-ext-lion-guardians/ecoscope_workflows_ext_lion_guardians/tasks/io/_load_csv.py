from pathlib import Path
from typing import Annotated
from wt_registry import register
from pydantic import Field, FilePath, AfterValidator
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme
import pandas as pd
from ecoscope.platform.annotations import (
    DataFrame,
    JsonSerializableDataFrameModel,
)
from ecoscope.platform.annotations import AdvancedField


def validate_tabular_file(path: Path) -> Path:
    valid_formats = [".csv", ".parquet"]
    if path.suffix.lower() not in valid_formats:
        raise ValueError(f"Invalid file format '{path.suffix}'. Allowed formats are: {', '.join(valid_formats)}")
    return path


@register()
def load_local_tabular_file(
    file_path: Annotated[
        FilePath,
        AfterValidator(validate_tabular_file),
        Field(description="Path to the tabular file (CSV or Parquet)."),
    ],
    encoding: Annotated[
        str,
        AdvancedField(
            default="latin-1",
            description=(
                "Text encoding used when reading CSV files. Defaults to 'latin-1', "
                "which handles non-UTF-8 characters common in field-collected data. "
                "Ignored for Parquet."
            ),
        ),
    ] = "latin-1",
) -> Annotated[DataFrame[JsonSerializableDataFrameModel], Field()]:
    """
    Load a local CSV or Parquet file into a pandas DataFrame.

    CSVs are read with encoding='latin-1' by default to avoid UnicodeDecodeError
    on files containing non-UTF-8 bytes (degree symbols, accents, smart quotes, etc.).
    Parquet files are read natively — encoding is not applicable.
    """
    normalized_path = remove_file_scheme(str(file_path))
    suffix = Path(normalized_path).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(normalized_path, encoding=encoding)
    elif suffix == ".parquet":
        return pd.read_parquet(normalized_path)
    else:
        # Redundant given the validator, but keeps the error surface clear
        raise ValueError(f"Unsupported file format: {suffix}")
