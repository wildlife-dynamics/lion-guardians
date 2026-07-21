"""Tests for ecoscope_workflows_ext_lion_guardians.tasks.io._load_csv.

`load_local_tabular_file` is decorated with `wt_registry.register()`, which is
a no-op at call time, so it behaves as a plain function here -- no pydantic
validation of the `Annotated[FilePath, AfterValidator(...)]` parameter
actually runs unless something explicitly invokes it (nothing does), so the
function's own suffix check in the body is what actually guards against
unsupported formats.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecoscope_workflows_ext_lion_guardians.tasks.io._load_csv import (
    load_local_tabular_file,
    validate_tabular_file,
)


# --------------------------------------------------------------------------
# validate_tabular_file
# --------------------------------------------------------------------------
class TestValidateTabularFile:
    def test_csv_path_returned_unchanged(self, tmp_path):
        path = tmp_path / "data.csv"
        assert validate_tabular_file(path) == path

    def test_parquet_path_returned_unchanged(self, tmp_path):
        path = tmp_path / "data.parquet"
        assert validate_tabular_file(path) == path

    def test_uppercase_extension_accepted(self, tmp_path):
        path = tmp_path / "data.CSV"
        assert validate_tabular_file(path) == path

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "data.txt"
        with pytest.raises(ValueError, match="Invalid file format '.txt'"):
            validate_tabular_file(path)

    def test_error_message_lists_allowed_formats(self, tmp_path):
        with pytest.raises(ValueError, match=r"Allowed formats are: \.csv, \.parquet"):
            validate_tabular_file(tmp_path / "data.json")

    def test_no_extension_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid file format ''"):
            validate_tabular_file(tmp_path / "data")


# --------------------------------------------------------------------------
# load_local_tabular_file
# --------------------------------------------------------------------------
class TestLoadLocalTabularFile:
    def test_loads_csv_with_default_columns(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n3,4\n")

        result = load_local_tabular_file(file_path=path)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 3]

    def test_loads_parquet(self, tmp_path):
        path = tmp_path / "data.parquet"
        pd.DataFrame({"x": [1, 2, 3]}).to_parquet(path)

        result = load_local_tabular_file(file_path=path)

        assert isinstance(result, pd.DataFrame)
        assert result["x"].tolist() == [1, 2, 3]

    def test_default_latin1_encoding_reads_non_utf8_bytes(self, tmp_path):
        path = tmp_path / "data.csv"
        # degree symbol (0xb0) is invalid utf-8 but valid latin-1
        path.write_bytes(b"name,temp\nsite\xb01,30\n")

        result = load_local_tabular_file(file_path=path)

        assert result["name"].iloc[0] == "site\xb01"

    def test_custom_encoding_used_for_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("name\nCafé\n", encoding="utf-8")

        result = load_local_tabular_file(file_path=path, encoding="utf-8")

        assert result["name"].iloc[0] == "Café"

    def test_encoding_ignored_for_parquet(self, tmp_path):
        path = tmp_path / "data.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(path)

        # Should not raise even though "utf-8" is meaningless for parquet.
        result = load_local_tabular_file(file_path=path, encoding="utf-8")
        assert result["x"].tolist() == [1]

    def test_file_scheme_prefix_is_normalized(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a\n1\n")

        result = load_local_tabular_file(file_path="file://" + str(path))

        assert result["a"].tolist() == [1]

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("a,b\n1,2\n")

        with pytest.raises(ValueError, match="Unsupported file format: .txt"):
            load_local_tabular_file(file_path=path)

    def test_uppercase_parquet_extension_still_loads(self, tmp_path):
        path = tmp_path / "data.PARQUET"
        pd.DataFrame({"x": [1]}).to_parquet(path)

        result = load_local_tabular_file(file_path=path)
        assert result["x"].tolist() == [1]
