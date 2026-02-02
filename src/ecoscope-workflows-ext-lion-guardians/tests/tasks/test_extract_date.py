import pytest
import pandas as pd
import geopandas as gpd
from ecoscope_workflows_ext_lion_guardians.tasks._tabular import extract_date_parts

# --- Fixtures ---


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "date": pd.to_datetime(
                [
                    "2024-01-15",
                    "2024-03-22",
                    "2024-06-01",
                    "2024-12-31",
                ]
            ),
            "value": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def string_date_df():
    """Date column as strings instead of datetime."""
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "date": ["2024-01-15", "2024-06-22", "2024-12-31"],
        }
    )


@pytest.fixture
def geodataframe_sample():
    """GeoDataFrame to verify AnyDataFrame compatibility."""
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {
            "event_id": [1, 2],
            "date": pd.to_datetime(["2024-03-10", "2024-09-05"]),
            "geometry": [Point(35.0, -1.0), Point(36.0, -2.0)],
        }
    )


# --- Basic extraction tests ---


class TestBasicExtraction:
    def test_default_parts(self, sample_df):
        """Default parts should extract month, day, and month_name."""
        result = extract_date_parts(sample_df, "date")
        assert "month" in result.columns
        assert "day" in result.columns
        assert "month_name" in result.columns

    def test_month_values(self, sample_df):
        result = extract_date_parts(sample_df, "date", parts=["month"])
        assert result["month"].tolist() == [1, 3, 6, 12]

    def test_day_values(self, sample_df):
        result = extract_date_parts(sample_df, "date", parts=["day"])
        assert result["day"].tolist() == [15, 22, 1, 31]

    def test_year_values(self, sample_df):
        result = extract_date_parts(sample_df, "date", parts=["year"])
        assert result["year"].tolist() == [2024, 2024, 2024, 2024]

    def test_month_name_values(self, sample_df):
        result = extract_date_parts(sample_df, "date", parts=["month_name"])
        assert result["month_name"].tolist() == ["January", "March", "June", "December"]

    def test_day_name_values(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),  # Mon, Tue, Wed
            }
        )
        result = extract_date_parts(df, "date", parts=["day_name"])
        assert result["day_name"].tolist() == ["Monday", "Tuesday", "Wednesday"]

    def test_quarter_values(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]),
            }
        )
        result = extract_date_parts(df, "date", parts=["quarter"])
        assert result["quarter"].tolist() == [1, 2, 3, 4]

    def test_week_values(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),  # Week 1 and Week 2
            }
        )
        result = extract_date_parts(df, "date", parts=["week"])
        assert result["week"].tolist() == [1, 2]

    def test_multiple_parts_at_once(self, sample_df):
        parts = ["month", "day", "year", "quarter"]
        result = extract_date_parts(sample_df, "date", parts=parts)
        for part in parts:
            assert part in result.columns


# --- Input type handling ---


class TestInputTypes:
    def test_string_date_column_is_converted(self, string_date_df):
        """Should auto-convert string dates to datetime before extracting."""
        result = extract_date_parts(string_date_df, "date", parts=["month"])
        assert result["month"].tolist() == [1, 6, 12]
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_geodataframe_returns_geodataframe(self, geodataframe_sample):
        """Should preserve GeoDataFrame type through extraction."""
        result = extract_date_parts(geodataframe_sample, "date", parts=["month"])
        assert isinstance(result, gpd.GeoDataFrame)
        assert "month" in result.columns
        assert result["month"].tolist() == [3, 9]


# --- Original DataFrame integrity ---


class TestDataIntegrity:
    def test_original_df_is_not_mutated(self, sample_df):
        original_cols = sample_df.columns.tolist()
        extract_date_parts(sample_df, "date", parts=["month", "year"])
        assert sample_df.columns.tolist() == original_cols

    def test_existing_columns_are_preserved(self, sample_df):
        result = extract_date_parts(sample_df, "date", parts=["month"])
        assert "event_id" in result.columns
        assert "value" in result.columns
        assert result["event_id"].tolist() == [1, 2, 3, 4]

    def test_row_count_unchanged(self, sample_df):
        result = extract_date_parts(sample_df, "date", parts=["month", "day", "year"])
        assert len(result) == len(sample_df)


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_parts_list(self, sample_df):
        """Passing an empty parts list should return df with no new columns."""
        result = extract_date_parts(sample_df, "date", parts=[])
        assert result.columns.tolist() == sample_df.columns.tolist()

    def test_invalid_part_is_silently_skipped(self, sample_df):
        """Unrecognised part names should be ignored, not raise."""
        result = extract_date_parts(sample_df, "date", parts=["month", "nonexistent_part"])
        assert "month" in result.columns
        assert "nonexistent_part" not in result.columns

    def test_single_row_df(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-06-15"])})
        result = extract_date_parts(df, "date", parts=["month", "day", "year"])
        assert result["month"].tolist() == [6]
        assert result["day"].tolist() == [15]
        assert result["year"].tolist() == [2024]

    def test_duplicate_parts_in_list(self, sample_df):
        """Duplicates should not cause errors or duplicate columns."""
        result = extract_date_parts(sample_df, "date", parts=["month", "month", "day"])
        assert "month" in result.columns
        assert "day" in result.columns

    def test_leap_year_feb_29(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-02-29"])})
        result = extract_date_parts(df, "date", parts=["month", "day"])
        assert result["month"].tolist() == [2]
        assert result["day"].tolist() == [29]

    def test_year_boundary(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-12-31", "2024-01-01"]),
            }
        )
        result = extract_date_parts(df, "date", parts=["year", "month", "day"])
        assert result["year"].tolist() == [2023, 2024]
        assert result["month"].tolist() == [12, 1]
        assert result["day"].tolist() == [31, 1]


# --- Error handling ---


class TestErrorHandling:
    def test_missing_date_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            extract_date_parts(sample_df, "nonexistent")

    def test_empty_dataframe_raises_or_returns_empty(self):
        """Empty df with correct column should return empty df with part columns."""
        df = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})
        result = extract_date_parts(df, "date", parts=["month", "day"])
        assert len(result) == 0
        assert "month" in result.columns
        assert "day" in result.columns
