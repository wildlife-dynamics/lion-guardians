"""Tests for ecoscope_workflows_ext_lion_guardians.tasks.io._patrols.

`filter_daytime_patrols` is decorated with `wt_registry.register()` (a no-op
at call time), so it is exercised here as a plain function against a
minimal DataFrame carrying the `groupby_col` / `fixtime` columns it relies
on -- there is no runtime schema enforcement of `PatrolObservationsGDF`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecoscope_workflows_ext_lion_guardians.tasks.io._patrols import (
    filter_daytime_patrols,
)


def _fix(patrol_id, day, hour, minute=0):
    return {
        "groupby_col": patrol_id,
        "fixtime": pd.Timestamp(f"2024-01-{day:02d} {hour:02d}:{minute:02d}:00", tz="UTC"),
    }


def _df(rows):
    return pd.DataFrame(rows)


class TestFilterDaytimePatrols:
    def test_patrol_fully_within_default_daytime_hours_is_kept(self):
        df = _df([_fix("p1", 1, 7), _fix("p1", 1, 18)])
        result = filter_daytime_patrols(df)
        assert set(result["groupby_col"]) == {"p1"}

    def test_patrol_starting_before_default_start_hour_is_dropped(self):
        df = _df([_fix("p1", 1, 5), _fix("p1", 1, 18)])
        result = filter_daytime_patrols(df)
        assert result.empty

    def test_patrol_ending_at_default_end_hour_is_dropped(self):
        # end_hour=19 is exclusive: a fix landing exactly at hour 19 fails.
        df = _df([_fix("p1", 1, 7), _fix("p1", 1, 19)])
        result = filter_daytime_patrols(df)
        assert result.empty

    def test_patrol_ending_just_before_default_end_hour_is_kept(self):
        df = _df([_fix("p1", 1, 7), _fix("p1", 1, 18, minute=59)])
        result = filter_daytime_patrols(df)
        assert set(result["groupby_col"]) == {"p1"}

    def test_start_hour_boundary_is_inclusive(self):
        df = _df([_fix("p1", 1, 6), _fix("p1", 1, 18)])
        result = filter_daytime_patrols(df)
        assert set(result["groupby_col"]) == {"p1"}

    def test_patrol_spanning_multiple_days_is_dropped_even_if_hours_ok(self):
        df = _df([_fix("p1", 1, 7), _fix("p1", 2, 18)])
        result = filter_daytime_patrols(df)
        assert result.empty

    def test_multiple_patrols_filtered_independently(self):
        df = _df(
            [
                _fix("keep", 1, 7),
                _fix("keep", 1, 18),
                _fix("drop_night", 1, 3),
                _fix("drop_night", 1, 18),
                _fix("drop_multiday", 1, 7),
                _fix("drop_multiday", 3, 12),
            ]
        )
        result = filter_daytime_patrols(df)
        assert set(result["groupby_col"]) == {"keep"}

    def test_custom_start_and_end_hour(self):
        df = _df([_fix("p1", 1, 8), _fix("p1", 1, 16)])
        # Default hours (6-19) would keep this; a narrower window excludes it.
        result = filter_daytime_patrols(df, start_hour=9, end_hour=17)
        assert result.empty

        result = filter_daytime_patrols(df, start_hour=8, end_hour=17)
        assert set(result["groupby_col"]) == {"p1"}

    def test_single_fix_patrol_start_equals_end(self):
        df = _df([_fix("p1", 1, 10)])
        result = filter_daytime_patrols(df)
        assert set(result["groupby_col"]) == {"p1"}

    def test_empty_dataframe_returns_empty_without_error(self):
        df = pd.DataFrame({"groupby_col": [], "fixtime": pd.to_datetime([]).tz_localize("UTC")})
        result = filter_daytime_patrols(df)
        assert result.empty

    def test_original_dataframe_not_mutated(self):
        df = _df(
            [
                _fix("keep", 1, 7),
                _fix("keep", 1, 18),
                _fix("drop", 1, 3),
                _fix("drop", 1, 4),
            ]
        )
        original_len = len(df)
        filter_daytime_patrols(df)
        assert len(df) == original_len

    def test_result_preserves_other_columns(self):
        rows = [_fix("p1", 1, 7), _fix("p1", 1, 18)]
        for r in rows:
            r["ranger_name"] = "Alice"
        df = _df(rows)
        result = filter_daytime_patrols(df)
        assert "ranger_name" in result.columns
        assert (result["ranger_name"] == "Alice").all()

    @pytest.mark.parametrize("hour", [0, 5, 19, 23])
    def test_patrols_entirely_outside_window_are_dropped(self, hour):
        df = _df([_fix("p1", 1, hour), _fix("p1", 1, hour)])
        result = filter_daytime_patrols(df)
        assert result.empty
