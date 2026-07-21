"""Tests for ecoscope_workflows_ext_lion_guardians.tasks.reporting._collared_lions_context."""

from __future__ import annotations

import pandas as pd
from wt_task.skip import SKIP_SENTINEL

from ecoscope_workflows_ext_lion_guardians.tasks.reporting._collared_lions_context import (
    create_collared_lions_context,
)


class TestCreateCollaredLionsContext:
    def test_df_none_uses_all_grouper(self):
        ctx = create_collared_lions_context(df=None, total_distance=None, home_range=None, speed_map=None)
        assert ctx["grouper_value"] == "All"

    def test_single_subject_name_becomes_grouper_value(self):
        df = pd.DataFrame({"subject_name": ["Simba", "Simba"]})
        ctx = create_collared_lions_context(df=df, total_distance=None, home_range=None, speed_map=None)
        assert ctx["grouper_value"] == "Simba"

    def test_multiple_subject_names_falls_back_to_all(self):
        df = pd.DataFrame({"subject_name": ["Simba", "Nala"]})
        ctx = create_collared_lions_context(df=df, total_distance=None, home_range=None, speed_map=None)
        assert ctx["grouper_value"] == "All"

    def test_missing_subject_name_column_falls_back_to_all(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        ctx = create_collared_lions_context(df=df, total_distance=None, home_range=None, speed_map=None)
        assert ctx["grouper_value"] == "All"

    def test_nan_values_dropped_before_resolving_unique_name(self):
        df = pd.DataFrame({"subject_name": ["Simba", None, "Simba"]})
        ctx = create_collared_lions_context(df=df, total_distance=None, home_range=None, speed_map=None)
        assert ctx["grouper_value"] == "Simba"

    def test_skip_sentinel_df_treated_like_none(self):
        ctx = create_collared_lions_context(df=SKIP_SENTINEL, total_distance=None, home_range=None, speed_map=None)
        assert ctx["grouper_value"] == "All"

    def test_skip_sentinel_scalars_become_none(self):
        ctx = create_collared_lions_context(
            df=None,
            total_distance=SKIP_SENTINEL,
            home_range=SKIP_SENTINEL,
            speed_map=SKIP_SENTINEL,
        )
        assert ctx["total_distance"] is None
        assert ctx["home_range_map"] is None
        assert ctx["speed_map"] is None

    def test_plain_scalars_pass_through(self):
        ctx = create_collared_lions_context(
            df=None,
            total_distance=42.5,
            home_range="/maps/home_range.png",
            speed_map="/maps/speed.png",
        )
        assert ctx["total_distance"] == 42.5
        assert ctx["home_range_map"] == "/maps/home_range.png"
        assert ctx["speed_map"] == "/maps/speed.png"

    def test_int_total_distance_preserved(self):
        ctx = create_collared_lions_context(df=None, total_distance=100, home_range=None, speed_map=None)
        assert ctx["total_distance"] == 100

    def test_list_with_sentinel_and_one_value_collapses_to_scalar(self):
        ctx = create_collared_lions_context(
            df=None,
            total_distance=[SKIP_SENTINEL, 7.0],
            home_range=None,
            speed_map=None,
        )
        assert ctx["total_distance"] == 7.0

    def test_return_dict_has_exactly_expected_keys(self):
        ctx = create_collared_lions_context(df=None, total_distance=None, home_range=None, speed_map=None)
        assert set(ctx.keys()) == {
            "grouper_value",
            "total_distance",
            "home_range_map",
            "speed_map",
        }
