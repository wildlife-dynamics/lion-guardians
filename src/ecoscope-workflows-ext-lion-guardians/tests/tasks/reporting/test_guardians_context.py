"""Tests for ecoscope_workflows_ext_lion_guardians.tasks.reporting._guardians_context.

`patrol_subject_stats_csv`, `patrol_subject_events_csv`, `patrol_subject_pivot_csv`,
`monthly_csv`, and `events_recorded_csv` all flow into unconditional DataFrame
operations (merge, drop, fillna) with no None-guard, so every test supplies
real DataFrames for those five parameters. Only the five map/chart path
parameters exercise the `_unwrap_skip` normalization, since those alone are
allowed to be None/SkipSentinel.
"""

from __future__ import annotations

import pandas as pd
import pytest
from wt_task.skip import SKIP_SENTINEL

from ecoscope_workflows_ext_lion_guardians.tasks.reporting._guardians_context import (
    create_guardians_context,
)


def _stats_df(**overrides):
    base = {
        "patrol_subject": ["Alice"],
        "no_of_patrols": [3],
        "total_distance": [12.345],
        "total_time": [4.567],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _events_df(**overrides):
    base = {"patrol_subject": ["Alice"], "no_of_events": [5]}
    base.update(overrides)
    return pd.DataFrame(base)


def _pivot_df(**overrides):
    base = {
        "Unnamed: 0": [0],
        "patrol_subject": ["Alice"],
        "arrest": [2],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _minimal_kwargs(**overrides):
    kwargs = dict(
        events_map=None,
        patrols_trajectories_map=None,
        time_density_map=None,
        pie_chart=None,
        time_series_bar_chart=None,
        monthly_csv=pd.DataFrame({"month": ["Jan"], "total": [1]}),
        patrol_subject_pivot_csv=_pivot_df(),
        patrol_subject_events_csv=_events_df(),
        patrol_subject_stats_csv=_stats_df(),
        events_recorded_csv=pd.DataFrame({"event_type": ["Arrest"], "count": [2]}),
    )
    kwargs.update(overrides)
    return kwargs


class TestCreateGuardiansContext:
    def test_return_dict_has_exactly_expected_keys(self):
        ctx = create_guardians_context(**_minimal_kwargs())
        assert set(ctx.keys()) == {
            "patrol_events_track_map",
            "patrol_trajectories_map",
            "patrol_time_density_map",
            "events_pie_chart",
            "events_time_series_bar_chart",
            "month_stats",
            "event_efforts",
            "guardian_stats",
            "patrol_events",
            "total_patrols",
            "total_time",
            "total_distance",
            "total_events_recorded",
        }

    def test_month_stats_matches_monthly_csv_records(self):
        monthly = pd.DataFrame({"month": ["Jan", "Feb"], "total": [1, 2]})
        ctx = create_guardians_context(**_minimal_kwargs(monthly_csv=monthly))
        assert ctx["month_stats"] == monthly.to_dict(orient="records")

    def test_event_efforts_matches_events_recorded_csv_records(self):
        events = pd.DataFrame({"event_type": ["Snare"], "count": [9]})
        ctx = create_guardians_context(**_minimal_kwargs(events_recorded_csv=events))
        assert ctx["event_efforts"] == events.to_dict(orient="records")

    def test_guardian_stats_merges_stats_and_events_on_patrol_subject(self):
        stats = _stats_df(
            patrol_subject=["Alice", "Bob"],
            no_of_patrols=[3, 2],
            total_distance=[10.0, 5.0],
            total_time=[1.0, 2.0],
        )
        events = _events_df(patrol_subject=["Alice", "Bob"], no_of_events=[5, 1])
        ctx = create_guardians_context(
            **_minimal_kwargs(patrol_subject_stats_csv=stats, patrol_subject_events_csv=events)
        )
        assert ctx["guardian_stats"] == [
            {
                "patrol_subject": "Alice",
                "no_of_patrols": 3,
                "total_distance": 10.0,
                "total_time": 1.0,
                "no_of_events": 5,
            },
            {
                "patrol_subject": "Bob",
                "no_of_patrols": 2,
                "total_distance": 5.0,
                "total_time": 2.0,
                "no_of_events": 1,
            },
        ]

    def test_guardian_stats_drops_patrol_subject_only_in_stats(self):
        # pd.merge defaults to an inner join, so subjects without a matching
        # events row are silently dropped from guardian_stats.
        stats = _stats_df(
            patrol_subject=["Alice", "Solo"],
            no_of_patrols=[3, 9],
            total_distance=[10.0, 20.0],
            total_time=[1.0, 2.0],
        )
        events = _events_df(patrol_subject=["Alice"], no_of_events=[5])
        ctx = create_guardians_context(
            **_minimal_kwargs(patrol_subject_stats_csv=stats, patrol_subject_events_csv=events)
        )
        assert [row["patrol_subject"] for row in ctx["guardian_stats"]] == ["Alice"]

    def test_no_of_patrols_cast_to_int(self):
        stats = _stats_df(no_of_patrols=[3.0])
        ctx = create_guardians_context(**_minimal_kwargs(patrol_subject_stats_csv=stats))
        assert ctx["guardian_stats"][0]["no_of_patrols"] == 3
        assert isinstance(ctx["guardian_stats"][0]["no_of_patrols"], int)

    def test_totals_summed_and_rounded_to_two_decimals(self):
        stats = _stats_df(
            patrol_subject=["Alice", "Bob"],
            no_of_patrols=[3, 2],
            total_distance=[10.111, 5.222],
            total_time=[1.111, 2.222],
        )
        events = _events_df(patrol_subject=["Alice", "Bob"], no_of_events=[5, 1])
        ctx = create_guardians_context(
            **_minimal_kwargs(patrol_subject_stats_csv=stats, patrol_subject_events_csv=events)
        )
        assert ctx["total_patrols"] == 5
        assert ctx["total_events_recorded"] == 6
        assert ctx["total_distance"] == pytest.approx(15.33)
        assert ctx["total_time"] == pytest.approx(3.33)

    def test_patrol_events_drops_unnamed_index_column(self):
        pivot = _pivot_df(**{"Unnamed: 0": [0], "patrol_subject": ["Alice"], "arrest": [2]})
        ctx = create_guardians_context(**_minimal_kwargs(patrol_subject_pivot_csv=pivot))
        row = next(iter(ctx["patrol_events"].values()))
        assert "Unnamed: 0" not in row

    def test_patrol_events_missing_unnamed_column_raises_key_error(self):
        # Documents existing behaviour: drop(columns=["Unnamed: 0"]) has no
        # errors="ignore", so a pivot csv without that column raises.
        pivot = pd.DataFrame({"patrol_subject": ["Alice"], "arrest": [2]})
        with pytest.raises(KeyError):
            create_guardians_context(**_minimal_kwargs(patrol_subject_pivot_csv=pivot))

    def test_patrol_events_nan_filled_with_zero_and_cast_to_int(self):
        pivot = _pivot_df(
            **{
                "Unnamed: 0": [0, 1],
                "patrol_subject": ["Alice", "Bob"],
                "arrest": [2.0, None],
            }
        )
        ctx = create_guardians_context(**_minimal_kwargs(patrol_subject_pivot_csv=pivot))
        rows = list(ctx["patrol_events"].values())
        assert rows[1]["arrest"] == 0
        assert isinstance(rows[1]["arrest"], int)

    def test_patrol_events_patrol_subject_column_not_cast_to_int(self):
        pivot = _pivot_df(**{"Unnamed: 0": [0], "patrol_subject": ["Alice"], "arrest": [2]})
        ctx = create_guardians_context(**_minimal_kwargs(patrol_subject_pivot_csv=pivot))
        row = next(iter(ctx["patrol_events"].values()))
        assert row["patrol_subject"] == "Alice"

    @pytest.mark.parametrize(
        "field",
        [
            "events_map",
            "patrols_trajectories_map",
            "time_density_map",
            "pie_chart",
            "time_series_bar_chart",
        ],
    )
    def test_skip_sentinel_map_and_chart_fields_become_none(self, field):
        ctx = create_guardians_context(**_minimal_kwargs(**{field: SKIP_SENTINEL}))
        key_map = {
            "events_map": "patrol_events_track_map",
            "patrols_trajectories_map": "patrol_trajectories_map",
            "time_density_map": "patrol_time_density_map",
            "pie_chart": "events_pie_chart",
            "time_series_bar_chart": "events_time_series_bar_chart",
        }
        assert ctx[key_map[field]] is None

    def test_plain_string_map_paths_pass_through(self):
        ctx = create_guardians_context(**_minimal_kwargs(events_map="/maps/events.png"))
        assert ctx["patrol_events_track_map"] == "/maps/events.png"

    def test_csv_path_string_is_read_as_dataframe(self, tmp_path):
        monthly_path = tmp_path / "monthly.csv"
        pd.DataFrame({"month": ["Mar"], "total": [7]}).to_csv(monthly_path, index=False)

        ctx = create_guardians_context(**_minimal_kwargs(monthly_csv=str(monthly_path)))

        assert ctx["month_stats"] == [{"month": "Mar", "total": 7}]
