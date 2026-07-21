"""Tests for ecoscope_workflows_ext_lion_guardians.tasks.reporting._vehicles_context.

Unlike `_guardians_context`, `create_vehicles_context` always calls
`pd.read_csv` on `summary_table` (it never accepts an in-memory DataFrame),
so every non-None case here writes a real CSV to `tmp_path`.
"""

from __future__ import annotations

import pandas as pd
import pytest
from wt_task.skip import SKIP_SENTINEL

from ecoscope_workflows_ext_lion_guardians.tasks.reporting._vehicles_context import (
    create_vehicles_context,
)


def _write_summary(tmp_path, **overrides):
    base = {
        "subject_name": ["Truck 1"],
        "min_speed": [1.111],
        "mean_speed": [10.222],
        "max_speed": [40.333],
        "total_distance": [123.456],
    }
    base.update(overrides)
    path = tmp_path / "summary.csv"
    pd.DataFrame(base).to_csv(path, index=False)
    return str(path)


class TestCreateVehiclesContext:
    def test_return_dict_has_exactly_expected_keys(self):
        ctx = create_vehicles_context(summary_table=None, speedmap=None, tracks_map=None, line_chart=None)
        assert set(ctx.keys()) == {
            "vehicle_plate",
            "min_speed",
            "mean_speed",
            "max_speed",
            "total_distance",
            "vehicle_speedmap",
            "vehicle_tracks_map",
            "vehicle_speed_line_chart",
        }

    def test_summary_table_none_leaves_stats_none(self):
        ctx = create_vehicles_context(summary_table=None, speedmap=None, tracks_map=None, line_chart=None)
        assert ctx["vehicle_plate"] is None
        assert ctx["min_speed"] is None
        assert ctx["mean_speed"] is None
        assert ctx["max_speed"] is None
        assert ctx["total_distance"] is None

    def test_summary_table_skip_sentinel_treated_like_none(self):
        ctx = create_vehicles_context(summary_table=SKIP_SENTINEL, speedmap=None, tracks_map=None, line_chart=None)
        assert ctx["vehicle_plate"] is None

    def test_summary_table_extracts_values_from_last_row(self, tmp_path):
        path = _write_summary(
            tmp_path,
            subject_name=["Truck 1", "Truck 2"],
            min_speed=[1.0, 2.0],
            mean_speed=[10.0, 20.0],
            max_speed=[40.0, 50.0],
            total_distance=[100.0, 200.0],
        )
        ctx = create_vehicles_context(summary_table=path, speedmap=None, tracks_map=None, line_chart=None)
        assert ctx["vehicle_plate"] == "Truck 2"
        assert ctx["min_speed"] == 2.0
        assert ctx["mean_speed"] == 20.0
        assert ctx["max_speed"] == 50.0
        assert ctx["total_distance"] == 200.0

    def test_speed_values_rounded_to_two_decimals(self, tmp_path):
        path = _write_summary(tmp_path)
        ctx = create_vehicles_context(summary_table=path, speedmap=None, tracks_map=None, line_chart=None)
        assert ctx["min_speed"] == 1.11
        assert ctx["mean_speed"] == 10.22
        assert ctx["max_speed"] == 40.33
        assert ctx["total_distance"] == 123.46

    def test_missing_columns_leave_corresponding_fields_none(self, tmp_path):
        path = tmp_path / "summary.csv"
        pd.DataFrame({"subject_name": ["Truck 1"]}).to_csv(path, index=False)

        ctx = create_vehicles_context(summary_table=str(path), speedmap=None, tracks_map=None, line_chart=None)
        assert ctx["vehicle_plate"] == "Truck 1"
        assert ctx["min_speed"] is None
        assert ctx["mean_speed"] is None
        assert ctx["max_speed"] is None
        assert ctx["total_distance"] is None

    def test_empty_summary_table_leaves_all_stats_none(self, tmp_path):
        path = tmp_path / "summary.csv"
        pd.DataFrame(
            columns=[
                "subject_name",
                "min_speed",
                "mean_speed",
                "max_speed",
                "total_distance",
            ]
        ).to_csv(path, index=False)

        ctx = create_vehicles_context(summary_table=str(path), speedmap=None, tracks_map=None, line_chart=None)
        assert ctx["vehicle_plate"] is None
        assert ctx["min_speed"] is None

    def test_nonexistent_summary_table_does_not_raise(self, tmp_path, capsys):
        missing = str(tmp_path / "does_not_exist.csv")
        ctx = create_vehicles_context(summary_table=missing, speedmap=None, tracks_map=None, line_chart=None)
        assert ctx["vehicle_plate"] is None
        captured = capsys.readouterr()
        assert "Error reading summary table" in captured.out

    def test_map_and_chart_paths_pass_through(self, tmp_path):
        path = _write_summary(tmp_path)
        ctx = create_vehicles_context(
            summary_table=path,
            speedmap="/maps/speed.png",
            tracks_map="/maps/tracks.png",
            line_chart="/charts/line.png",
        )
        assert ctx["vehicle_speedmap"] == "/maps/speed.png"
        assert ctx["vehicle_tracks_map"] == "/maps/tracks.png"
        assert ctx["vehicle_speed_line_chart"] == "/charts/line.png"

    @pytest.mark.parametrize("field", ["speedmap", "tracks_map", "line_chart"])
    def test_skip_sentinel_map_and_chart_fields_become_none(self, field):
        kwargs = dict(summary_table=None, speedmap=None, tracks_map=None, line_chart=None)
        kwargs[field] = SKIP_SENTINEL
        ctx = create_vehicles_context(**kwargs)
        key_map = {
            "speedmap": "vehicle_speedmap",
            "tracks_map": "vehicle_tracks_map",
            "line_chart": "vehicle_speed_line_chart",
        }
        assert ctx[key_map[field]] is None
