import pandas as pd
from wt_registry import register
from wt_task.skip import SkipSentinel
from typing import Optional, Dict, Any
from ecoscope_workflows_ext_ste.tasks.reporting._mapbook_context import _unwrap_skip


@register()
def create_guardians_context(
    events_map: str | SkipSentinel | None,
    patrols_trajectories_map: str | SkipSentinel | None,
    time_density_map: str | SkipSentinel | None,
    pie_chart: str | SkipSentinel | None,
    time_series_bar_chart: str | SkipSentinel | None,
    monthly_csv: str | SkipSentinel | None,
    patrol_subject_pivot_csv: str | SkipSentinel | None,  # patrol subject pivotted events
    patrol_subject_events_csv: str | SkipSentinel | None,  # patrol subject with events
    patrol_subject_stats_csv: str | SkipSentinel | None,
    events_recorded_csv: str | SkipSentinel | None,
) -> Dict[str, Optional[Any]]:
    def ensure_dataframe(value):
        if value is None:
            return None
        if isinstance(value, str):
            return pd.read_csv(value)
        return value

    events_map = _unwrap_skip(events_map)
    patrols_trajectories_map = _unwrap_skip(patrols_trajectories_map)
    time_density_map = _unwrap_skip(time_density_map)

    pie_chart = _unwrap_skip(pie_chart)
    time_series_bar_chart = _unwrap_skip(time_series_bar_chart)

    monthly_csv = _unwrap_skip(monthly_csv)
    patrol_subject_pivot_csv = _unwrap_skip(patrol_subject_pivot_csv)
    patrol_subject_events_csv = _unwrap_skip(patrol_subject_events_csv)
    patrol_subject_stats_csv = _unwrap_skip(patrol_subject_stats_csv)
    events_recorded_csv = _unwrap_skip(events_recorded_csv)

    monthly_csv = ensure_dataframe(monthly_csv)
    patrol_subject_pivot_csv = ensure_dataframe(patrol_subject_pivot_csv)
    patrol_subject_events_csv = ensure_dataframe(patrol_subject_events_csv)
    patrol_subject_stats_csv = ensure_dataframe(patrol_subject_stats_csv)
    events_recorded_csv = ensure_dataframe(events_recorded_csv)

    merged = patrol_subject_stats_csv.merge(
        patrol_subject_events_csv, left_on="patrol_subject", right_on="patrol_subject"
    )
    merged = merged[["patrol_subject", "no_of_patrols", "total_distance", "total_time", "no_of_events"]]
    merged["no_of_patrols"] = merged["no_of_patrols"].astype(int)

    print(f"inspecting patrol subject pivot CSV: {patrol_subject_pivot_csv.columns}")
    print(f" checking out the data : {patrol_subject_pivot_csv.head()}")

    patrol_subject_pivot_csv = patrol_subject_pivot_csv.drop(columns=["Unnamed: 0"])
    patrol_subject_pivot_csv = patrol_subject_pivot_csv.fillna(0)
    numeric_cols = patrol_subject_pivot_csv.columns.difference(["patrol_subject"])
    patrol_subject_pivot_csv[numeric_cols] = patrol_subject_pivot_csv[numeric_cols].fillna(0).astype(int)

    context = {
        "patrol_events_track_map": events_map,
        "patrol_trajectories_map": patrols_trajectories_map,
        "patrol_time_density_map": time_density_map,
        "events_pie_chart": pie_chart,
        "events_time_series_bar_chart": time_series_bar_chart,
        "month_stats": monthly_csv.to_dict(orient="records"),
        "event_efforts": events_recorded_csv.to_dict(orient="records"),
        "guardian_stats": merged.to_dict(orient="records"),
        "patrol_events": patrol_subject_pivot_csv.to_dict(orient="index"),
        "total_patrols": merged["no_of_patrols"].sum().astype(int),
        "total_time": round(merged["total_time"].sum().astype(float), 2),
        "total_distance": round(merged["total_distance"].sum().astype(float), 2),
        "total_events_recorded": merged["no_of_events"].sum().astype(int),
    }
    print(f"Patrols context :{context}")
    return context
