import os
import uuid
import pandas as pd
from pathlib import Path
from docxtpl import DocxTemplate
from wt_registry import register
from wt_task.skip import SkipSentinel
from typing import Optional, Dict, Any
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme
from ecoscope_workflows_ext_ste.tasks.reporting._mapbook_context import (
    DEFAULT_IMAGE_EXTENSIONS,
    build_docx_context,
    _unwrap_skip,
)


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


@register()
def render_guardians_page(
    template_path: str,
    output_dir: str,
    context: Dict[str, Any],
    filename: Optional[str] = None,
    strict_images: bool = False,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
) -> str:
    """Render the guardians report from a docx template and return its path.

    strict_images=False (default): missing/unreadable images are logged and
    rendered as blank slots. strict_images=True: they raise instead.
    """
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_dir is empty after normalization")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if context is None:
        raise ValueError("context cannot be None — pass an empty dict {} if no images are needed.")

    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = f"{uuid.uuid4().hex[:8]}.docx"
    output_path = Path(output_dir) / filename

    # --- image validation: warn by default, raise only in strict mode ---
    for field_name, value in context.items():
        if not isinstance(value, str) or not value:
            continue
        normalized = remove_file_scheme(value)
        if Path(normalized).suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
            if not os.path.exists(normalized):
                msg = f"{field_name}: image not found: {normalized}"
                if strict_images:
                    raise FileNotFoundError(msg)
                print(msg)

    try:
        tpl = DocxTemplate(template_path)
    except Exception as e:
        raise ValueError(f"Failed to load template {template_path}: {e}") from e

    rendered_context = build_docx_context(
        context=context,
        template=tpl,
        box_h_cm=box_h_cm,
        box_w_cm=box_w_cm,
    )

    try:
        tpl.render(rendered_context)
        tpl.save(output_path)
    except Exception as e:
        raise ValueError(f"Failed to render or save {output_path}: {e}") from e

    print(f"Rendered guardians report: {output_path}")
    return str(output_path)
