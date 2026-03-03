from typing import Optional, Dict, Any
import os
import pandas as pd
from pathlib import Path
from docx.shared import Inches
from docxtpl import DocxTemplate, InlineImage
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame
from ecoscope_workflows_core.skip import SkipSentinel, SKIP_SENTINEL
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme


def _format_temporal_grouper(grouper: Any, df: AnyDataFrame) -> str:
    """
    Format temporal grouper with human-readable labels.
    Extracts actual temporal values from the segment_start column.
    """
    # Try to get the temporal_index attribute
    temporal_index = getattr(grouper, "temporal_index", None)

    if temporal_index and df is not None:
        # Get the directive (e.g., '%B' for full month name, '%A' for day name)
        directive = getattr(temporal_index, "directive", None)

        # Check if segment_start column exists
        if "segment_start" in df.columns:
            try:
                # Get unique dates from segment_start
                dates = df["segment_start"].dropna()

                if len(dates) > 0:
                    # Format based on directive
                    if directive == "%B":  # Full month name
                        unique_months = dates.dt.strftime("%B").unique()
                        if len(unique_months) == 1:
                            return unique_months[0]
                        elif len(unique_months) > 1:
                            return f"{unique_months[0]} - {unique_months[-1]}"

                    elif directive == "%b":  # Abbreviated month name
                        unique_months = dates.dt.strftime("%b").unique()
                        if len(unique_months) == 1:
                            return unique_months[0]
                        elif len(unique_months) > 1:
                            return f"{unique_months[0]} - {unique_months[-1]}"

                    elif directive == "%A":  # Full day name
                        unique_days = dates.dt.strftime("%A").unique()
                        if len(unique_days) == 1:
                            return unique_days[0]
                        elif len(unique_days) > 1:
                            return f"{unique_days[0]} - {unique_days[-1]}"

                    elif directive == "%a":  # Abbreviated day name
                        unique_days = dates.dt.strftime("%a").unique()
                        if len(unique_days) == 1:
                            return unique_days[0]
                        elif len(unique_days) > 1:
                            return f"{unique_days[0]} - {unique_days[-1]}"

                    elif directive == "%Y":  # Year
                        unique_years = dates.dt.strftime("%Y").unique()
                        if len(unique_years) == 1:
                            return unique_years[0]
                        elif len(unique_years) > 1:
                            return f"{unique_years[0]} - {unique_years[-1]}"

                    elif directive == "%d":  # Day of month
                        unique_days = dates.dt.strftime("%d").unique()
                        if len(unique_days) == 1:
                            return f"Day {unique_days[0]}"
                        elif len(unique_days) > 1:
                            return f"Day {unique_days[0]} - {unique_days[-1]}"

                    elif directive == "%j":  # Day of year
                        unique_days = dates.dt.strftime("%j").unique()
                        if len(unique_days) == 1:
                            return f"Day {int(unique_days[0])}"
                        elif len(unique_days) > 1:
                            return f"Day {int(unique_days[0])} - {int(unique_days[-1])}"

                    elif directive == "%U" or directive == "%W":  # Week number
                        unique_weeks = dates.dt.strftime("%U").unique()
                        if len(unique_weeks) == 1:
                            return f"Week {int(unique_weeks[0])}"
                        elif len(unique_weeks) > 1:
                            return f"Week {int(unique_weeks[0])} - {int(unique_weeks[-1])}"

                    else:
                        # Default: try to format with the directive
                        formatted = dates.dt.strftime(directive).unique()
                        if len(formatted) == 1:
                            return formatted[0]
                        elif len(formatted) > 1:
                            return f"{formatted[0]} - {formatted[-1]}"

            except Exception as e:
                print(f"Error extracting temporal value from segment_start: {e}")

        # Fallback: Use directive mapping
        directive_mapping = {
            "%B": "Monthly",
            "%b": "Monthly",
            "%A": "Day of Week",
            "%a": "Day of Week",
            "%Y": "Yearly",
            "%d": "Daily",
            "%j": "Day of Year",
            "%U": "Weekly",
            "%W": "Weekly",
        }

        if directive in directive_mapping:
            return directive_mapping[directive]

    # Final fallback
    return "Temporal"


@task
def guardians_ctx(
    grouper_name: tuple | list | str | None,
    events_map: str | SkipSentinel | None,
    patrols_trajectories_map: str | SkipSentinel | None,
    time_density_map: str | SkipSentinel | None,
    pie_chart: str | SkipSentinel | None,
    time_series_bar_chart: str | SkipSentinel | None,
    monthly_csv: str | SkipSentinel | None,
    patrol_subject_pivot_csv: str | SkipSentinel | None,  # patrol subject pivotted events
    patrol_subject_events_csv: str | SkipSentinel | None,  # patrol subject with events
    patrol_subject_stats_csv: str | SkipSentinel | None,
    events_recorded_csv: str | SkipSentinel | None,  # events
    df: Optional[AnyDataFrame] = None,
) -> Dict[str, Optional[Any]]:
    def unwrap_skip(value):
        """
        Recursively unwrap SkipSentinel values from nested structures.
        CRITICAL: Converts SkipSentinel to None to preserve argument positions.
        """
        if value is None or value is SKIP_SENTINEL:
            return None

        if isinstance(value, (list, tuple)):
            # Convert each item, converting SkipSentinel to None
            unwrapped_items = [unwrap_skip(v) for v in value]

            # Filter out None values only for finding valid items to return
            non_none_items = [item for item in unwrapped_items if item is not None]

            # If no valid items found, return None
            if not non_none_items:
                return None

            # If only one valid item, return it directly (flatten single-item containers)
            if len(non_none_items) == 1:
                return non_none_items[0]

            # If multiple valid items, return the container with all items (including Nones)
            # This preserves the structure but only matters for complex nested cases
            return type(value)(unwrapped_items)

        return value

    # Extract grouper value dynamically
    grouper_value = "All"
    print(f"grouper name raw: {grouper_name} (type: {type(grouper_name)})")

    if grouper_name:
        if isinstance(grouper_name, str):
            grouper_value = grouper_name
        elif isinstance(grouper_name, (list, tuple)) and len(grouper_name) > 0:
            # Check if it's a tuple structure like (('index_name', 'All'),)
            first_item = grouper_name[0]

            # Handle tuple structure (('index_name', 'All'),)
            if isinstance(first_item, tuple) and len(first_item) == 2:
                key, value = first_item
                if key == "index_name" and value == "All":
                    grouper_value = "All"
                else:
                    grouper_value = str(value)
                print(f"Extracted from tuple structure: {grouper_value}")
            # Handle grouper objects
            elif hasattr(first_item, "__class__"):
                grouper = first_item
                grouper_type = grouper.__class__.__name__
                print(f"grouper_type: {grouper_type}")

                if grouper_type == "ValueGrouper":
                    index_name = getattr(grouper, "index_name", None)
                    if df is not None and index_name and index_name in df.columns:
                        unique_values = df[index_name].unique()
                        if len(unique_values) == 1:
                            grouper_value = str(unique_values[0])
                        else:
                            grouper_value = index_name
                    else:
                        grouper_value = index_name if index_name else "Value"

                elif grouper_type == "TemporalGrouper":
                    grouper_value = _format_temporal_grouper(grouper, df)

                elif grouper_type == "AllGrouper":
                    grouper_value = "All"
                else:
                    grouper_value = str(grouper_name)
            else:
                grouper_value = str(first_item)
        else:
            grouper_value = str(grouper_name)

    print(f"grouper_value: {grouper_value}")

    def ensure_dataframe(value):
        if value is None:
            return None
        if isinstance(value, str):
            return pd.read_csv(value)
        return value

    events_map = unwrap_skip(events_map)
    patrols_trajectories_map = unwrap_skip(patrols_trajectories_map)
    time_density_map = unwrap_skip(time_density_map)

    pie_chart = unwrap_skip(pie_chart)
    time_series_bar_chart = unwrap_skip(time_series_bar_chart)

    monthly_csv = unwrap_skip(monthly_csv)
    patrol_subject_pivot_csv = unwrap_skip(patrol_subject_pivot_csv)
    patrol_subject_events_csv = unwrap_skip(patrol_subject_events_csv)
    patrol_subject_stats_csv = unwrap_skip(patrol_subject_stats_csv)
    events_recorded_csv = unwrap_skip(events_recorded_csv)

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
        "grouper_value": grouper_value,
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


@task
def generate_guardians_report(
    template_path: str,
    output_dir: str,
    context: Dict[str, Any],
    filename: Optional[str] = None,
    box_w_inches: float = 6.0,
    box_h_inches: float = 4.0,
    validate_images: bool = True,
) -> str:
    """
    Renders a .docx report from a DocxTemplate template.

    Args:
        template_path:   Path to the .docx template file.
        output_dir:      Directory where the report will be saved.
        context:         Mixed-type dict — image paths (str), tables (list/dict), scalars, etc.
        filename:        Output filename. Auto-generated if None.
        box_w_inches:    Image width in inches.
        box_h_inches:    Image height in inches.
        validate_images: Warn when an expected image file is missing.

    Returns:
        Absolute path to the generated .docx file.
    """
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    print(f"\nTemplate Path: {template_path}")
    print(f"Output Directory: {output_dir}")

    # --- Validate inputs ---
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"Output directory not found: {output_dir}")
    if context is None:
        raise ValueError("context cannot be None — pass an empty dict {} if no images are needed.")

    tpl = DocxTemplate(template_path)

    # --- Derive image directory from the first image path found in context ---
    # Images may live in a different directory than output_dir (e.g. a workflow outputs folder)
    image_dir = output_dir  # fallback to output_dir if no image paths found in context
    for value in context.values():
        if isinstance(value, str) and Path(value).suffix.lower() in IMAGE_EXTS:
            image_dir = str(Path(value).parent)
            break

    # --- Scan image_dir for available images, keyed by stem ---
    images_found: Dict[str, str] = {}
    if os.path.isdir(image_dir):
        for entry in os.scandir(image_dir):
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTS:
                images_found[Path(entry.name).stem] = entry.path
    else:
        print(f"Warning: Image directory not found: {image_dir}")

    print(f"Images found: {list(images_found.keys())}")

    print(f"\nImage dir scanned: {image_dir}")
    print(f"Images found: {images_found}")

    # --- Build render context without mutating the caller's dict ---
    render_context: Dict[str, Any] = {}
    for template_var, value in context.items():
        # Only attempt image resolution for string values with image extensions
        if isinstance(value, str) and Path(value).suffix.lower() in IMAGE_EXTS:
            file_stem = Path(value).stem
            print(f"\n[DEBUG] template_var: {template_var}")
            print(f"[DEBUG] value (from context): {value}")
            print(f"[DEBUG] file_stem extracted: {file_stem}")
            print(f"[DEBUG] stem in images_found: {file_stem in images_found}")
            if file_stem in images_found:
                print(f"[DEBUG] resolved path: {images_found[file_stem]}")
                print(f"[DEBUG] file exists: {os.path.isfile(images_found[file_stem])}")
                try:
                    render_context[template_var] = InlineImage(
                        tpl,
                        images_found[file_stem],
                        width=Inches(box_w_inches),
                        height=Inches(box_h_inches),
                    )
                except Exception as e:
                    print(f"Warning: Could not load image '{template_var}': {e}")
                    render_context[template_var] = None
            else:
                render_context[template_var] = None
                if validate_images:
                    print(f"Warning: No image found for '{template_var}' (expected stem: '{file_stem}')")
        else:
            # Pass through everything else — lists, dicts, scalars, numpy types, etc.
            render_context[template_var] = value
    # print(f"Rendered context: {render_context}")

    # --- Render and save ---
    output_filename = filename or f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.docx"
    output_path = os.path.join(output_dir, output_filename)

    tpl.render(render_context)
    tpl.save(output_path)

    print("\nDocument generated successfully!")
    print(f"Output: {output_path}")
    print("=" * 80)

    return str(os.path.abspath(output_path))
