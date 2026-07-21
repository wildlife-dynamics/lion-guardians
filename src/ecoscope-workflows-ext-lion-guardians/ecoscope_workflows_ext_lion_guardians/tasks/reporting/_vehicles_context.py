import pandas as pd
from wt_registry import register
from wt_task.skip import SkipSentinel
from typing import Dict
from ecoscope_workflows_ext_ste.tasks.reporting._mapbook_context import _unwrap_skip


@register()
def create_vehicles_context(
    summary_table: str | SkipSentinel | None,
    speedmap: str | SkipSentinel | None,
    tracks_map: str | SkipSentinel | None,
    line_chart: str | SkipSentinel | None,
) -> Dict[str, str | int | float | None]:
    """
    Create context dictionary for mapbook with grouper information and map paths.

    Args:
        grouper_name: The grouper identifier (can be various types)
        df: The dataframe to extract grouper values from
        summary_table: Path to the summary table CSV file
        speedmap: Path to the speed map image
        tracks_map: Path to the tracks map image
        line_chart: Path to the line chart image

    Returns:
        Dictionary containing grouper_value and map paths
    """
    summary_table = _unwrap_skip(summary_table)
    speedmap = _unwrap_skip(speedmap)
    tracks_map = _unwrap_skip(tracks_map)
    line_chart = _unwrap_skip(line_chart)

    # Read the CSV as a DataFrame (not converting to list)
    vehicle_plate = None
    min_speed = None
    mean_speed = None
    max_speed = None
    total_distance = None

    if summary_table:
        try:
            vehicle_stats_df = pd.read_csv(summary_table)

            # Extract values from the last row
            if not vehicle_stats_df.empty:
                if "subject_name" in vehicle_stats_df.columns:
                    vehicle_plate = vehicle_stats_df["subject_name"].iloc[-1]
                if "min_speed" in vehicle_stats_df.columns:
                    min_speed = round(vehicle_stats_df["min_speed"].iloc[-1], 2)
                if "mean_speed" in vehicle_stats_df.columns:
                    mean_speed = round(vehicle_stats_df["mean_speed"].iloc[-1], 2)
                if "max_speed" in vehicle_stats_df.columns:
                    max_speed = round(vehicle_stats_df["max_speed"].iloc[-1], 2)
                if "total_distance" in vehicle_stats_df.columns:
                    total_distance = round(vehicle_stats_df["total_distance"].iloc[-1], 2)
        except Exception as e:
            print(f"Error reading summary table {summary_table}: {e}")
    else:
        print("Warning: summary_table path is None")

    # Build context with the required keys
    ctx = {
        "vehicle_plate": vehicle_plate,
        "min_speed": min_speed,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "total_distance": total_distance,
        "vehicle_speedmap": speedmap,
        "vehicle_tracks_map": tracks_map,
        "vehicle_speed_line_chart": line_chart,
    }

    print(f"Vehicles context: {ctx}")
    return ctx
