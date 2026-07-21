from wt_registry import register
from wt_task.skip import SkipSentinel
from typing import Dict
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope_workflows_ext_ste.tasks.reporting._mapbook_context import _unwrap_skip
from ecoscope_workflows_ext_ste.tasks.transformation._tabular import safe_string


@register()
def create_collared_lions_context(
    df: AnyDataFrame | SkipSentinel | None,
    total_distance: int | float | SkipSentinel | None,
    home_range: str | SkipSentinel | None,
    speed_map: str | SkipSentinel | None,
) -> Dict[str, str | int | float | None]:
    """
    Create context dictionary for mapbook with grouper information and map paths.

    Args:
        df: The dataframe to extract grouper values from
        total_distance: Total distance value for the group
        home_range: Path to the home range map image
        speed_map: Path to the speed map image

    Returns:
        Dictionary containing grouper_value, total_distance, home_range_map, and speed_map
    """
    df = _unwrap_skip(df)
    total_distance = _unwrap_skip(total_distance)
    home_range = _unwrap_skip(home_range)
    speed_map = _unwrap_skip(speed_map)

    grouper_value = "All"
    safe_name = None

    if df is not None and "subject_name" in df.columns:
        unique_names = df["subject_name"].dropna().unique()
        if len(unique_names) == 1:
            grouper_value = str(unique_names[0])
            safe_name = safe_string(grouper_value)
        elif len(unique_names) > 1:
            # Grouped by something other than subject_name (e.g. sex/subtype):
            # per-subject file naming does not apply.
            print(f"{len(unique_names)} subject names in group; " f"cannot resolve per-subject map files.")
    else:
        print("df is None or missing 'subject_name'; using grouper_value='All'")

    print(f"grouper_value={grouper_value!r}, safe_name={safe_name!r}")

    ctx = {
        "grouper_value": grouper_value,
        "total_distance": total_distance,
        "home_range_map": home_range,
        "speed_map": speed_map,
    }

    print(f"Collared Lions context: {ctx}")
    return ctx
