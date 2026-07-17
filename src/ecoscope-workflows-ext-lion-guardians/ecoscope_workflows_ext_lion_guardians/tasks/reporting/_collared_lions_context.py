import os
from pathlib import Path
from docxtpl import DocxTemplate
from wt_registry import register
from wt_task.skip import SkipSentinel
from typing import Dict, Optional, Any
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme
from ecoscope_workflows_ext_ste.tasks.reporting._mapbook_context import (
    DEFAULT_IMAGE_EXTENSIONS,
    build_docx_context,
    _unwrap_skip,
    _default_filename,
)
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


@register()
def render_collared_lions_page(
    template_path: str,
    output_dir: str,
    context: dict[str, Any],
    filename: Optional[str] = None,
    strict_images: bool = False,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
) -> str:
    """Render one collared lions page from a docx template and return its path.

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

    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = _default_filename(context)
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
    return str(output_path)
