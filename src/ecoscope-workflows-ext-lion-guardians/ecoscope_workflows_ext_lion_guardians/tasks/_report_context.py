import os
import uuid
import logging
import warnings
import pandas as pd
from pathlib import Path
from docx import Document
from pydantic import Field
from docx.shared import Cm
from datetime import datetime
from dataclasses import asdict, dataclass
from docxcompose.composer import Composer
from docxtpl import DocxTemplate, InlineImage
from ecoscope_workflows_core.decorators import task
from typing import Annotated, Optional, Dict, Union, Any
from ecoscope_workflows_core.indexes import CompositeFilter
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_core.skip import SkipSentinel

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@dataclass
class ReportContext:
    subject_name: Optional[str] = None
    grouper_type: Optional[str] = None
    grouper_eq: Optional[str] = None
    grouper_name: Optional[str] = None
    mean_speed: Optional[Union[int, float]] = None
    min_speed: Optional[Union[int, float]] = None
    max_speed: Optional[Union[int, float]] = None
    total_distance: Optional[Union[int, float]] = None
    home_range_ecomap: Optional[str] = None


def normalize_file_url(path: str) -> str:
    """Convert file:// URL to local path, handling malformed Windows URLs."""
    if not path.startswith("file://"):
        return path

    path = path[7:]  # Remove "file://"

    if os.name == "nt":
        # Remove leading slash before drive letter: /C:/path -> C:/path
        if path.startswith("/") and len(path) > 2 and path[2] in (":", "|"):
            path = path[1:]

        path = path.replace("/", "\\")
        path = path.replace("|", ":")
    else:
        # Unix paths should start with / after removing file://
        # file:///home/user -> /home/user (already has /)
        # file://home/user -> /home/user (needs /)
        if not path.startswith("/"):
            path = "/" + path

    return path


def validate_and_prepare_image(
    tpl: DocxTemplate, value: Any, box_w_cm: float, box_h_cm: float, validate: bool = True
) -> Any:
    """Validate image path and return InlineImage or original value."""
    if not isinstance(value, str):
        return value

    path = Path(value)
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
        return value

    if validate:
        if not path.exists():
            warnings.warn(f"Image file not found: {value}")
            return None  # Return None instead of the invalid path
        if not path.is_file():
            warnings.warn(f"Image path is not a file: {value}")
            return None

    try:
        img = InlineImage(tpl, str(path), width=Cm(box_w_cm), height=Cm(box_h_cm))
        return img
    except Exception as e:
        warnings.warn(f"Failed to create InlineImage for {value}: {e}")
        return None


@task
def create_cover_context_page(
    report_period: TimeRange,
    prepared_by: str,
    count: int,
    template_path: Annotated[
        str,
        Field(
            description="Path to the .docx template file.",
        ),
    ],
    output_directory: Annotated[
        str,
        Field(
            description="Directory to save the generated .docx file.",
        ),
    ],
    filename: Annotated[
        Optional[str],
        Field(
            description="Optional filename. If not provided, a random UUID-based filename will be generated.",
            exclude=True,
        ),
    ] = None,
) -> Annotated[
    str,
    Field(
        description="Full path to the generated .docx file.",
    ),
]:
    """
    Create a context page document from a template and context dictionary.

    Args:
        template_path (str): Path to the .docx template file.
        output_directory (str): Directory to save the generated .docx file.
        context (dict): Dictionary with context values for the template.
        logo_width_cm (float): Width of the logo in centimeters. Default is 7.7.
        logo_height_cm (float): Height of the logo in centimeters. Default is 1.93.
        filename (str, optional): Optional filename for the generated file.
            If not provided, a random UUID-based filename will be generated.

    Returns:
        str: Full path to the generated .docx file.
    """
    # Normalize paths
    template_path = normalize_file_url(template_path)
    output_directory = normalize_file_url(output_directory)

    # Validate paths
    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_directory.strip():
        raise ValueError("output_directory is empty after normalization")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    os.makedirs(output_directory, exist_ok=True)
    formatted_date = datetime.now()
    formatted_date_str = formatted_date.strftime("%Y-%m-%d %H:%M:%S")
    fmt = getattr(report_period, "time_format", "%Y-%m-%d")
    formatted_time_range = f"{report_period.since.strftime(fmt)} to {report_period.until.strftime(fmt)}"

    logger.info(f"Report period: {formatted_time_range}")
    logger.info(f"Report date generated: {formatted_date_str}")
    logger.info(f"Report prepared by: {prepared_by}")
    logger.info(f"Report count: {count}")
    logger.info(f"Report ID: REP-{uuid.uuid4().hex[:8].upper()}")

    cover_page_context = {
        "report_id": f"REP-{uuid.uuid4().hex[:8].upper()}",
        "subject_count": str(count),
        "time_generated": formatted_date_str,
        "report_period": formatted_time_range,
        "prepared_by": prepared_by,
    }

    if not filename:
        filename = "context_page.docx"
    output_path = Path(output_directory) / filename

    doc = DocxTemplate(template_path)
    doc.render(cover_page_context)
    doc.save(output_path)
    return str(output_path)


@task
def create_report_context(
    template_path: str,
    output_directory: str,
    grouper_type: str,
    grouper_eq: str,
    grouper_value: str,
    subject_metrics: str,
    home_range_ecomap: Optional[str] = None,
    filename: Optional[str] = None,
    validate_images: bool = True,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
) -> str:
    """Generate a DOCX report from a template with subject metrics and images."""

    # Normalize paths
    template_path = normalize_file_url(template_path)
    output_directory = normalize_file_url(output_directory)
    subject_metrics = normalize_file_url(subject_metrics)
    if home_range_ecomap:
        home_range_ecomap = normalize_file_url(home_range_ecomap)

    # Validate paths
    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_directory.strip():
        raise ValueError("output_directory is empty after normalization")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not os.path.exists(subject_metrics):
        raise FileNotFoundError(f"Subject metrics CSV not found: {subject_metrics}")

    # Validate image path exists
    if home_range_ecomap and not os.path.exists(home_range_ecomap):
        raise FileNotFoundError(f"Home range ecomap not found: {home_range_ecomap}")

    os.makedirs(output_directory, exist_ok=True)

    # Load and validate CSV
    try:
        df = pd.read_csv(subject_metrics)
    except Exception as e:
        raise ValueError(f"Failed to read CSV {subject_metrics}: {e}")

    if df.empty:
        raise ValueError("Subject metrics CSV is empty")

    # Validate required columns
    required_cols = ["mean_speed", "min_speed", "max_speed", "total_distance", "subject_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}. " f"Available columns: {list(df.columns)}")

    # Derive subject_name from extra__name column
    unique_names = [f for f in df["subject_name"].unique() if pd.notna(f) and str(f).lower() != "total"]

    if len(unique_names) > 1:
        subject_name = "Overall"
    elif len(unique_names) == 1:
        subject_name = unique_names[0]
    else:
        warnings.warn("No valid subject names found in subject_name column, using 'Unknown'")
        subject_name = "Unknown"

    # Generate output filename based on grouper values if not provided
    if not filename:
        filename = f"report_{uuid.uuid4().hex[:4]}.docx"

    output_path = Path(output_directory) / filename

    # Extract metrics from last row
    last_row = df.iloc[-1]

    def safe_float(value: Any, default: float = 0.0, col_name: str = "") -> float:
        """Safely convert value to float with fallback."""
        if pd.isna(value):
            warnings.warn(f"{col_name} is NaN, using {default}")
            return default
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            warnings.warn(f"Cannot convert {col_name} value '{value}' to float: {e}. Using {default}")
            return default

    mean_speed = round(safe_float(last_row["mean_speed"], col_name="mean_speed"), 2)
    max_speed = round(safe_float(last_row["max_speed"], col_name="max_speed"), 2)
    min_speed = round(safe_float(last_row["min_speed"], col_name="min_speed"), 2)
    total_distance = round(safe_float(last_row["total_distance"], col_name="total_distance"), 2)

    # Initialize template
    tpl = DocxTemplate(template_path)

    # Create context with grouper information
    ctx = ReportContext(
        subject_name=subject_name,
        grouper_type=grouper_type,
        grouper_eq=grouper_eq,
        grouper_name=grouper_value,
        mean_speed=mean_speed,
        max_speed=max_speed,
        min_speed=min_speed,
        total_distance=total_distance,
        home_range_ecomap=home_range_ecomap,
    )

    # Prepare result dictionary with image handling
    result: Dict[str, Any] = {}
    for key, value in asdict(ctx).items():
        processed_value = validate_and_prepare_image(tpl, value, box_w_cm, box_h_cm, validate_images)
        result[key] = processed_value

    # Render and save
    try:
        tpl.render(result)
        tpl.save(str(output_path))

        # Verify file was created
        if not output_path.exists():
            raise RuntimeError(f"File was not created at {output_path}")

    except Exception as e:
        logger.warning(f"ERROR during render/save: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Failed to render/save template: {e}")

    return str(output_path)


def _fallback_to_none_doc(
    obj: tuple[CompositeFilter | None, str] | SkipSentinel,
) -> tuple[CompositeFilter | None, str] | None:
    return None if isinstance(obj, SkipSentinel) else obj


@dataclass
class GroupedDoc:
    """Analogous to GroupedWidget but for document pages."""

    views: dict[CompositeFilter | None, Optional[str]]

    @classmethod
    def from_single_view(cls, item: tuple[CompositeFilter | None, str]) -> "GroupedDoc":
        view, path = item
        return cls(views={view: path})

    @property
    def merge_key(self) -> str:
        """
        Determine how docs should be grouped.
        Default: group by filename stem of the first non-None path in views.
        If you want another grouping (e.g. based on metadata), replace this logic.
        """
        # pick any path available
        for p in self.views.values():
            if p:
                return Path(p).stem
        # fallback unique key (shouldn't happen normally)
        return uuid.uuid4().hex

    def __ior__(self, other: "GroupedDoc") -> "GroupedDoc":
        """Merge views from other into self. Keys must be compatible by merge_key."""
        if self.merge_key != other.merge_key:
            raise ValueError(f"Cannot merge GroupedDoc with different keys: {self.merge_key} != {other.merge_key}")
        # update views (later views override same view key)
        self.views.update(other.views)
        return self


@task
def merge_docx_files(
    cover_page_path: Annotated[str, Field(description="Path to the cover page .docx file")],
    context_page_items: Annotated[list[Any], Field(description="List of context page document paths to merge.")],
    output_directory: Annotated[str, Field(description="Directory where combined docx will be written")],
    filename: Annotated[Optional[str], Field(description="Optional output filename")] = None,
) -> Annotated[str, Field(description="Path to the combined .docx file")]:
    """
    Combine cover + context pages into a single DOCX.
    Accepts context_page_items entries that are:
      - a string path, or
      - a tuple/list containing one or more strings (path is inferred).
    """

    def extract_path(item, idx):
        if isinstance(item, str):
            return item
        if isinstance(item, (list, tuple)):
            for x in item:
                if isinstance(x, str) and os.path.exists(x):
                    return x
            for x in reversed(item):
                if isinstance(x, str):
                    return x
            raise ValueError(f"context_page_items[{idx}] is a list/tuple but contains no string path: {item!r}")
        raise ValueError(f"context_page_items[{idx}] has unsupported type {type(item)}; expected str or tuple/list")

    normalized_paths = []
    for idx, item in enumerate(context_page_items):
        if item is None:
            continue
        try:
            path = extract_path(item, idx)
        except ValueError as e:
            raise ValueError(f"Error processing context_page_items at index {idx}: {e}") from e
        normalized_paths.append(path)

    if not os.path.exists(cover_page_path):
        raise FileNotFoundError(f"Cover page file not found: {cover_page_path}")

    for p in normalized_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Context page file not found: {p}")

    output_directory = normalize_file_url(output_directory)
    if not output_directory.strip():
        raise ValueError("output_directory is empty after normalization")

    os.makedirs(output_directory, exist_ok=True)

    if not filename:
        filename = "overall_report.docx"
    output_path = Path(output_directory) / filename

    master = Document(cover_page_path)
    composer = Composer(master)

    for doc_path in normalized_paths:
        doc = Document(doc_path)
        composer.append(doc)

    composer.save(str(output_path))
    return str(output_path)


@task
def round_off_values(value: float, dp: int) -> float:
    return round(value, dp)


@task
def get_split_group_names(
    split_data: Annotated[
        list[tuple[CompositeFilter, Any]],
        Field(description="Output from split_groups: [(CompositeFilter, df), ...]"),
    ],
) -> list[str]:
    """
    Extract the first grouper value from each split group.

    Example:
        Input: [
            ((('region', '=', 'Atlantida'), ('year', '=', '2024')), df1),
            ((('region', '=', 'Yoro'),), df2),
        ]
        Output: ['Atlantida', 'Yoro']
    """
    names: list[str] = []
    for composite_filter, _ in split_data:
        if composite_filter:
            # take the first filter tuple: (index_name, '=', value)
            _, _, value = composite_filter[0]
            names.append(str(value))
        else:
            names.append("Unknown")
    return names
