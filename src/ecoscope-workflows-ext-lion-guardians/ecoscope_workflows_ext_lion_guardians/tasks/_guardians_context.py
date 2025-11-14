import os
import uuid
import logging 
import warnings
import pandas as pd
from pathlib import Path 
from docx import Document
from docx.shared import Cm
from pydantic import Field
from datetime import datetime
from dataclasses import asdict,dataclass
from docxcompose.composer import Composer
from docxtpl import DocxTemplate,InlineImage
from ecoscope_workflows_core.decorators import task
from typing import Annotated, Optional, Any,Union,Dict
from ecoscope_workflows_core.tasks.filter._filter import TimeRange 
from ecoscope_workflows_core.indexes import CompositeFilter
from ecoscope_workflows_core.skip import SkippedDependencyFallback, SkipSentinel

@dataclass
class ReportContext:
    total_patrols: Optional[Union[int,float]] = None
    total_distance: Optional[Union[int,float]]= None 
    total_time: Optional[Union[int,float]] = None
    min_speed: Optional[Union[int,float]] = None 
    max_speed: Optional[Union[int,float]] = None 
    
    patrol_events_track_map: Optional[str] = None
    patrol_time_density_map: Optional[str] = None 
    events_pie_chart: Optional[str] = None
    events_time_series_bar_chart: Optional[str] = None 
    
    patrol_events: Optional[str] = None  # No of events recorded by guardian
    event_efforts: Optional[str]= None # No of event types recorded
    month_stats: Optional[str] = None # Month on month patrol analysis
    guardian_stats: Optional[str] = None # Guardians patrol analysis
    
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
    
logger = logging.getLogger(__name__)

def normalize_file_url(path: str) -> str:
    """Convert file:// URL to local path, handling malformed Windows URLs."""
    if not path.startswith("file://"):
        return path

    path = path[7:]
    
    if os.name == 'nt':
        # Remove leading slash before drive letter: /C:/path -> C:/path
        if path.startswith('/') and len(path) > 2 and path[2] in (':', '|'):
            path = path[1:]

        path = path.replace('/', '\\')
        path = path.replace('|', ':')
    else:
        if not path.startswith('/'):
            path = '/' + path
    
    return path

def validate_and_prepare_image(
    tpl: DocxTemplate,
    value: Any,
    box_w_cm: float,
    box_h_cm: float,
    validate: bool = True
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
        print(f"Creating InlineImage for: {value}")
        img = InlineImage(tpl, str(path), width=Cm(box_w_cm), height=Cm(box_h_cm))
        print(f"Successfully created InlineImage")
        return img
    except Exception as e:
        warnings.warn(f"Failed to create InlineImage for {value}: {e}")
        print(f"InlineImage error: {type(e).__name__}: {e}")
        return None


# custom template
@task
def create_cover_context_page(
    report_period: TimeRange,
    prepared_by: str,
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
            description="Optional filename for the generated file. If not provided, a random UUID-based filename will be generated.",
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
    formatted_time_range = (
        f"{report_period.since.strftime(fmt)} to {report_period.until.strftime(fmt)}"
    )

    logger.info(f"Report period: {formatted_time_range}")
    logger.info(f"Report date generated: {formatted_date_str}")
    logger.info(f"Report prepared by: {prepared_by}")
    logger.info(f"Report ID: REP-{uuid.uuid4().hex[:8].upper()}")
    

    cover_page_context = {
         "report_id": f"REP-{uuid.uuid4().hex[:8].upper()}",
         "time_generated": formatted_date_str,
         "report_period": formatted_time_range,
         "prepared_by": prepared_by
     }

    
    if not filename:
        filename = f"context_page_{uuid.uuid4().hex}.docx"
    output_path = Path(output_directory) / filename

    doc = DocxTemplate(template_path)
    doc.render(cover_page_context)
    doc.save(output_path)
    return str(output_path)

@task
def create_report_context(
    template_path: str,
    output_directory: str,
    patrol_type_effort_path: Optional[str] = None,
    patrol_events_track_map: Optional[str] = None,
    patrol_time_density_map: Optional[str] = None,
    events_pie_chart: Optional[str] = None,
    events_time_series_bar_chart: Optional[str] = None,
    patrol_events: Optional[str] = None,
    event_efforts: Optional[str] = None,
    month_stats: Optional[str] = None,
    guardian_stats: Optional[str] = None,
    filename: Optional[str] = None,
    validate_images: bool = True,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
) -> str:
    """
    Generate a DOCX report from a template with patrol metrics and visualizations.
    
    Args:
        template_path: Path to the DOCX template file
        output_directory: Directory where the report will be saved
        patrol_type: Type identifier for the patrol
        patrol_type_effort_path: Path to patrol effort metrics CSV
        patrol_events_track_map: Path to patrol events tracking map image
        patrol_time_density_map: Path to time density heatmap image
        events_pie_chart: Path to events distribution pie chart
        events_time_series_bar_chart: Path to events time series chart
        patrol_events: Path to patrol events data CSV
        event_efforts: Path to event efforts data CSV
        month_stats: Path to monthly statistics CSV
        guardian_stats: Path to guardian statistics CSV
        filename: Output filename (auto-generated if None)
        validate_images: Whether to validate image paths exist
        box_h_cm: Image box height in centimeters
        box_w_cm: Image box width in centimeters
    
    Returns:
        str: Path to the generated report file
        
    Raises:
        ValueError: If required paths are empty after normalization
        FileNotFoundError: If required files don't exist
    """
    
    # Group paths for easier management
    required_paths = {
        'template_path': template_path,
        'output_directory': output_directory,
    }
    
    optional_csv_paths = {
        'patrol_type_effort': patrol_type_effort_path,
        'patrol_events': patrol_events,
        'event_efforts': event_efforts,
        'month_stats': month_stats,
        'guardian_stats': guardian_stats,
    }
    
    optional_image_paths = {
        'patrol_events_track_map': patrol_events_track_map,
        'patrol_time_density_map': patrol_time_density_map,
        'events_pie_chart': events_pie_chart,
        'events_time_series_bar_chart': events_time_series_bar_chart,
    }
    
    # Normalize all paths
    normalized_paths = {}
    
    # Normalize required paths
    for key, path in required_paths.items():
        normalized = normalize_file_url(path)
        if not normalized.strip():
            raise ValueError(f"{key} is empty after normalization")
        normalized_paths[key] = normalized
    
    # Normalize optional CSV paths
    for key, path in optional_csv_paths.items():
        if path:
            normalized_paths[key] = normalize_file_url(path)
        else:
            normalized_paths[key] = None
    
    # Normalize optional image paths
    for key, path in optional_image_paths.items():
        if path:
            normalized_paths[key] = normalize_file_url(path)
        else:
            normalized_paths[key] = None
    
    # Validate required files exist
    if not os.path.exists(normalized_paths['template_path']):
        raise FileNotFoundError(
            f"Template file not found: {normalized_paths['template_path']}"
        )
    
    # Validate optional CSV files exist
    for key in optional_csv_paths.keys():
        path = normalized_paths.get(key)
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"{key} file not found: {path}")
    
    # Validate optional image files exist (if validation enabled)
    if validate_images:
        for key in optional_image_paths.keys():
            path = normalized_paths.get(key)
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"{key} image not found: {path}")
    
    # Create output directory
    os.makedirs(normalized_paths['output_directory'], exist_ok=True)
    if not filename:
        filename = f"{uuid.uuid4().hex}.docx"
    
    output_path = Path(normalized_paths['output_directory']) / filename
    patrol_metrics = pd.read_csv(normalized_paths['patrol_type_effort'])
    patrol_events = pd.read_csv(normalized_paths['patrol_events'])
    
    event_efforts = pd.read_csv(normalized_paths['event_efforts'])
    month_stats = pd.read_csv(normalized_paths['month_stats'])
    guardian_stats = pd.read_csv(normalized_paths['guardian_stats'])

    last_row = patrol_metrics.iloc[-1]
    
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
            
    total_patrols = round(safe_float(last_row["no_of_patrols"], col_name="no_of_patrols"), 0)
    total_distance = round(safe_float(last_row["total_distance"], col_name="total_distance"), 2)
    total_time = round(safe_float(last_row["total_time"], col_name="total_time"), 2)
    min_speed = round(safe_float(last_row["min_speed"], col_name="min_speed"), 2)
    max_speed = round(safe_float(last_row["max_speed"], col_name="max_speed"), 2)
    
    patrol_events_dict  = patrol_events.to_dict(orient="records")
    event_efforts_dict = event_efforts.to_dict(orient="records")
    month_stats_dict = month_stats.to_dict(orient="records")
    guardian_stats_dict = guardian_stats.to_dict(orient="records")
    
    # Initialize template
    tpl = DocxTemplate(template_path)
    
    # Create context
    # In your create_report_context function, change line 160:
    ctx = ReportContext(
        total_patrols = total_patrols,
        total_distance = total_distance,
        total_time = total_time,
        min_speed = min_speed,
        max_speed = max_speed,
    
        patrol_events_track_map = normalized_paths["patrol_events_track_map"],  # Remove the 's'
        patrol_time_density_map = normalized_paths["patrol_time_density_map"],
    
        events_pie_chart = normalized_paths["events_pie_chart"],
        events_time_series_bar_chart = normalized_paths["events_time_series_bar_chart"],
    
        patrol_events =patrol_events_dict ,
        event_efforts = event_efforts_dict, 
        month_stats =  month_stats_dict,
        guardian_stats = guardian_stats_dict
    )    # Prepare result dictionary with image handling
    result: Dict[str, Any] = {}
    for key, value in asdict(ctx).items():
        processed_value = validate_and_prepare_image(
            tpl, value, box_w_cm, box_h_cm, validate_images
        )
        result[key] = processed_value
    
    # Render and save
    try:
        tpl.render(result)
        tpl.save(str(output_path))
        
        # Verify file was created
        if not output_path.exists():
            raise RuntimeError(f"File was not created at {output_path}")
        
    except Exception as e:
        print(f"ERROR during render/save: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to render/save template: {e}")
    
    return str(output_path)



def _fallback_to_none_doc(
    obj: tuple[CompositeFilter | None, str] | SkipSentinel
    ) -> tuple[CompositeFilter | None, str] | None:
    return None if isinstance(obj, SkipSentinel) else obj


@task
def combine_docx_files(
    cover_page_path: Annotated[str, Field(description="Path to the cover page .docx file")],
    context_page_items: Annotated[
        list[
            Annotated[
                tuple[CompositeFilter | None, str],
                SkippedDependencyFallback(_fallback_to_none_doc),
            ]
        ],
        Field(description="List of context pages. Items can be SkipSentinel and will be filtered out.", exclude=True),
    ],
    output_directory: Annotated[str, Field(description="Directory where combined docx will be written")],
    filename: Annotated[Optional[str], Field(description="Optional output filename")] = None,
) -> Annotated[str, Field(description="Path to the combined .docx file")]:
    """
    Combine cover + grouped context pages into a single DOCX.
    """

    
    valid_items = [it for it in context_page_items if it is not None]    
    grouped_docs = [GroupedDoc.from_single_view(it) for it in valid_items]

    merged_map: dict[str, GroupedDoc] = {}
    for gd in grouped_docs:
        key = gd.merge_key
        if key not in merged_map:
            merged_map[key] = gd
        else:
            merged_map[key] = gd

    final_paths: list[str] = []
    for group in merged_map.values():
        for view_key, p in group.views.items():
            if p is not None:
                final_paths.append(p)

    if not os.path.exists(cover_page_path):
        raise FileNotFoundError(f"Cover page file not found: {cover_page_path}")
        
    for p in final_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Context page file not found: {p}")
    
    output_directory = normalize_file_url(output_directory)
    if not output_directory.strip():
        raise ValueError("output_directory is empty after normalization")

    os.makedirs(output_directory, exist_ok=True)
    if not filename:
        filename = f"overall_report.docx"
    output_path = Path(output_directory) / filename

    master = Document(cover_page_path)
    composer = Composer(master)
    for doc_path in final_paths:
        doc = Document(doc_path)
        composer.append(doc) 
        
    composer.save(output_path)
    return str(output_path)

@task
def round_off_values(value: float, dp: int) -> float:
    return round(value, dp)