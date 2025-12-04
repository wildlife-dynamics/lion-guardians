
import os
import re
import math
import logging
import warnings
from enum import Enum
import geopandas as gpd
from pathlib import Path
from pydantic import BaseModel,Field
from pydantic.json_schema import SkipJsonSchema
from ecoscope_workflows_core.decorators import task
from typing import Annotated,Literal, Union,List,Dict,TypedDict,Optional
from ecoscope_workflows_core.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope_workflows_ext_custom.tasks.results._map import (
    UnitType,
    ColorAccessor,
    FloatAccessor,
    LegendDefinition,
    PathLayerStyle,
    LayerStyleBase,
    ScatterplotLayerStyle,
    PolygonLayerStyle,
    TextLayerStyle,
    IconLayerStyle,
    ViewState,
    LegendFromDataframe,
    TileLayer,
    LayerDefinition,
    LegendStyle,
    view_state_from_layers,
    create_path_layer,
    create_scatterplot_layer,
    PydeckAnnotation,
    TileLayerPresets
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PYDECK_CUSTOM_LIBRARIES = [
    {
        "libraryName": "ecoscopeDeckWidgets",
        "resourceUri": "https://cdn.jsdelivr.net/npm/@ecoscope/ecoscope-deck-widgets@0.0.5/dist/bundle.js",
    }
]

# load gpkg 
class SupportedFormat(str, Enum):
    GPKG = ".gpkg"
    GEOJSON = ".geojson"
    SHP = ".shp"

SUPPORTED_FORMATS = [f.value for f in SupportedFormat]

class MapStyleConfig(BaseModel):
    styles: Dict[str, Dict] = Field(default_factory=dict)
    legend: Dict[str, List[str]] = Field(default_factory=dict)

class GeometrySummary(TypedDict):
    primary_type: Literal["Polygon", "Point", "LineString", "Other", "Mixed", "Line"]
    
class MapProcessingConfig(BaseModel):
    path: str = Field(..., description="Directory path to load geospatial files from")
    target_crs: Union[int, str] = Field(default=4326, description="Target CRS to convert maps to")
    recursive: bool = Field(default=False, description="Whether to walk folders recursively")

class GeoJsonLayerStyle(LayerStyleBase):
    stroked: Annotated[bool, AdvancedField(default=True)] = True
    filled: Annotated[bool, AdvancedField(default=True)] = True
    extruded: Annotated[bool, AdvancedField(default=False)] = False
    wireframe: Annotated[bool, AdvancedField(default=True)] = True
    get_fill_color: Annotated[
        ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)
    ] = None
    get_line_color: Annotated[
        ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)
    ] = None
    get_line_width: Annotated[
        FloatAccessor | SkipJsonSchema[None], AdvancedField(default=3)
    ] = 3
    get_elevation: Annotated[
        FloatAccessor | SkipJsonSchema[None], AdvancedField(default=0)
    ] = 0

    line_width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    line_width_scale: Annotated[float, AdvancedField(default=1)] = 1
    line_width_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    line_width_max_pixels: Annotated[
        float | SkipJsonSchema[None], AdvancedField(default=None)
    ] = None


LayerStyle = Union[
    PathLayerStyle,
    ScatterplotLayerStyle,
    PolygonLayerStyle,
    TextLayerStyle,
    IconLayerStyle,
    GeoJsonLayerStyle,  
]

@task
def create_geojson_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(description="GeoDataFrame to render as GeoJSON.", exclude=True),
    ],
    layer_style: Annotated[
        GeoJsonLayerStyle | SkipJsonSchema[None],
        AdvancedField(
            default=GeoJsonLayerStyle(),
            description="Style for the GeoJsonLayer.",
        ),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(default=None),
    ] = None,
) -> Annotated[LayerDefinition, Field()]:
    """
    Creates a GeoJsonLayer — works with Shapely, supports 3D, legend, hover.
    Use this instead of create_polygon_layer for any polygon data.
    """
    return LayerDefinition(
        layer_type="GeoJsonLayer",
        geodataframe=geodataframe,
        layer_style=layer_style or GeoJsonLayerStyle(),
        legend=legend,
    )
    
def _model_dump_with_pydeck_literals(model: LayerStyleBase):
    """
    Utility function to convert our annotated "PydeckStrings" to pdk.types.Strings at runtime
    """
    import pydeck as pdk  # type: ignore[import-untyped]

    model_dump = model.model_dump(exclude_none=True)
    for field, field_info in model.__class__.model_fields.items():
        if PydeckAnnotation in field_info.metadata and model_dump[field]:
            model_dump[field] = pdk.types.String(model_dump[field])
    return model_dump

@task
def draw_custom_map(
    geo_layers: Annotated[
        LayerDefinition | list[LayerDefinition],
        Field(description="A list of map layers to add to the map.", exclude=True),
    ],
    tile_layers: Annotated[
        list[TileLayer] | SkipJsonSchema[None],
        Field(description="A list of named tile layer with opacity, ie OpenStreetMap."),
    ] = None,
    static: Annotated[bool, Field(description="Set to true to disable map pan/zoom.")] = False,
    title: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description="Title drawn on the canvas."),
    ] = None,
    legend_style: Annotated[
        LegendStyle | SkipJsonSchema[None],
        Field(description="Additional arguments for configuring the legend."),
    ] = None,
    max_zoom: Annotated[int, Field(description="Maximum zoom level of the map.")] = 20,
    view_state: Annotated[
        ViewState | SkipJsonSchema[None],
        Field(description="Manually set the view state of the map.", exclude=True),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(description="Dashboard widget id.", exclude=True),
    ] = None,
) -> Annotated[str, Field()]:
    """
    Renders an interactive map.
    - PolygonLayer → GeoJsonLayer (fixes invisible Shapely polygons)
    - All other layers unchanged
    - Legend, title, widgets, depthTest, repeat → exactly as before
    """
    import pydeck as pdk
    import json

    pdk.settings.custom_libraries = PYDECK_CUSTOM_LIBRARIES
    DEFAULT_WIDGETS = [
        pdk.Widget("NorthArrowWidget", placement="top-left", id="NorthArrowWidget"),
        pdk.Widget("ScaleWidget", placement="bottom-left", id="ScaleWidget"),
        pdk.Widget("SaveImageWidget", placement="top-right", id="SaveImageWidget"),
    ]

    tile_layers = tile_layers or []
    legend_style = legend_style or LegendStyle()
    legend_values: list = []
    map_layers: list = []
    map_widgets: list = DEFAULT_WIDGETS.copy()

    current_max_zoom = max_zoom
    for tl in tile_layers:
        layer = pdk.Layer(
            "TiledBitmapLayer",
            data=tl.url,
            opacity=tl.opacity,
            tile_size=256,
            max_zoom=tl.max_zoom,
            min_zoom=tl.min_zoom,
            widget_id=pdk.types.String(widget_id),
        )
        map_layers.append(layer)
        if tl.max_zoom and tl.max_zoom < current_max_zoom:
            current_max_zoom = tl.max_zoom

    geo_layers = [geo_layers] if isinstance(geo_layers, LayerDefinition) else geo_layers

    for layer_def in geo_layers:
        gdf = layer_def.geodataframe.to_crs("EPSG:4326")
        layer = pdk.Layer(
                type=layer_def.layer_type,
                data=gdf,
                **_model_dump_with_pydeck_literals(layer_def.layer_style),
            )
        map_layers.append(layer)
        if layer_def.legend:
            if isinstance(layer_def.legend, list):
                legend_values.extend(layer_def.legend)
            elif isinstance(layer_def.legend, LegendFromDataframe):
                legend_values.extend(
                    layer_def.legend.build_legend_from_dataframe(layer_def.geodataframe)
                )

    if legend_values:
        map_widgets.append(
            pdk.Widget(
                "LegendWidget",
                id="LegendWidget",
                legend_values=legend_values,
                title=legend_style.display_name,
                placement=legend_style.placement,
            )
        )
    if title:
        map_widgets.append(pdk.Widget("TitleWidget", id="TitleWidget", title=title))

    deck = pdk.Deck(
        layers=map_layers,
        widgets=map_widgets,
        initial_view_state=view_state
        or view_state_from_layers(layers=geo_layers, max_zoom=current_max_zoom),
        views=pdk.View("MapView", controller=not static, repeat=True),
        map_style=pdk.map_styles.LIGHT_NO_LABELS,
        parameters={
            "depthTest": any(
                getattr(l, "extruded", False) or getattr(l, "getElevation", 0) > 0
                for l in map_layers
            )
        },
    )

    html = deck.to_html(as_string=True)
    return html.replace("@9.1.7", "@9.2.1", 2)  

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

@task
def load_geospatial_files(config: MapProcessingConfig) -> Dict[str, AnyGeoDataFrame]:
    """
    Load geospatial files from `config.path` and return a dict mapping
    relative file path -> cleaned GeoDataFrame (reprojected to target_crs).
    """
    # Convert to Path object
    base_path_str = normalize_file_url(config.path)
    base_path = Path(base_path_str)
    
    # Validate path exists
    if not base_path.exists():
        raise FileNotFoundError(f"Path does not exist: {base_path}")
    
    if not base_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base_path}")

    target_crs = config.target_crs

    loaded_files: Dict[str, AnyGeoDataFrame] = {}
    normalized_suffixes = {
        s.lower() if s.startswith(".") else f".{s.lower()}" 
        for s in SUPPORTED_FORMATS
    }
    
    # Use correct iterator method
    iterator = base_path.rglob("*") if config.recursive else base_path.iterdir()

    for p in iterator:
        try:
            if not p.is_file():
                continue

            if p.suffix.lower() not in normalized_suffixes:
                continue

            file_path = str(p)
            gdf = gpd.read_file(file_path)

            if gdf is None or gdf.empty:
                logger.info("Skipped empty or unreadable file: %s", file_path)
                continue

            if gdf.crs is None:
                logger.warning("File has no CRS, skipping reprojection: %s", file_path)
            else:
                try:
                    gdf_crs = gdf.crs
                    if gdf_crs != target_crs:
                        # Reproject to target CRS
                        gdf = gdf.to_crs(target_crs)
                except Exception as e:
                    logger.warning("Failed to normalize or compare CRS for %s: %s", file_path, e)

            # Remove invalid geometries
            cleaned = remove_invalid_geometries(gdf)
            
            # Create relative path key
            key = str(p.relative_to(base_path))
            loaded_files[key] = cleaned

        except Exception:
            logger.error("Error processing %s", p, exc_info=True)

    logger.info("Loaded %d vector files from %s", len(loaded_files), base_path)
    return loaded_files

def _build_legend_values(style_cfg: "MapStyleConfig") -> List[dict]:
    """Convert {"label": [...], "color": [...]} → [{"label":…, "color":…}]"""
    labels = style_cfg.legend.get("label", [])
    colors = style_cfg.legend.get("color", [])
    return [{"label": lbl, "color": col} for lbl, col in zip(labels, colors)]

@task 
def remove_invalid_geometries(
    gdf: Annotated[
        AnyGeoDataFrame, 
        Field(description="GeoDataFrame to filter for valid geometries.", 
              exclude=True)
              ],
) -> AnyGeoDataFrame:
    return gdf.loc[(~gdf.geometry.isna()) & (~gdf.geometry.is_empty)]

@task
def remove_invalid_point_geometries(
    gdf: Annotated[
        AnyGeoDataFrame, 
        Field(description="GeoDataFrame to filter for valid geometries.", exclude=True)
        ],
) -> AnyGeoDataFrame:
    if gdf.empty:
        logger.warning("Warning: Input dataframe is empty")
        return gdf.copy()

    if "geometry" not in gdf.columns:
        raise ValueError("DataFrame must have a 'geometry' column")

    # 1. Remove null/empty geometries
    mask_null = (
        gdf.geometry.is_empty | 
        gdf.geometry.isna() |
        gdf.geometry.isnull()
    )

    # 2. Remove (0, 0) points
    mask_zero = (
        (gdf.geometry.x == 0) & 
        (gdf.geometry.y == 0)
    )

    # Combine masks
    mask_remove = mask_null | mask_zero

    if mask_remove.any():
        removed = mask_remove.sum()
        logger.info(f"Removing {removed} invalid records: "
              f"{mask_null.sum()} null + {mask_zero.sum()} at (0,0)")
        gdf = gdf[~mask_remove].reset_index(drop=True)

    return gdf

@task
def custom_deckgl_layer(
    filename: str,
    gdf: AnyGeoDataFrame,
    style_config: "MapStyleConfig",
    primary_type: str,
) -> Optional[LayerDefinition]:
    """
    Create ONE pydeck layer for a single file.
    - Exact match on filename (lower-cased)
    - Legend attached only when it exists
    - Returns None → caller skips the file
    """
    canonical = {
        "MultiPolygon": "Polygon",
        "MultiPoint": "Point",
        "MultiLineString": "LineString",
    }.get(primary_type, primary_type)

    style_key = filename.lower()
    style_params = style_config.styles.get(style_key)
    if not style_params:
        logger.debug("No style defined for %s – skipping", filename)
        return None

    legend = None
    if _build_legend_values(style_config):
        legend = _build_legend_values(style_config)

    gdf = remove_invalid_geometries(gdf)
    try:
        if canonical == "Polygon":
            return create_geojson_layer(
                geodataframe=gdf,
                layer_style=GeoJsonLayerStyle(**style_params),
                legend=legend,
            )
        if canonical == "Point":
            return create_scatterplot_layer(
                geodataframe=gdf,
                layer_style=ScatterplotLayerStyle(**style_params),
                legend=legend,
            )
        if canonical == "LineString":
            return create_path_layer(
                geodataframe=gdf,
                layer_style=PathLayerStyle(**style_params),
                legend=legend,
            )
    except Exception as exc:
        logger.error("Failed to build %s layer for %s: %s", canonical, filename, exc, exc_info=True)

    logger.warning("Unsupported geometry %s in %s", primary_type, filename)
    return None

def _zoom_from_bbox(minx, miny, maxx, maxy, map_width_px=800, map_height_px=600) -> float:
    width_deg = abs(maxx - minx)
    height_deg = abs(maxy - miny)
    center_lat = (miny + maxy) / 2

    height_km = height_deg * 111.0
    width_km = width_deg * 111.0 * abs(math.cos(math.radians(center_lat)))

    world_width_km = 40075
    world_height_km = 40075

    zoom_for_width = math.log2(world_width_km * map_width_px / (512 * width_km))
    zoom_for_height = math.log2(world_height_km * map_height_px / (512 * height_km))

    zoom = min(zoom_for_width, zoom_for_height)
    zoom = round(max(0, min(20, zoom)), 2)
    return zoom

@task
def view_state_deck_gdf(
    gdf, 
    pitch: int = 0, 
    bearing: int = 0,
) -> ViewState:

    if gdf.empty:
        raise ValueError("GeoDataFrame is empty. Cannot compute ViewState.")

    if gdf.crs is None or not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2.0
    center_lat = (miny + maxy) / 2.0
    zoom = _zoom_from_bbox(minx, miny, maxx, maxy)
    return ViewState(
        longitude=center_lon, 
        latitude=center_lat, 
        zoom=zoom, 
        pitch=pitch, 
        bearing=bearing
        )
@task
def clean_file_keys(file_dict: Dict[str, AnyGeoDataFrame]) -> Dict[str, AnyGeoDataFrame]:
    """
    Clean dictionary keys by removing file extensions and normalizing names.
    Args:
        file_dict: Dictionary mapping filenames to GeoDataFrames.
    Returns:
        A new dictionary with standardized, lowercase keys suitable for map layer identifiers.
    """
    def clean_key(key: str) -> str:
        for ext in SUPPORTED_FORMATS:
            if key.lower().endswith(ext):
                key = key[: -len(ext)]
                break

        key = re.sub(r'\band\b', '_', key, flags=re.IGNORECASE)
        key = re.sub(r'[^A-Za-z0-9_]+', '_', key)
        key = re.sub(r'_+', '_', key)  # collapse multiple underscores
        return key.strip('_').lower()
    return {clean_key(k): v for k, v in file_dict.items()}

@task 
def select_koi(file_dict: Dict[str,AnyGeoDataFrame],key_value:str)->  AnyGeoDataFrame:
    if key_value not in file_dict.keys():
        raise ValueError(f"Key '{key_value}' not found. Available keys: {list(file_dict.keys())}")
    return file_dict[key_value]

def detect_geometry_type(gdf: AnyGeoDataFrame) -> GeometrySummary:
    """
    Detect the dominant geometry type in a GeoDataFrame and count each type.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame whose geometries will be analyzed.

    Returns:
        GeometrySummary: Dict containing the detected geometry type and counts per geometry class.
    """
    geom_counts = gdf.geometry.geom_type.value_counts().to_dict()
    unique_types = list(geom_counts.keys())

    if len(unique_types) == 1:
        geom = unique_types[0]
        mapping = {
            "Polygon": "Polygon",
            "MultiPolygon": "Polygon",
            "Point": "Point",
            "MultiPoint": "Point",
            "LineString": "LineString",
            "MultiLineString": "LineString",
        }
        primary_type = mapping.get(geom, "Other")
    else:
        primary_type = "Mixed"

    return {"primary_type": primary_type, "counts": geom_counts}


@task
def create_map_layers(
    file_dict: Dict[str, AnyGeoDataFrame], 
    style_config: MapStyleConfig
    ) -> List[LayerDefinition]:
    """
    Create styled map layers from a dictionary of GeoDataFrames using the provided style config.

    Args:
        file_dict: Dictionary mapping filenames to GeoDataFrames.
        style_config: Object holding style definitions and legend config.
    Returns:
        A list of styled map layer objects.
    """
    layers: List[LayerDefinition] = []
    cleaned_files = clean_file_keys(file_dict)

    for filename, gdf in cleaned_files.items():
        try:
            try:
                gdf = remove_invalid_geometries(gdf)
            except Exception:
                gdf = gdf.loc[(~gdf.geometry.isna()) & (~gdf.geometry.is_empty)]

            geom_analysis = detect_geometry_type(gdf=gdf)
            gdf_geom_type = geom_analysis["primary_type"]
            gdf_counts = geom_analysis.get("counts", {})
            logger.info("%s geometry type: %s counts: %s", filename, gdf_geom_type, gdf_counts)
            layer =custom_deckgl_layer(filename, gdf, style_config, gdf_geom_type)

            if layer is not None:
                layers.append(layer)

        except Exception as e:
            logger.error("Error processing layer for '%s': %s", filename, e, exc_info=True)
    logger.info("Successfully created %d map layers", len(layers))
    return layers

@task
def make_text_layer(
    txt_gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="Input GeoDataFrame with points/polygons and label data."),
    ],
    label_column: Annotated[
        str,
        Field(default="label", description="Column containing the text to display."),
    ] = "label",
    fallback_columns: Annotated[
        List[str],
        Field(
            default=["name", "title", "id"],
            description="Fallback columns to try if label_column missing.",
        ),
    ] = ["name", "title", "id"],
    use_centroid: Annotated[
        bool,
        Field(default=True, description="Place text at geometry centroid."),
    ] = True,
    color: Annotated[
        List[int],
        Field(default=[0, 0, 0, 255], description="RGBA text color [R, G, B, A]."),
    ] = [0, 0, 0, 255],
    size: Annotated[
        int,
        Field(default=16, ge=1, description="Font size in pixels."),
    ] = 16,
    font_family: Annotated[
        str,
        Field(default="Helvetica Neue", description="Font family (web-safe preferred)."),
    ] = "Helvetica Neue",
    font_weight: Annotated[
        Literal["normal", "bold", "100", "200", "300", "400", "500", "600", "700", "800", "900"],
        Field(default="normal"),
    ] = "normal",
    background: Annotated[
        bool,
        Field(default=False, description="Draw background behind text."),
    ] = False,
    background_color: Annotated[
        List[int] | None,
        Field(default=None, description="RGBA background color if background=True."),
    ] = None,
    background_padding: Annotated[
        List[float] | None,
        Field(default=None, description="[top/bottom, left/right] or [t, r, b, l] padding."),
    ] = None,
    text_anchor: Annotated[
        Literal["start", "middle", "end"],
        Field(default="middle", description="Horizontal anchor."),
    ] = "middle",
    alignment_baseline: Annotated[
        Literal["top", "center", "bottom"],
        Field(default="center", description="Vertical alignment."),
    ] = "center",
    billboard: Annotated[
        bool,
        Field(default=True, description="Always face camera (good for 3D)."),
    ] = True,
    pickable: Annotated[
        bool,
        Field(default=True, description="Enable hover tooltips."),
    ] = True,
    tooltip_columns: Annotated[
        List[str] | None,
        Field(default=None, description="Columns to show in tooltip on hover."),
    ] = None,
    target_crs: Annotated[
        str,
        Field(default="EPSG:4326", description="Output CRS (usually EPSG:4326)."),
    ] = "EPSG:4326",
) -> LayerDefinition:
    """
    Create a clean, interactive TextLayer from any GeoDataFrame.
    - Auto-finds label column
    - Centroid placement
    - Full styling via TextLayerStyle
    - Tooltip support
    - Works in 3D (billboard=True)
    """
    if txt_gdf is None or txt_gdf.empty:
        raise ValueError("txt_gdf cannot be None or empty.")

    gdf = txt_gdf.copy()

    label_col = None
    for col in [label_column] + fallback_columns:
        if col in gdf.columns:
            label_col = col
            break
    if not label_col:
        raise ValueError(
            f"No label column found. Tried: {label_column}, {fallback_columns}. "
            f"Available: {list(gdf.columns)}"
        )
    if label_col != "label":
        gdf = gdf.rename(columns={label_col: "label"})
    gdf["label"] = gdf["label"].astype(str).replace("nan", "")
    if use_centroid:
        gdf["geometry"] = gdf.centroid

    gdf = gdf.to_crs(target_crs)

    if tooltip_columns:
        keep_cols = ["geometry", "label"] + [c for c in tooltip_columns if c in gdf.columns and c != "label"]
        gdf = gdf[keep_cols]
    else:
        gdf = gdf[["geometry", "label"]]

    style = TextLayerStyle(
        get_text="label",
        get_position="geometry.coordinates",
        get_color=color,
        get_size=size,
        font_family=font_family,
        font_weight=font_weight,
        get_text_anchor=text_anchor,
        get_alignment_baseline=alignment_baseline,
        billboard=billboard,
        background=background,
        get_background_color=background_color or [255, 255, 255, 200],
        background_padding=background_padding or [4, 8],
        pickable=pickable,
        size_units="pixels",
    )

    return LayerDefinition(
        layer_type="TextLayer",
        geodataframe=gdf,
        layer_style=style,
        legend=None,  # Add legend support later if needed
    )

@task
def merge_static_and_grouped_layers(
    static_layers: Annotated[
        Union[LayerDefinition, List[LayerDefinition | List[LayerDefinition]]], 
        Field(description="Static layers from local files or base maps.")
    ] = [],
    grouped_layers: Annotated[
        Union[LayerDefinition, List[LayerDefinition | List[LayerDefinition]]],
        Field(description="Grouped layers generated from split/grouped data."),
    ] = [],
) -> list[LayerDefinition]:
    """
    Combine static and grouped map layers into a single list for rendering in `draw_ecomap`.
    Automatically flattens nested lists to handle cases where layer generation tasks return lists.
    """
    def flatten_layers(layers):
        """Recursively flatten nested lists of LayerDefinition objects."""
        if not isinstance(layers, list):
            return [layers]
        
        flattened = []
        for item in layers:
            if isinstance(item, list):
                # Recursively flatten if it's a list
                flattened.extend(flatten_layers(item))
            else:
                # Add individual LayerDefinition objects
                flattened.append(item)
        return flattened
    
    # Flatten both static and grouped layers
    flat_static = flatten_layers(static_layers) if static_layers else []
    flat_grouped = flatten_layers(grouped_layers) if grouped_layers else []
    
        # Combine all layers
    all_layers = flat_static + flat_grouped
    
    # Separate text layers from other layers
    text_layers = []
    other_layers = []
    
    for layer in all_layers:
        if isinstance(layer.layer_style, TextLayerStyle):
            text_layers.append(layer)
        else:
            other_layers.append(layer)
    
    return other_layers + text_layers

def _custom_tile_layer_json_schema() -> dict:
    schema = TileLayer.model_json_schema()
    schema["properties"]["url"]["title"] = "Custom Layer URL"
    schema["properties"]["opacity"]["title"] = "Custom Layer Opacity"
    schema["properties"]["max_zoom"]["title"] = "Custom Layer Max Zoom"
    schema["properties"]["min_zoom"]["title"] = "Custom Layer Min Zoom"
    schema["title"] = "Custom Layer (Advanced)"
    return schema

def _preset_tile_layer_json_schema(preset_name: str) -> dict:
    schema = TileLayer.model_json_schema()
    url = TileLayerPresets.get(preset_name, {}).get("url")
    title = TileLayerPresets.get(preset_name, {}).get("title")
    schema["properties"]["url"] |= {
        "const": url,
        "default": url,
        "enum": [url],
        "title": "Preset Layer URL",
    }
    schema["properties"]["url"].pop("description")
    schema["properties"]["url"].pop("pattern")
    schema["properties"].pop("max_zoom")
    schema["properties"].pop("min_zoom")
    schema["title"] = title
    return schema


def _preset_or_custom_json_schema_extra(schema: dict) -> None:
    schema["items"]["title"] = "Base Layer"
    schema["items"]["anyOf"] = [
        _preset_tile_layer_json_schema(preset) for preset in TileLayerPresets.keys()
    ]
    schema["items"]["anyOf"].append(_custom_tile_layer_json_schema())
    schema["default"] = [
        TileLayer(layer_name="TERRAIN")._as_json_schema_default(),
        TileLayer(layer_name="SATELLITE", opacity=0.5)._as_json_schema_default(),
    ]
    schema["ecoscope:advanced"] = True
    schema["items"].pop("$ref")


@task
def set_custom_base_maps(
    base_maps: Annotated[
        list[TileLayer] | SkipJsonSchema[None],
        Field(
            json_schema_extra=_preset_or_custom_json_schema_extra,
            title=" ",
            description="Select tile layers to use as base layers in map outputs.",
        ),
    ] = None,
) -> Annotated[list[TileLayer], Field()]:
    if base_maps is None:
        base_maps = [
            TileLayer(layer_name="USGS HILLSHADE", opacity=0.55),
            TileLayer(layer_name="ROADMAP", opacity=0.55),
        ]
    return base_maps