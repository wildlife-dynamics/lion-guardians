from ._mapdeck import (
    draw_custom_map,
    make_text_layer,
    clean_file_keys,
    create_map_layers,
    custom_deckgl_layer,
    view_state_deck_gdf,
    create_gdf_from_dict,
    create_geojson_layer,
    load_geospatial_files,
    exclude_geom_outliers,
    remove_invalid_geometries,
    split_gdf_by_column,
    custom_deckgl_layer_from_dict,
    create_styled_layers_from_dict,
    remove_invalid_point_geometries,
    merge_static_and_grouped_layers,
    select_koi,
    set_custom_base_maps,
    create_styled_layers_from_gdf,
)

from ._zip import zip_grouped_by_key, flatten_tuple, zip_lists
from ._download_file import download_file_and_persist
from ._guardians_context import (
    create_cover_context_page,
    create_report_context,
    merge_docx_files,
    get_split_group_names,
    create_report_context_from_tuple,
)
from ._tabular import add_totals_row, rename_columns, extract_date_parts
from ._retrieve_patrols import (
    get_patrol_observations_from_patrols_dataframe,
    get_patrols_from_combined_parameters,
    get_patrol_observations_from_patrols_dataframe_and_combined_params,
    get_event_type_display_names_from_events_aliased,
)

__all__ = [
    "create_styled_layers_from_gdf",
    "create_report_context_from_tuple",
    # _mapdeck
    "zip_lists",
    "create_gdf_from_dict",
    "exclude_geom_outliers",
    "split_gdf_by_column",
    "custom_deckgl_layer_from_dict",
    "create_styled_layers_from_dict",
    "clean_file_keys",
    "create_geojson_layer",
    "create_map_layers",
    "custom_deckgl_layer",
    "draw_custom_map",
    "load_geospatial_files",
    "make_text_layer",
    "merge_static_and_grouped_layers",
    "remove_invalid_geometries",
    "remove_invalid_point_geometries",
    "select_koi",
    "set_custom_base_maps",
    "view_state_deck_gdf",
    # _zip
    "flatten_tuple",
    "zip_grouped_by_key",
    # _download_file
    "download_file_and_persist",
    # _tabular
    "add_totals_row",
    "extract_date_parts",
    "rename_columns",
    "get_patrol_observations_from_patrols_dataframe",
    "get_patrols_from_combined_parameters",
    "get_patrol_observations_from_patrols_dataframe_and_combined_params",
    "get_event_type_display_names_from_events_aliased",
    "create_cover_context_page",
    "create_report_context",
    "merge_docx_files",
    "get_split_group_names",
]
