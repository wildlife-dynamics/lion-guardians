from typing import cast,Annotated,Literal
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerClient
from ecoscope_workflows_ext_ecoscope.tasks.io._earthranger import (
    CombinedPatrolAndEventsParams,
    PatrolsDF,EmptyDataFrame,get_patrols,
    PatrolObservationsGDF,
    IncludePatrolDetailsAnnotation,
    RaiseOnEmptyAnnotation,
    SubPageSizeAnnotation
)

from ecoscope_workflows_ext_ecoscope.schemas import (
    EventGDF,
    EventsWithDisplayNamesGDF,
    PatrolObservationsGDF,
    PatrolsDF,
)
from ecoscope_workflows_core.annotations import (
    AdvancedField,
    EmptyDataFrame,
)

AppendCategorySelection = Literal["duplicates", "always", "never"]
AppendCategorySelectionAnnotation = Annotated[
    AppendCategorySelection, AdvancedField(default="duplicates")
]

@task
def get_patrol_observations_from_patrols_dataframe(
    client: EarthRangerClient,
    patrols_df: PatrolsDF,
    include_patrol_details: IncludePatrolDetailsAnnotation = True,
    raise_on_empty: RaiseOnEmptyAnnotation = True,
    sub_page_size: SubPageSizeAnnotation = 100,
) -> PatrolObservationsGDF | EmptyDataFrame:
    """Get observations for a patrol type from EarthRanger."""
    from ecoscope.relocations import Relocations  # type: ignore[import-untyped]

    patrol_obs_relocs = client.get_patrol_observations(
        patrols_df=patrols_df,
        include_patrol_details=include_patrol_details,
        sub_page_size=sub_page_size,
    )
    if isinstance(patrol_obs_relocs, Relocations):
        patrol_obs_relocs = patrol_obs_relocs.gdf

    if raise_on_empty and patrol_obs_relocs.empty:
        raise ValueError(
            "No data returned from EarthRanger for get_patrol_observations_with_patrol_filter"
        )

    return cast(PatrolObservationsGDF, patrol_obs_relocs)

@task 
def get_patrols_from_combined_parameters(
    combined_params: CombinedPatrolAndEventsParams
)-> PatrolsDF | EmptyDataFrame:
    return get_patrols.validate().call(**combined_params.get_patrols_params())

@task
def get_patrol_observations_from_patrols_dataframe_and_combined_params(
    patrols_df: PatrolsDF,
    combined_params: CombinedPatrolAndEventsParams,
) -> PatrolObservationsGDF | EmptyDataFrame:
    return get_patrol_observations_from_patrols_dataframe.validate().call(
        patrols_df=patrols_df,
        **combined_params.get_patrol_observations_from_patrols_df_params(),
    )

@task
def get_event_type_display_names_from_events_aliased(
    client: EarthRangerClient,
    events_gdf: EventGDF,
    append_category_names: AppendCategorySelectionAnnotation = "duplicates",
) -> EventsWithDisplayNamesGDF:
    events_gdf = client.get_event_type_display_names_from_events(  # type: ignore[assignment]
        events_gdf=events_gdf,
        append_category_names=append_category_names,
    )
    return cast(EventsWithDisplayNamesGDF, events_gdf)