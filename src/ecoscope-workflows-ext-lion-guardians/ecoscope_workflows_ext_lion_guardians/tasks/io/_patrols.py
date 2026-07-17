from typing import cast
from wt_registry import register
from ecoscope.platform.tasks.io._earthranger import PatrolObservationsGDF


@register()
def filter_daytime_patrols(
    df: PatrolObservationsGDF,
    start_hour: int = 6,
    end_hour: int = 19,
) -> PatrolObservationsGDF:
    """
    Keep only patrols whose fixes start and end on the same calendar day
    and fall within daytime hours.

    Relies on the PatrolObservationsGDF schema, which guarantees a `groupby_col`
    (one value per patrol) and a timezone-aware `fixtime`.

    Parameters
    ----------
    start_hour : earliest allowed start hour (inclusive), 0–23.
    end_hour   : latest allowed end hour (exclusive), 0–23.
    """
    g = df.groupby("groupby_col")["fixtime"]
    start = g.transform("min")
    end = g.transform("max")
    same_day = start.dt.normalize() == end.dt.normalize()
    daytime = (start.dt.hour >= start_hour) & (end.dt.hour < end_hour)

    df = df[same_day & daytime].copy()
    return cast(PatrolObservationsGDF, df)
