import re
import requests
import pandas as pd
import geopandas as gpd
from pydantic import Field
from typing import Annotated
from wt_registry import register
from ecoscope.platform.annotations import AnyGeoDataFrame
from ecoscope_workflows_ext_custom.tasks.results._map import (
    GeoJSONLayerStyle,
    IconLayerStyle,
    LayerDefinition,
    LegendSegment,
    LegendValue,
    _color_tuple_to_css,
)


@register()
def create_spatial_features_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(
            description="Styled spatial features from get_spatial_features.",
            exclude=True,
        ),
    ],
) -> Annotated[list[LayerDefinition], Field()]:
    """Create GeoJsonLayer and IconLayer definitions from a styled spatial features GeoDataFrame."""

    class UnifiedLegend:
        GREY = "rgba(128, 128, 128, 1.0)"

        def __init__(self, gdf: gpd.GeoDataFrame) -> None:
            self.gdf = gdf
            self._cache: dict[str, str] = {}

        def __call__(self) -> LegendSegment | None:
            if "legend_label" not in self.gdf.columns:
                return None
            return LegendSegment(title=self._title(), values=self._values())

        def _title(self) -> str:
            return str(self.gdf["legend_title"].iloc[0]) if "legend_title" in self.gdf.columns else ""

        def _values(self) -> list[LegendValue]:
            seen: dict[str, str] = {}
            for _, row in self.gdf.iterrows():
                label = str(row["legend_label"])
                if label not in seen:
                    seen[label] = self._color(row)
            return [LegendValue(label=k, color=v) for k, v in seen.items()]

        def _color(self, row: pd.Series) -> str:  # type: ignore[type-arg]
            return self._icon_tint(row) or self._geom_color(row) or self._svg_color(row) or self.GREY

        def _icon_tint(self, row: pd.Series) -> str | None:  # type: ignore[type-arg]
            ic = row.get("icon_color")
            return self._css(ic) if ic is not None else None

        def _geom_color(self, row: pd.Series) -> str | None:  # type: ignore[type-arg]
            geom = row.geometry.geom_type if row.geometry is not None else ""
            fill = row.get("get_fill_color")
            line = row.get("get_line_color")
            color = line if ("LineString" in geom or not self._opaque(fill)) else fill
            return self._css(color) if self._opaque(color) else None

        def _svg_color(self, row: pd.Series) -> str | None:  # type: ignore[type-arg]
            if (url := row.get("icon_url")) is None:
                return None
            url = str(url)
            if url not in self._cache:
                self._cache[url] = self._fetch_svg(url) or self.GREY
            return self._cache[url]

        @staticmethod
        def _opaque(color: object) -> bool:
            return isinstance(color, list) and len(color) == 4 and color[3] != 0

        @staticmethod
        def _css(rgba: object) -> str:
            return _color_tuple_to_css(tuple(int(c) for c in rgba))  # type: ignore[attr-defined, arg-type]

        @staticmethod
        def _fetch_svg(url: str) -> str | None:
            try:
                resp = requests.get(url, timeout=5, verify=False)
                if match := re.search(r'fill="(#[0-9a-fA-F]{3,8})"', resp.text):
                    from ecoscope.base.utils import (  # type: ignore[import-untyped]
                        hex_to_rgba,
                    )

                    return _color_tuple_to_css(
                        tuple(int(c) for c in hex_to_rgba(match.group(1)))  # type: ignore[arg-type]
                    )
            except Exception:
                pass
            return None

    class GeoJsonLayer:
        _COLS = frozenset({"get_fill_color", "get_line_color", "get_line_width", "get_point_radius"})

        def __init__(self, gdf: gpd.GeoDataFrame, legend: LegendSegment | None) -> None:
            self.gdf = gdf
            self.legend = legend

        def __call__(self) -> LayerDefinition:
            present = self._COLS & set(self.gdf.columns)
            return LayerDefinition(
                layer_type="GeoJsonLayer",
                layer_style=GeoJSONLayerStyle(**{c: c for c in present}),
                legend=self.legend,
                geodataframe=self.gdf,  # type: ignore[arg-type]
            )

    class IconLayer:
        DEFAULT_SIZE = 15.0

        def __init__(self, gdf: gpd.GeoDataFrame, legend: LegendSegment | None) -> None:
            self.gdf = gdf
            self.legend = legend

        def __call__(self) -> LayerDefinition:
            has_custom = "icon_color" in self.gdf.columns
            gdf = self.gdf.copy()
            gdf["_icon_data"] = self._icon_data(gdf, has_custom)  # type: ignore[assignment]
            return LayerDefinition(
                layer_type="IconLayer",
                layer_style=IconLayerStyle(
                    get_icon="_icon_data",
                    get_size="icon_size" if "icon_size" in gdf.columns else self.DEFAULT_SIZE,
                    get_color="icon_color" if has_custom else None,
                ),
                legend=self.legend,
                geodataframe=gdf,  # type: ignore[arg-type]
            )

        def _icon_data(self, gdf: gpd.GeoDataFrame, has_custom: bool) -> list[dict]:  # type: ignore[type-arg]
            sizes = (
                pd.to_numeric(gdf["icon_size"], errors="coerce").fillna(self.DEFAULT_SIZE).tolist()
                if "icon_size" in gdf.columns
                else [self.DEFAULT_SIZE] * len(gdf)
            )
            return [
                {
                    "url": str(u),
                    "width": int(s),
                    "height": int(s),
                    **({"mask": True} if has_custom else {}),
                }
                for u, s in zip(gdf["icon_url"], sizes)
            ]

    gdf: gpd.GeoDataFrame = geodataframe  # type: ignore[assignment]

    if "icon_url" in gdf.columns and gdf["icon_url"].notna().any():
        mask = gdf["icon_url"].notna()
        icon_gdf = gdf[mask].explode(index_parts=False).reset_index(drop=True)
        other_gdf = gdf[~mask].copy()
    else:
        icon_gdf, other_gdf = gpd.GeoDataFrame(), gdf

    legend = UnifiedLegend(gdf)()
    layers: list[LayerDefinition] = []

    if not other_gdf.empty:
        layers.append(GeoJsonLayer(other_gdf, legend)())
    if not icon_gdf.empty:
        layers.append(IconLayer(icon_gdf, legend if other_gdf.empty else None)())

    return layers
