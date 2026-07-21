"""Tests for ecoscope_workflows_ext_lion_guardians.tasks.results._spatial_features.

`create_spatial_features_layer` expects an already-styled GeoDataFrame (the
styling step -- e.g. `_apply_geo_style` in ecoscope-workflows-ext-custom --
lives upstream of this task), so fixtures here construct gdfs with the
deck.gl-ready columns (`get_fill_color`, `get_line_color`, `icon_url`, ...)
directly rather than deriving them from raw style config.
"""

from __future__ import annotations

from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from ecoscope_workflows_ext_lion_guardians.tasks.results._spatial_features import (
    create_spatial_features_layer,
)

MODULE = "ecoscope_workflows_ext_lion_guardians.tasks.results._spatial_features"


def _polygon_gdf(**extra) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
            **extra,
        },
        crs="EPSG:4326",
    )


def _styled_polygon_gdf(**extra) -> gpd.GeoDataFrame:
    return _polygon_gdf(
        get_fill_color=[[255, 0, 0, 255], [0, 255, 0, 255]],
        get_line_color=[[0, 0, 0, 255], [0, 0, 0, 255]],
        get_line_width=[2.0, 2.0],
        legend_label=["Park", "Reserve"],
        legend_title=["Parks", "Parks"],
        **extra,
    )


def _icon_gdf(**extra) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0), Point(1, 1)],
            "icon_url": ["/static/ranger.svg", "/static/ranger.svg"],
            "icon_size": [20.0, 20.0],
            "legend_label": ["Ranger", "Ranger"],
            "legend_title": ["Rangers", "Rangers"],
            **extra,
        },
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# create_spatial_features_layer -- layer splitting & structure
# ---------------------------------------------------------------------------
class TestCreateSpatialFeaturesLayer:
    def test_polygon_only_returns_one_geojson_layer(self):
        layers = create_spatial_features_layer(geodataframe=_styled_polygon_gdf())
        assert len(layers) == 1
        assert layers[0].layer_type == "GeoJsonLayer"

    def test_icon_only_returns_one_icon_layer(self):
        layers = create_spatial_features_layer(geodataframe=_icon_gdf())
        assert len(layers) == 1
        assert layers[0].layer_type == "IconLayer"

    def test_mixed_returns_both_layer_types(self):
        mixed = gpd.GeoDataFrame(
            pd.concat(
                [
                    _styled_polygon_gdf(icon_url=[None, None], icon_size=[None, None]),
                    _icon_gdf(),
                ],
                ignore_index=True,
            ),
            crs="EPSG:4326",
        )
        types = {layer.layer_type for layer in create_spatial_features_layer(geodataframe=mixed)}
        assert types == {"GeoJsonLayer", "IconLayer"}

    def test_legend_on_geojson_not_icon_when_mixed(self):
        mixed = gpd.GeoDataFrame(
            pd.concat(
                [
                    _styled_polygon_gdf(icon_url=[None, None], icon_size=[None, None]),
                    _icon_gdf(),
                ],
                ignore_index=True,
            ),
            crs="EPSG:4326",
        )
        layers = create_spatial_features_layer(geodataframe=mixed)
        geojson = next(layer for layer in layers if layer.layer_type == "GeoJsonLayer")
        icon = next(layer for layer in layers if layer.layer_type == "IconLayer")
        assert geojson.legend is not None
        assert icon.legend is None

    def test_legend_on_icon_when_icon_only(self):
        layers = create_spatial_features_layer(geodataframe=_icon_gdf())
        assert layers[0].legend is not None

    def test_empty_gdf_returns_empty_list(self):
        gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
        assert create_spatial_features_layer(geodataframe=gdf) == []

    def test_geojson_style_columns_wired(self):
        layers = create_spatial_features_layer(geodataframe=_styled_polygon_gdf())
        style = layers[0].layer_style
        assert style.get_fill_color == "get_fill_color"
        assert style.get_line_color == "get_line_color"
        assert style.get_line_width == "get_line_width"

    def test_geojson_style_omits_absent_columns(self):
        gdf = _polygon_gdf(get_fill_color=[[255, 0, 0, 255], [255, 0, 0, 255]])
        layers = create_spatial_features_layer(geodataframe=gdf)
        style = layers[0].layer_style
        assert style.get_fill_color == "get_fill_color"
        assert style.get_line_color is None

    def test_icon_data_has_url_width_height(self):
        layers = create_spatial_features_layer(geodataframe=_icon_gdf())
        data = layers[0].geodataframe["_icon_data"].iloc[0]
        assert data["url"] == "/static/ranger.svg"
        assert data["width"] == 20
        assert data["height"] == 20

    def test_icon_data_no_mask_without_custom_color(self):
        layers = create_spatial_features_layer(geodataframe=_icon_gdf())
        assert "mask" not in layers[0].geodataframe["_icon_data"].iloc[0]

    def test_icon_data_has_mask_with_custom_icon_color(self):
        gdf = _icon_gdf(icon_color=[[255, 0, 0, 255], [255, 0, 0, 255]])
        layers = create_spatial_features_layer(geodataframe=gdf)
        assert layers[0].geodataframe["_icon_data"].iloc[0].get("mask") is True

    def test_icon_style_construction_succeeds_regardless_of_custom_color(self):
        # The installed IconLayerStyle (pinned via ecoscope-workflows-ext-custom)
        # has no `get_color` field, so pydantic's default extra="ignore" silently
        # drops the `get_color=...` kwarg either way -- construction must not raise.
        for gdf in (
            _icon_gdf(),
            _icon_gdf(icon_color=[[255, 0, 0, 255], [255, 0, 0, 255]]),
        ):
            layers = create_spatial_features_layer(geodataframe=gdf)
            assert getattr(layers[0].layer_style, "get_color", None) is None

    def test_icon_get_size_references_column_when_present(self):
        layers = create_spatial_features_layer(geodataframe=_icon_gdf())
        assert layers[0].layer_style.get_size == "icon_size"

    def test_icon_get_size_falls_back_to_default_literal_when_absent(self):
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0)],
                "icon_url": ["/static/ranger.svg"],
            },
            crs="EPSG:4326",
        )
        layers = create_spatial_features_layer(geodataframe=gdf)
        assert layers[0].layer_style.get_size == 15.0

    def test_multipoint_exploded_for_icon_layer(self):
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [MultiPoint([(0, 0), (1, 1)])],
                "icon_url": ["/static/ranger.svg"],
                "icon_size": [20.0],
            },
            crs="EPSG:4326",
        )
        layers = create_spatial_features_layer(geodataframe=gdf)
        icon_layer = next(layer for layer in layers if layer.layer_type == "IconLayer")
        assert len(icon_layer.geodataframe) == 2
        assert all(g.geom_type == "Point" for g in icon_layer.geodataframe.geometry)

    def test_no_legend_when_no_legend_label_column(self):
        gdf = _polygon_gdf(get_fill_color=[[255, 0, 0, 255], [255, 0, 0, 255]])
        layers = create_spatial_features_layer(geodataframe=gdf)
        assert layers[0].legend is None


# ---------------------------------------------------------------------------
# Legend colour priority: icon_tint > geom_color > svg_fetch > grey fallback
# ---------------------------------------------------------------------------
class TestLegendColourPriority:
    def _gdf(self, **cols) -> gpd.GeoDataFrame:
        gdf = _polygon_gdf(**cols)
        gdf["legend_label"] = ["Group A", "Group B"]
        gdf["legend_title"] = ["Parks", "Parks"]
        return gdf

    def _layers(self, gdf):
        return create_spatial_features_layer(geodataframe=gdf)

    def test_no_legend_when_legend_label_absent(self):
        gdf = _polygon_gdf(get_fill_color=[[255, 0, 0, 255], [0, 255, 0, 255]])
        assert self._layers(gdf)[0].legend is None

    def test_title_taken_from_legend_title_column(self):
        gdf = self._gdf(get_fill_color=[[255, 0, 0, 255], [0, 255, 0, 255]])
        assert self._layers(gdf)[0].legend.title == "Parks"

    def test_fill_colour_used_for_polygon(self):
        gdf = self._gdf(get_fill_color=[[255, 0, 0, 255], [0, 255, 0, 255]])
        color = self._layers(gdf)[0].legend.values[0].color
        assert "255, 0, 0" in color

    def test_line_colour_used_for_linestring(self):
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(2, 2), (3, 3)]),
                ],
                "get_fill_color": [[0, 0, 0, 0], [0, 0, 0, 0]],
                "get_line_color": [[0, 0, 255, 255], [0, 0, 255, 255]],
                "legend_label": ["Trail", "Road"],
                "legend_title": ["Lines", "Lines"],
            },
            crs="EPSG:4326",
        )
        color = self._layers(gdf)[0].legend.values[0].color
        assert "0, 0, 255" in color

    def test_line_colour_used_when_polygon_fill_transparent(self):
        gdf = self._gdf(
            get_fill_color=[[0, 0, 0, 0], [0, 0, 0, 0]],
            get_line_color=[[255, 165, 0, 255], [255, 165, 0, 255]],
        )
        color = self._layers(gdf)[0].legend.values[0].color
        assert "255, 165, 0" in color

    def test_icon_tint_beats_fill(self):
        gdf = self._gdf(get_fill_color=[[255, 0, 0, 255], [255, 0, 0, 255]])
        gdf["icon_color"] = [[0, 128, 0, 255], [0, 128, 0, 255]]
        color = self._layers(gdf)[0].legend.values[0].color
        assert "0, 128, 0" in color

    def test_null_icon_tint_falls_through_to_fill(self):
        gdf = self._gdf(get_fill_color=[[255, 0, 0, 255], [255, 0, 0, 255]])
        gdf["icon_color"] = [None, None]
        color = self._layers(gdf)[0].legend.values[0].color
        assert "255, 0, 0" in color

    def test_svg_colour_fetched_for_icon_url(self):
        gdf = self._gdf(
            get_fill_color=[[0, 0, 0, 0], [0, 0, 0, 0]],
            icon_url=[
                "https://er.test/static/ranger.svg",
                "https://er.test/static/ranger.svg",
            ],
        )
        with patch(f"{MODULE}.requests.get") as mocked:
            mocked.return_value.text = 'fill="#FFAB24"'
            layers = self._layers(gdf)
        assert "255, 171, 36" in layers[0].legend.values[0].color

    def test_svg_fetch_called_once_per_unique_url(self):
        gdf = self._gdf(
            get_fill_color=[[0, 0, 0, 0], [0, 0, 0, 0]],
            icon_url=[
                "https://er.test/static/ranger.svg",
                "https://er.test/static/ranger.svg",
            ],
        )
        with patch(f"{MODULE}.requests.get") as mocked:
            mocked.return_value.text = 'fill="#FFAB24"'
            self._layers(gdf)
        mocked.assert_called_once()

    def test_grey_fallback_when_no_colour_info(self):
        gdf = self._gdf()  # no colour columns at all
        color = self._layers(gdf)[0].legend.values[0].color
        assert "128, 128, 128" in color

    def test_each_unique_label_appears_once(self):
        gdf = _polygon_gdf(get_fill_color=[[255, 0, 0, 255]] * 2)
        gdf["legend_label"] = ["Same", "Same"]
        gdf["legend_title"] = ["T", "T"]
        assert len(self._layers(gdf)[0].legend.values) == 1

    def test_two_distinct_labels_both_appear(self):
        gdf = self._gdf(get_fill_color=[[255, 0, 0, 255], [0, 255, 0, 255]])
        assert len(self._layers(gdf)[0].legend.values) == 2

    def test_svg_cache_prevents_duplicate_http_calls(self):
        n = 5
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(i, i) for i in range(n)],
                "legend_label": [f"G{i}" for i in range(n)],
                "legend_title": ["T"] * n,
                "get_fill_color": [[0, 0, 0, 0]] * n,
                "icon_url": ["https://er.test/same.svg"] * n,
            },
            crs="EPSG:4326",
        )
        with patch(f"{MODULE}.requests.get") as mocked:
            mocked.return_value.text = 'fill="#646464"'
            create_spatial_features_layer(geodataframe=gdf)
        mocked.assert_called_once()

    def test_two_different_svg_urls_fetch_twice(self):
        gdf = self._gdf(
            get_fill_color=[[0, 0, 0, 0], [0, 0, 0, 0]],
            icon_url=["https://er.test/a.svg", "https://er.test/b.svg"],
        )
        with patch(f"{MODULE}.requests.get") as mocked:
            mocked.return_value.text = 'fill="#FFAB24"'
            self._layers(gdf)
        assert mocked.call_count == 2

    def test_svg_fetch_error_falls_back_to_grey(self):
        gdf = self._gdf(
            get_fill_color=[[0, 0, 0, 0], [0, 0, 0, 0]],
            icon_url=["https://er.test/broken.svg", "https://er.test/broken.svg"],
        )
        with patch(f"{MODULE}.requests.get", side_effect=Exception("network error")):
            layers = self._layers(gdf)
        assert "128, 128, 128" in layers[0].legend.values[0].color
