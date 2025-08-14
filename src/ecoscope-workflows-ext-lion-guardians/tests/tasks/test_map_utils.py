import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as geom
import pytest

from ecoscope_workflows_ext_lion_guardians.tasks import (
    clean_geodataframe,
    check_shapefile_geometry_type,
    create_layer_from_gdf,
    create_view_state_from_gdf,
    annotate_gdf_dict_with_geometry_type,
    combine_map_layers,
)


class StubStyle:
    def __init__(self, styles, legend=None):
        self.styles = styles
        self.legend = legend


class StubLayer:  # stand-in for LayerDefinition
    def __init__(self, name):
        self.name = name


@pytest.fixture
def gdf_points():
    pts = [geom.Point(36.8, -1.3), geom.Point(36.82, -1.31)]
    return gpd.GeoDataFrame({"id": [1, 2]}, geometry=pts, crs="EPSG:4326")


@pytest.fixture
def gdf_lines():
    lines = [geom.LineString([(0, 0), (1, 1)]), geom.LineString([(1, 0), (2, 1)])]
    return gpd.GeoDataFrame({"id": [1, 2]}, geometry=lines, crs="EPSG:4326")


@pytest.fixture
def gdf_polys():
    polys = [geom.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
    return gpd.GeoDataFrame({"id": [1]}, geometry=polys, crs="EPSG:4326")


def test_clean_geodataframe_drops_empty_and_nan(gdf_points):
    gdf = pd.concat(
        [
            gdf_points,
            gpd.GeoDataFrame({"id": [99]}, geometry=[geom.GeometryCollection()], crs="EPSG:4326"),
            gpd.GeoDataFrame({"id": [100]}, geometry=[np.nan], crs="EPSG:4326"),
        ],
        ignore_index=True,
    )
    out = clean_geodataframe(gdf)
    # Only real points should remain (2 rows)
    assert len(out) == 2
    assert out.geometry.is_empty.any() is False
    assert out.geometry.isna().any() is False


@pytest.mark.parametrize(
    "fixture,expected",
    [
        ("gdf_points", "Point"),
        ("gdf_lines", "LineString"),
        ("gdf_polys", "Polygon"),
    ],
)
def test_check_shapefile_geometry_type_simple(request, fixture, expected):
    gdf = request.getfixturevalue(fixture)
    assert check_shapefile_geometry_type(gdf) == expected


def test_check_shapefile_geometry_type_mixed(gdf_points, gdf_lines):
    mixed = pd.concat([gdf_points, gdf_lines], ignore_index=True)
    assert check_shapefile_geometry_type(mixed) == "Mixed"


def test_create_layer_from_gdf_missing_style(gdf_points):
    style = StubStyle(styles={})
    assert create_layer_from_gdf("points", gdf_points, style, "Point") is None


def test_create_view_state_from_gdf_center(gdf_polys):
    vs = create_view_state_from_gdf(gdf_polys)
    # Polygon from (0,0) to (2,2) should center at (1,1)
    assert pytest.approx(vs.longitude, rel=1e-4) == 1.0
    assert pytest.approx(vs.latitude, rel=1e-4) == 1.0
    assert 2.0 <= vs.zoom <= 20.0


# ---- annotate_gdf_dict_with_geometry_type ----------------------------------


def test_annotate_gdf_dict_with_geometry_type(gdf_points, gdf_polys):
    d = annotate_gdf_dict_with_geometry_type({"pts": gdf_points, "poly": gdf_polys})
    assert d["pts"]["geometry_type"] == "Point"
    assert d["poly"]["geometry_type"] == "Polygon"
    assert d["pts"]["gdf"] is gdf_points


def test_combine_map_layers_lists():
    a, b = StubLayer("a"), StubLayer("b")
    out = combine_map_layers(static_layers=[a], grouped_layers=[b])
    assert [x.name for x in out] == ["a", "b"]


def test_combine_map_layers_singletons():
    a, b = StubLayer("a"), StubLayer("b")
    out = combine_map_layers(static_layers=a, grouped_layers=b)
    assert [x.name for x in out] == ["a", "b"]
