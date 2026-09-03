import types
from pathlib import Path

import geopandas as gpd
import pytest
from numpy.testing import assert_array_equal

from parcel2d_modflow.exceptions import ValidationError
from parcel2d_modflow.preprocessing.parcels import (
    closest_weather_station,
    prepare_parcels,
)
from parcel2d_modflow.validation.validate import ValidationWarning


@pytest.fixture
def test_stations():
    return gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "station": ["A", "B", "C"],
            "geometry": gpd.points_from_xy([1.5, 2, 4], [1.1, 2, 3]),
            "weather_rg": ["northeast", "southwest", "southwest"],
        },
        crs=28992,
    )


@pytest.mark.unittest
def test_prepare_parcels(parcels, model_settings, soilmap, weather_data):
    with pytest.warns(
        ValidationWarning,
        match="Input parcels GeoDataFrame contains unknown parcel attributes: soil_unit",
    ):
        parcels = prepare_parcels(parcels, model_settings, soilmap, weather_data)
    assert isinstance(parcels, types.GeneratorType)

    p1 = next(parcels)
    assert p1.name == "A"
    assert Path(model_settings.workdir / f"{p1.name + '_' + p1.soilcode}").exists()
    assert p1.x == p1.y == 1.0
    assert p1.width == 2
    assert p1.surface_level == -2.0
    assert p1.soilcode == "hVb"
    assert p1.summer_stage == -2.4
    assert p1.winter_stage == -2.5
    assert p1.nearest_weather_station == 260
    assert p1.weather_rg == "northeast"
    assert_array_equal(p1.soilprofile["normalsoilprofile_id"], [1010, 1010, 1010, 1010])
    assert_array_equal(
        p1.soilprofile.columns,
        [
            "normalsoilprofile_id",
            "lowervalue",
            "uppervalue",
            "organicmattercontent",
            "peattype",
            "loamcontent",
            "lutitecontent",
            "siltcontent",
            "cnratio",
            "soilunit_code",
            "sand",
            "lithology",
            "thickness",
            "geology",
        ],
    )

    p2 = next(parcels)
    assert p2.name == "B"
    assert Path(model_settings.workdir / f"{p2.name + '_' + p2.soilcode}").exists()
    assert p2.x == 3.0
    assert p2.y == 1.0
    assert p2.width == 2
    assert p2.surface_level == -2.0
    assert p2.soilcode == "hVc"
    assert p2.summer_stage == -2.4
    assert p2.winter_stage == -2.5
    assert p2.nearest_weather_station == 260
    assert p2.weather_rg == "northeast"
    assert_array_equal(p2.soilprofile["normalsoilprofile_id"], [1050, 1050, 1050, 1050])


@pytest.mark.unittest
def test_prepare_parcels_validation_error(
    parcels, model_settings, soilmap, weather_data
):
    # Drop a mandatory column to trigger validation error
    parcels = parcels.drop(columns=["name"])
    with pytest.raises(
        ValidationError,
        match="Parcels DataFrame is missing mandatory columns: name.",
    ):
        prepare_parcels(parcels, model_settings, soilmap, weather_data)


@pytest.mark.unittest
def test_closest_weather_station(parcels, test_stations):
    index = closest_weather_station(parcels, test_stations)
    assert_array_equal(index, [0, 1])
