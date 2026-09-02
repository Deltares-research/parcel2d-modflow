import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from parcel2d_modflow import config
from parcel2d_modflow.constants import BestKappa, ParameterCorrectionCurve
from parcel2d_modflow.exceptions import ValidationError
from parcel2d_modflow.io import read
from parcel2d_modflow.modeldata import (
    GroundwaterData,
    ModelData,
    Presets,
    Soilmap,
    WeatherData,
)


@pytest.fixture
def modflow_parameter_file(tmp_path, modflow_parameters):
    file = tmp_path / "modflow_parameters.csv"
    modflow_parameters.to_csv(file, index=False)
    return file


@pytest.fixture
def modflow_parameter_file_missing_columns(tmp_path, modflow_parameters):
    modflow_parameters.rename(columns={"kh (m/d)": "wrong"}, inplace=True)
    file = tmp_path / "modflow_parameters_missing_columns.csv"
    modflow_parameters.to_csv(file)
    return file


@pytest.fixture
def modflow_parameter_file_wrong_index(tmp_path, modflow_parameters):
    modflow_parameters["wrong_index"] = modflow_parameters["runnr"] + 10
    file = tmp_path / "modflow_parameters_wrong_index.csv"
    modflow_parameters.to_csv(file)
    return file


@pytest.fixture
def invalid_bro_soilmap(testdatadir):
    return testdatadir / "test_invalid_soilmap_v2023.gpkg"


@pytest.mark.unittest
def test_read_config(testdatadir):
    config_file = testdatadir / "config_parcel2d.toml"
    c = read.read_config(config_file)
    assert isinstance(c, config.Config)
    assert isinstance(c.settings, config.ModelSettings)
    assert isinstance(c.modflow_settings, config.ModflowSettings)
    assert isinstance(c.data, config.InputData)
    assert isinstance(c.output, config.OutputSettings)
    assert isinstance(c.run_settings, config.RunSettings)

    with pytest.raises(ValidationError):
        read.read_config(testdatadir / "invalid_config.toml")


@pytest.mark.parametrize("extension", ["parquet", "geoparquet"])
def test_read_parcels(parcels, extension, tmp_path):
    file = tmp_path / f"parcels.{extension}"
    parcels.to_parquet(file)
    read_parcels = read.read_parcels(file)
    assert isinstance(read_parcels, gpd.GeoDataFrame)
    assert read_parcels.equals(parcels)
    assert read_parcels.crs == 28992
    assert "x" in read_parcels.columns
    assert "y" in read_parcels.columns

    # Test when parcels with unknown CRS are read
    file_no_crs = tmp_path / f"parcels_no_crs.{extension}"
    parcels.set_crs(None, allow_override=True).to_parquet(file_no_crs)
    read_parcels_no_crs = read.read_parcels(file_no_crs)
    assert read_parcels_no_crs.crs == 28992
    assert read_parcels_no_crs["x"].equals(read_parcels["x"])
    assert read_parcels_no_crs["y"].equals(read_parcels["y"])

    # Test when parcels with a different CRS are read
    file_different_crs = tmp_path / f"parcels_different_crs.{extension}"
    parcels.to_crs(epsg=4326).to_parquet(file_different_crs)
    read_parcels_different_crs = read.read_parcels(file_different_crs)
    assert read_parcels_different_crs.crs == 28992
    # Don't compare x and y coordinates here because they will be different due to reprojection


@pytest.mark.unittest
def test_read_data_from_config(config_instance):
    model_data = read.read_data_from_config(config_instance)
    assert isinstance(model_data, ModelData)
    assert isinstance(model_data.parcels, gpd.GeoDataFrame)
    assert isinstance(model_data.groundwater, GroundwaterData)
    assert isinstance(model_data.soilmap, Soilmap)
    assert isinstance(model_data.parameters, pd.DataFrame)
    assert isinstance(model_data.presets, Presets)


@pytest.mark.unittest
def test_read_groundwater_data(
    lhm_confining_nc, lhm_flux_nc, lhm_recharge_nc, lhm_phreatic_head_nc
):
    lhm_data = read.read_groundwater_data(
        lhm_confining_nc,
        lhm_flux_nc,
        lhm_recharge_nc,
        lhm_phreatic_head_nc,
    )
    assert isinstance(lhm_data, GroundwaterData)
    assert isinstance(lhm_data.confining, xr.Dataset)
    assert isinstance(lhm_data.flux, xr.DataArray)
    assert isinstance(lhm_data.recharge, xr.DataArray)
    assert isinstance(lhm_data.head, xr.DataArray)
    assert lhm_data.confining.sizes == {"x": 1, "y": 1}
    assert lhm_data.flux.sizes == {"x": 1, "y": 1, "time": 304}
    assert lhm_data.recharge.sizes == {"x": 1, "y": 1, "time": 365}
    assert lhm_data.head.sizes == {"x": 1, "y": 1, "time": 365}

    assert_array_equal(
        lhm_data.confining.data_vars,
        ["bottom", "thickness", "resistance", "k_value_1aq", "kd_value_1aq"],
    )

    # Test with all inputs None
    lhm_data = read.read_groundwater_data(None, None, None, None)
    assert isinstance(lhm_data, GroundwaterData)
    assert lhm_data.confining is None
    assert lhm_data.flux is None
    assert lhm_data.recharge is None
    assert lhm_data.head is None


@pytest.mark.unittest
def test_read_bro_soilmap(simple_bro_soilmap):
    bro_soilmap = read.read_bro_soilmap(simple_bro_soilmap)

    assert isinstance(bro_soilmap, Soilmap)
    assert isinstance(bro_soilmap.soilmap, gpd.GeoDataFrame)
    assert isinstance(bro_soilmap.soilprofiles, pd.DataFrame)

    assert_array_equal(
        bro_soilmap.soilmap.columns,
        ["maparea_id", "geometry", "normalsoilprofile_id", "soilunit_code"],
    )
    assert_array_equal(
        bro_soilmap.soilprofiles.columns,
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
        ],
    )


@pytest.mark.unittest
def test_read_invalid_soilmap(invalid_bro_soilmap):
    with pytest.raises(ValidationError):
        read.read_bro_soilmap(invalid_bro_soilmap)


@pytest.mark.unittest
def test_read_modflow_parameters(
    modflow_parameter_file,
    modflow_parameter_file_wrong_index,
    modflow_parameter_file_missing_columns,
):
    modflow_parameters = read.read_modflow_parameters(modflow_parameter_file)
    assert isinstance(modflow_parameters, pd.DataFrame)
    assert_array_equal(
        modflow_parameters.columns, ["runnr", "kh (m/d)", "sy_peat (-)", "sy_clay (-)"]
    )

    error = "Index of modflow parameters DataFrame is not correct. Expected a RangeIndex starting from 0 with step 1"
    with pytest.raises(ValidationError, match=error):
        read.read_modflow_parameters(
            modflow_parameter_file_wrong_index, index_col="wrong_index"
        )

    with pytest.raises(
        ValidationError, match="Modflow parameters DataFrame is missing columns:"
    ):
        read.read_modflow_parameters(modflow_parameter_file_missing_columns)


@pytest.mark.unittest
def test_read_weather_data(
    weather_station_shape, knmi_measurement_data, weather_regions
):
    weather_data = read.read_weather_data(
        weather_station_shape, knmi_measurement_data, weather_regions
    )
    assert isinstance(weather_data, WeatherData)
    assert isinstance(weather_data.stations, gpd.GeoDataFrame)
    assert isinstance(weather_data.measurements, pd.DataFrame)
    assert isinstance(weather_data.regions, gpd.GeoDataFrame)
    assert isinstance(weather_data.correction_params, ParameterCorrectionCurve)
    assert isinstance(weather_data.kappa, BestKappa)

    # Make sure units are converted correctly
    assert weather_data.measurements["TG"].iloc[1] == 3.6
    assert weather_data.measurements["RH"].iloc[1] == 0.0003
    assert weather_data.measurements["EV24"].iloc[1] == 0.0001

    assert_array_equal(
        weather_data.stations.columns,
        ["id", "station", "geometry", "index_right", "weather_rg"],
    )
    assert_array_equal(weather_data.measurements.columns, ["STN", "TG", "RH", "EV24"])
    assert weather_data.measurements.index.name == "YYYYMMDD"
    assert_array_equal(weather_data.regions.columns, ["weather_rg", "geometry"])
    assert weather_data.correction_params._fields == ("a", "b", "c", "d")
    assert weather_data.kappa._fields == ("rmse", "r", "nse")

    weather_data = read.read_weather_data(
        weather_station_shape,
        knmi_measurement_data,
        weather_regions,
        correction_params=dict(a=12.0),
        kappa=dict(r=3.0),
    )
    assert weather_data.correction_params.a == 12.0
    assert weather_data.kappa.r == 3.0


@pytest.mark.unittest
def test_read_knmi_temperature(knmi_measurement_data):
    temperature = read.read_knmi_measurements(knmi_measurement_data)
    assert isinstance(temperature, pd.DataFrame)
    assert_array_equal(temperature.columns, ["STN", "TG", "RH", "EV24"])
    assert temperature.index.name == "YYYYMMDD"


@pytest.mark.xfail(
    reason="Dimension names in test_ditch_da do not match expected names. First make choice about dimension names in the model run and then adapt test_ditch_da accordingly."
)
@pytest.mark.unittest
def test_read_presets(testdatadir):
    nc_file = testdatadir / "test_ditch_da.nc"
    presets = read.read_presets(ditch_stage_nc=nc_file, ssi_stage_nc=nc_file)
    assert isinstance(presets, Presets)
    assert isinstance(presets.ditch_stage, xr.DataArray)
    assert isinstance(presets.pssi_stage, xr.DataArray)
