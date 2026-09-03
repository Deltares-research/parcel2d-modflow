import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import geometry as gmt

from parcel2d_modflow import config, read_groundwater_data
from parcel2d_modflow.base import Parcel
from parcel2d_modflow.modeldata import Presets, Soilmap


@pytest.fixture
def testdatadir():
    return Path(__file__).parent / "data"


@pytest.fixture
def model_settings(tmp_path):
    return config.ModelSettings(
        workdir=tmp_path,
        start_date="2022-01-01",
        end_date="2022-01-07",
        clean_workdir=True,
    )


@pytest.fixture
def empty_parcel():
    """
    Test `Parcel` object with minimal attributes set.

    """
    return Parcel(name="a", x=1.3, y=1.2, width=20, surface_level=-2.0)


@pytest.fixture
def parcel(model_settings, soilmap):
    """
    Test `Parcel` object containing attributes from preparation functions (i.e. `discretization,
    `soilprofile`) for somers runs and most optional `Parcel` attributes set.

    """
    soilprofile = soilmap.soilprofiles.loc[
        soilmap.soilprofiles["normalsoilprofile_id"] == 1010
    ]
    soilprofile["geology"] = 1
    p = Parcel(
        name="A",
        x=1.0,
        y=1.0,
        width=2,
        surface_level=-2.0,
        soilcode="hVb",
        summer_stage=-2.4,
        winter_stage=-2.5,
        nearest_weather_station=260,
        weather_rg="northeast",
        trench_depth=0.3,
        trench_locations=1,
        drain_depth=0.7,
        drain_distance=1.0,
        pssi_summer_stage=-2.1,
        pssi_winter_stage=-2.2,
    )
    p.soilprofile = soilprofile
    p.discretize_soildepth(model_settings)
    return p


@pytest.fixture
def parcels():
    """
    Simple GeoDataFrame with two parcels for testing purposes.

    """
    return gpd.GeoDataFrame(
        {
            "name": ["A", "B"],
            "x": [1.0, 3.0],
            "y": [1.0, 1.0],
            "soil_unit": ["hVb", "hVc"],
            "soilcode": ["hVb", "hVc"],
            "width": [2, 2],
            "surface_level": [-2.0, -2.0],
            "winter_stage": [-2.5, -2.5],
            "summer_stage": [-2.4, -2.4],
            "geometry": [gmt.box(0, 0, 2, 2), gmt.box(2, 0, 4, 2)],
        },
        crs=28992,
    )


@pytest.fixture
def modflow_parameters():
    """
    Simple DataFrame with modflow parameters for two runs.

    """
    return pd.DataFrame(
        {
            "runnr": [1, 2],
            "kh (m/d)": [0.9, 0.7],
            "sy_peat (-)": [0.4, 0.5],
            "sy_clay (-)": [0.3, 0.3],
        }
    )


@pytest.fixture
def modflow_executable():
    """
    Path to the Modflow executable.

    """
    mf_dir = Path(__file__).parents[1]
    if sys.platform.startswith("win"):
        mf_exe = Path(mf_dir / r"mfutil/modflow6.exe")
    else:
        mf_exe = Path(mf_dir / r"mfutil/mf6")
    return str(mf_exe)


@pytest.fixture
def soilmap():
    """
    Test `parcel2d_modflow.modeldata.Soilmap` object with two soil units and profiles.

    """
    gdf = gpd.GeoDataFrame(
        {
            "maparea_id": ["a", "b"],
            "normalsoilprofile_id": [1010, 1050],
            "soilunit_code": ["hVb", "hVc"],
            "geometry": [gmt.box(0, 0, 2, 2), gmt.box(2, 0, 4, 2)],
        },
        crs=28992,
    )
    profiles = pd.DataFrame(
        {
            "normalsoilprofile_id": [1010, 1010, 1010, 1010, 1050, 1050, 1050, 1050],
            "lowervalue": [0, 0.2, 0.35, 0.7, 0, 0.15, 0.3, 0.5],
            "uppervalue": [0.2, 0.35, 0.7, 1.2, 0.15, 0.3, 0.5, 1.2],
            "organicmattercontent": [0.35, 0.25, 0.50, 0.70, 0.35, 0.50, 0.75, 0.80],
            "peattype": [
                "verweerdKleirijk",
                "",
                "bosveen",
                "bosveen",
                "verweerdKleirijk",
                "",
                "zeggeveen",
                "zeggeveen",
            ],
            "loamcontent": [80, 95, 95, 95, 80, 95, 75, 75],
            "lutitecontent": [40, 60, 60, 60, 40, 60, 18, 18],
            "siltcontent": [40, 35, 35, 35, 40, 35, 57, 57],
            "cnratio": [12, 14, 18, 18, 12, 14, 22, 22],
            "soilunit_code": ["hVb", "hVb", "hVb", "hVb", "hVc", "hVc", "hVc", "hVc"],
            "sand": [20, 5, 5, 5, 20, 5, 25, 25],
            "lithology": [3, 2, 1, 1, 3, 3, 1, 1],
            "thickness": [0.2, 0.15, 0.35, 0.5, 0.15, 0.15, 0.2, 0.7],
        }
    )
    return Soilmap(gdf, profiles)


@pytest.fixture
def lhm_confining_nc(testdatadir):
    """
    Fixture to create a tmp netcdf file that contains relevant LHM confining layer
    information to test.

    """
    return testdatadir / "lhm_confining.nc"


@pytest.fixture
def lhm_flux_nc(testdatadir):
    """
    Fixture to create a tmp netcdf file that contains relevant LHM flux information to test.

    """
    return testdatadir / "lhm_flux.nc"


@pytest.fixture
def lhm_recharge_nc(testdatadir):
    """
    Fixture to create a tmp netcdf file that contains relevant LHM recharge information to test.

    """
    return testdatadir / "lhm_recharge.nc"


@pytest.fixture
def lhm_phreatic_head_nc(testdatadir):
    """
    Fixture to create a tmp netcdf file that contains relevant LHM phreatic head information to test.
    """
    return testdatadir / "lhm_phreatic_head.nc"


@pytest.fixture
def lhm_data(lhm_confining_nc, lhm_flux_nc, lhm_recharge_nc, lhm_phreatic_head_nc):
    """
    Fixture that reads LHM confining, flux, and recharge data from the LHM NetCDF fixtures
    to use as `GroundwaterData`.

    """
    return read_groundwater_data(
        lhm_confining_nc, lhm_flux_nc, lhm_recharge_nc, lhm_phreatic_head_nc
    )


@pytest.fixture
def simple_bro_soilmap(testdatadir):
    """
    Small extraction of 4 soilunits from the BRO soilmap geopackage.

    """
    return testdatadir / r"test_soilmap_v2023.gpkg"


@pytest.fixture
def config_instance(testdatadir, model_settings, modflow_executable, tmp_path):
    return config.Config(
        settings=model_settings,
        modflow_settings=config.ModflowSettings(
            modflow_executable=modflow_executable,
            parameters=testdatadir / "test_parameters.csv",
        ),
        run_settings=config.RunSettings(),
        data=config.InputData(
            parcels=testdatadir / "test_parcels.geoparquet",
            confining_nc=testdatadir / "lhm_confining.nc",
            flux_nc=testdatadir / "lhm_flux.nc",
            recharge_nc=testdatadir / "lhm_recharge.nc",
            soilmap_gpkg=testdatadir / "test_soilmap_v2023.gpkg",
        ),
        output=config.OutputSettings(directory=tmp_path / "output"),
    )


@pytest.fixture
def weather_station_shape(tmp_path):
    """
    Dummy weather station shapefile with one weather station.

    """
    stations = gpd.GeoDataFrame(
        {"id": [260], "station": ["De Bilt"], "geometry": [gmt.Point(2.0, 2.0)]},
        crs=28992,
    )
    outfile = tmp_path / "stations.shp"
    stations.to_file(outfile)
    return outfile


@pytest.fixture
def knmi_measurement_data(testdatadir):
    """
    Small selection of KNMI temperature data for testing purposes in the format of data
    downloaded from https://daggegevens.knmi.nl/klimatologie/daggegevens.

    """
    return testdatadir / r"knmi_measurements.txt"


@pytest.fixture
def weather_regions(tmp_path):
    """
    Dummy shapefile with one weather region.

    """
    regions = gpd.GeoDataFrame(
        {"weather_rg": ["northeast"], "geometry": [gmt.box(0, 0, 4, 4)]},
        crs=28992,
    )
    outfile = tmp_path / "regions.shp"
    regions.to_file(outfile)
    return outfile


@pytest.fixture
def weather_data(weather_station_shape, knmi_measurement_data, weather_regions):
    """
    `somers.modeldata.WeatherData` fixture that reads weather data from the weather station
    shape, KNMI temperature data, and weather regions fixtures.

    """
    from parcel2d_modflow import read_weather_data

    return read_weather_data(
        weather_station_shape, knmi_measurement_data, weather_regions
    )


@pytest.fixture
def presets():
    """
    `Presets` fixture containing dummy input for all optional preset data.

    """
    resistance = 5000
    time = pd.date_range("2021-01-01", "2024-12-31", freq="D")

    # Create dummy seasonal variation following a sinus curve with amplitude of 0.2 and
    # a period of one year and add some random noise.
    rng = np.random.default_rng(seed=42)
    season_variation = 0.2 * np.sin(2 * np.pi * (time.dayofyear - 91) / 365.25)
    noise = rng.normal(0, 0.03, len(time))

    ditch_stage = xr.DataArray(
        [-2.45 + season_variation + noise],
        coords={"name": ["A"], "time": time},
        dims=("name", "time"),
    )
    pssi_stage = xr.DataArray(
        [-2.15 + season_variation + noise],
        coords={"name": ["A"], "time": time},
        dims=("name", "time"),
    )

    return Presets(
        resistance=resistance, ditch_stage=ditch_stage, pssi_stage=pssi_stage
    )
