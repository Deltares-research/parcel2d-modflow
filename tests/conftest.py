import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely import geometry as gmt

from parcel2d_modflow import read_lhm_data
from parcel2d_modflow.base import ModelSettings, Parcel
from parcel2d_modflow.modeldata import Presets, Soilmap


@pytest.fixture
def testdatadir():
    return Path(__file__).parent / "data"


@pytest.fixture
def model_settings(tmp_path):
    return ModelSettings(
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
    `somers.modeldata.LhmData` fixture that reads LHM confining, flux, and recharge data
    from the LHM NetCDF fixtures.

    """
    return read_lhm_data(
        lhm_confining_nc, lhm_flux_nc, lhm_recharge_nc, lhm_phreatic_head_nc
    )


@pytest.fixture
def simple_bro_soilmap():
    """
    Small extraction of 4 soilunits from the BRO soilmap geopackage.

    """
    return Path(__file__).parent / r"data/test_soilmap_v2023.gpkg"


def _create_preset(data, date_range, name):
    """
    Helper function to create a pandas DataFrame with a given name and date range to create
    dummy input for preset variables.

    """
    data = pd.DataFrame(data, columns=[name], index=date_range)
    data.index.name = "time"
    return data


@pytest.fixture
def presets(model_settings):
    """
    `Presets` fixture containing dummy input for all optional somers presets.

    """
    resistance = 5000
    recharge = _create_preset(
        [
            4.811e-04,
            2.928e-03,
            3.807e-05,
            3.652e-04,
            3.970e-03,
            9.327e-04,
            1.742e-03,
        ],
        model_settings.date_range,
        "recharge (m/d)",
    )
    flux = _create_preset(
        [-0.000936, -0.000951, -0.000949, -0.000909, -0.000929, -0.000924, -0.000922],
        model_settings.date_range,
        "preset_aquifer_flux (m/d)",
    )
    ditch_stage = _create_preset(
        [-2.51, -2.51, -2.51, -2.50, -2.50, -2.49, -2.49],
        model_settings.date_range,
        "phreatic_head (m-nap)",
    )
    pssi_stage = _create_preset(
        [-2.51, -2.51, -2.50, -2.50, -2.50, -2.49, -2.49],
        model_settings.date_range,
        "phreatic_head (m-nap)",
    )
    return Presets(
        resistance,
        recharge,
        flux,
        ditch_stage,
        pssi_stage,
        soilcode=None,
        carbon_profile=None,
    )
