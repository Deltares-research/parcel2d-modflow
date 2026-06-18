from pathlib import Path

import pandas as pd
import pytest

from parcel2d_modflow.base import ModelSettings, Parcel
from parcel2d_modflow.modeldata import Presets


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
