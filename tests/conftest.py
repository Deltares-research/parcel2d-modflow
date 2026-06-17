from pathlib import Path

import pandas as pd
import pytest

from parcel2d_modflow.base import ModelSettings, Parcel


@pytest.fixture
def testdatadir():
    return Path(__file__).parent / "data"


@pytest.fixture
def model_settings(tmp_path):
    return ModelSettings(
        workdir=tmp_path,
        start_date="2022-01-01",
        end_date="2022-07-01",
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
