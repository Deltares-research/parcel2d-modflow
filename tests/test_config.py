import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from parcel2d_modflow import config


@pytest.fixture
def base_model_settings():
    return {
        "workdir": ".",
        "start_date": dt.datetime(2022, 1, 1),
        "end_date": dt.datetime(2022, 12, 31),
    }


@pytest.mark.unittest
def test_model_settings():
    workdir = "."
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    settings = config.ModelSettings(
        workdir=workdir, start_date=start_date, end_date=end_date
    )

    expected_date_range = pd.date_range(start_date, end_date, freq="d")
    assert isinstance(settings, config.ModelSettings)
    assert isinstance(settings.workdir, Path)
    assert isinstance(settings.start_date, pd.Timestamp)
    assert isinstance(settings.end_date, pd.Timestamp)
    assert_array_equal(settings.date_range, expected_date_range)
    assert settings.date_range.name == "time"
    assert settings.summer_start == 4
    assert settings.winter_start == 10
    assert settings.dimension == "2D"
    assert settings.ditch_depth == 0.7
    assert settings.ditch_resistance == 1.0
    assert settings.min_water_depth == 0.4
    assert not settings.add_trenches
    assert settings.trench_resistance == 1.0
    assert settings.min_drain_depth == 0.2
    assert settings.soilprofile_thickness == 1.2
    assert settings.soil_layer_thickness == 0.05
    assert settings.dx == 0.5
    assert settings.dz_resistance_layer == 0.5
    assert not settings.save_flopy
    assert_array_equal(
        settings.winter_period, [True] * 90 + [False] * 183 + [True] * 92
    )

    # Test with different stress frequency and unspecified workdir (should use default)
    expected_date_range = pd.date_range(start_date, end_date, freq="h")
    model_settings = config.ModelSettings(
        start_date=start_date, end_date=end_date, stress_frequency="h"
    )
    assert isinstance(model_settings.workdir, Path)  # Default workdir should be a Path
    assert model_settings.workdir.exists()
    assert model_settings.workdir.stem == "somers_monitoring"
    assert_array_equal(model_settings.date_range, expected_date_range)

    # Test leap day handling
    settings = config.ModelSettings(start_date="2020-01-01", end_date="2020-12-31")
    assert pd.Timestamp("2020-02-29") in settings.date_range
    settings = config.ModelSettings(
        start_date="2020-01-01", end_date="2020-12-31", include_leap_days=False
    )
    assert pd.Timestamp("2020-02-29") not in settings.date_range


@pytest.mark.parametrize(
    "start, end",
    [
        (dt.date.fromisoformat("2022-01-01"), dt.date.fromisoformat("2022-12-31")),
        (pd.to_datetime("2022-01-01"), pd.to_datetime("2022-12-31")),
        ("2022-01-01", "2022-12-31"),
        ("01-01-2022", "31-12-2022"),
        ("2022/01/01", "2022/12/31"),
        ("2022-12-31", "2022-01-01"),
    ],
    ids=[
        "date",
        "pd.Timestamp",
        "iso-format-str",
        "non-iso-format-str",
        "slash-format-str",
        "end-before-start",
    ],
)
def test_model_settings_date_inputs(start, end, request):
    testcase = request.node.callspec.id
    if testcase in {"non-iso-format-str", "slash-format-str"}:
        with pytest.raises(ValidationError):
            config.ModelSettings(workdir=".", start_date=start, end_date=end)
    elif testcase == "end-before-start":
        with pytest.raises(ValidationError, match="start_date must be before end_date"):
            config.ModelSettings(workdir=".", start_date=start, end_date=end)
    else:
        settings = config.ModelSettings(workdir=".", start_date=start, end_date=end)
        assert (
            len(settings.date_range) == 365
        )  # Check if the correct date range is made.


@pytest.mark.parametrize(
    "key, valid_input, invalid_input",
    [
        ("summer_start", 4, 0),
        ("winter_start", 10, 13),
        ("dimension", "2D", "3D"),
        ("ditch_depth", 0.7, -0.1),
        ("ditch_resistance", 1.0, -0.1),
        ("min_water_depth", 0.4, -0.1),
        ("trench_resistance", 1.0, -0.1),
        ("min_drain_depth", 0.2, -0.1),
        ("soilprofile_thickness", 1.2, 1.3),
        ("soil_layer_thickness", 0.05, 0.0),
        ("dx", 0.5, 0.0),
        ("dz_resistance_layer", 0.5, 0.0),
    ],
)
def test_model_settings_inputs(base_model_settings, key, valid_input, invalid_input):
    base_model_settings[key] = valid_input
    settings = config.ModelSettings(**base_model_settings)
    assert getattr(settings, key) == valid_input

    base_model_settings[key] = invalid_input
    with pytest.raises(ValidationError):
        config.ModelSettings(**base_model_settings)


@pytest.mark.unittest
def test_run_settings():
    settings = config.RunSettings()

    assert isinstance(settings, config.RunSettings)
    assert settings.multiprocessing is True
    assert settings.batch_size == 100
    assert settings.multiprocess_scale == 1.0
    assert settings.log_level == "INFO"

    settings = config.RunSettings(
        multiprocessing=False, batch_size=50, multiprocess_scale=0.5, log_level="DEBUG"
    )
    assert settings.multiprocessing is False
    assert settings.batch_size == 50
    assert settings.multiprocess_scale == 0.5
    assert settings.log_level == "DEBUG"

    with pytest.raises(ValidationError):
        config.RunSettings(multiprocess_scale=1.5)

    with pytest.raises(ValidationError):
        config.RunSettings(log_level="info")


@pytest.mark.unittest
def test_modflow_settings():
    settings = config.ModflowSettings(
        modflow_executable="mf.exe", parameters="parameters.csv"
    )

    assert isinstance(settings, config.ModflowSettings)
    assert isinstance(settings.modflow_executable, Path)
    assert isinstance(settings.parameters, Path)
    assert settings.aquifer_method == "flux"
    assert settings.gw_recharge_method == "recharge"
    assert settings.measure == "ref"
    assert settings.evt_method == "woerkom"
    assert settings.modflow_kwargs == {}

    settings = config.ModflowSettings(
        modflow_executable="mf.exe",
        parameters="parameters.csv",
        gw_recharge_method="precip_evap",
        measure="ssi",
        evt_method="combi",
        modflow_kwargs={"key": "value"},
    )
    assert settings.gw_recharge_method == "precip_evap"
    assert settings.measure == "ssi"
    assert settings.evt_method == "combi"
    assert settings.modflow_kwargs == {"key": "value"}

    settings = config.ModflowSettings(
        modflow_executable="mf.exe",
        parameters="parameters.csv",
        measure="pssi",
        evt_method="boon",
    )
    assert settings.measure == "pssi"
    assert settings.evt_method == "boon"

    with pytest.raises(ValidationError):  # Missing mandatory field 'parameters'
        config.ModflowSettings(modflow_executable="mf.exe")

    with pytest.raises(ValidationError):  # Missing mandatory field 'modflow_executable'
        config.ModflowSettings(parameters="parameters.csv")

    with pytest.raises(ValidationError) as excinfo:
        config.ModflowSettings(
            modflow_executable="mf.exe",
            parameters="parameters.csv",
            aquifer_method="invalid_method",
            gw_recharge_method="invalid_method",
            measure="invalid_measure",
            evt_method="invalid_method",
        )
        assert "aquifer_method" in str(excinfo.value)
        assert "gw_recharge_method" in str(excinfo.value)
        assert "measure" in str(excinfo.value)
        assert "evt_method" in str(excinfo.value)


@pytest.mark.unittest
def test_data():
    data = config.InputData(
        parcels="parcels.geoparquet",
        confining_nc="confining.nc",
        flux_nc="flux.nc",
        recharge_nc="recharge.nc",
        soilmap_gpkg="soilmap.gpkg",
    )

    assert isinstance(data, config.InputData)
    assert isinstance(data.parcels, Path)
    assert isinstance(data.confining_nc, Path)
    assert isinstance(data.flux_nc, Path)
    assert isinstance(data.recharge_nc, Path)
    assert isinstance(data.soilmap_gpkg, Path)

    with pytest.raises(ValidationError):
        config.InputData(
            parcels="parcels.geoparquet",
            confining_nc=None,
            flux_nc=None,
            recharge_nc=None,
            soilmap_gpkg=None,
        )

    with pytest.raises(ValidationError):
        config.InputData(parcels="parcels.geoparquet")


@pytest.mark.unittest
def test_output():
    output = config.OutputSettings(directory="output")

    assert isinstance(output, config.OutputSettings)
    assert isinstance(output.directory, Path)
    assert output.file is None
    assert output.format == "geoparquet"
    assert output.prefix == "batch"

    output = config.OutputSettings(
        directory="output", file="output.parquet", format="parquet", prefix="test"
    )
    assert isinstance(output.file, Path)
    assert output.format == "parquet"
    assert output.prefix == "test"

    with pytest.raises(ValidationError):
        config.OutputSettings()  # Missing required field 'directory'

    with pytest.raises(ValidationError):
        config.OutputSettings(directory="output", file="output.invalid_format")

    with pytest.raises(ValidationError):
        config.OutputSettings(directory="output", format="invalid_format")
