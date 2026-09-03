import sys
from pathlib import Path
from typing import NamedTuple

import flopy
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from parcel2d_modflow import components
from parcel2d_modflow.base import Parcel
from parcel2d_modflow.config import ModelSettings
from parcel2d_modflow.exceptions import (
    InvalidInputError,
    MissingColumnError,
    ValidationError,
)
from parcel2d_modflow.mf._model import ModflowModel
from parcel2d_modflow.mf.module import Modflow
from parcel2d_modflow.modeldata import GroundwaterData, Presets, WeatherData


class Params(NamedTuple):
    """
    Small utility NamedTuple to parametrize inputs with for the `modflow_module` fixture
    in tests.

    """

    gw_recharge_method: str = "recharge"
    measure: str = "ref"


@pytest.fixture
def start_date():
    return pd.to_datetime("01-01-2022", format="%d-%m-%Y")


@pytest.fixture
def end_date():
    return pd.to_datetime("31-12-2022", format="%d-%m-%Y")


@pytest.fixture
def settings_with_trenches(model_settings):
    return model_settings.model_copy(update={"add_trenches": True})


@pytest.fixture
def modflow_module(
    modflow_parameters: pd.DataFrame,
    modflow_executable: str,
    request: pytest.FixtureRequest,
):
    """
    Empty (not-initialized) `Modflow` module.

    """
    params = getattr(
        request, "param", Params(gw_recharge_method="recharge", measure="ref")
    )

    modflow_parameters["entry_drain_resistance (d)"] = 1.0
    return Modflow(
        parameters=modflow_parameters,
        modflow_executable=modflow_executable,
        aquifer_method="flux",
        gw_recharge_method=params.gw_recharge_method,
        measure=params.measure,
    )


@pytest.fixture
def initialized_modflow_module(
    modflow_module,
    parcel: Parcel,
    settings_with_trenches: ModelSettings,
    lhm_data: GroundwaterData,
    weather_data: WeatherData,
):
    """
    Modflow module with everything initialized.

    """
    modflow_module.initialize(
        parcel, settings_with_trenches, lhm=lhm_data, weather=weather_data
    )
    return modflow_module


@pytest.fixture
def initialized_modflow_with_presets(
    modflow_module: Modflow,
    parcel: Parcel,
    settings_with_trenches: ModelSettings,
    lhm_data: GroundwaterData,
    presets: Presets,
):
    """
    Initialized `Modflow` module with "flux" method and "ssi" measure and containing all
    required components for a Modflow model run, based on the fixture `Presets`.

    """
    modflow_module.initialize(
        parcel, settings_with_trenches, lhm=lhm_data, weather=None, presets=presets
    )
    return modflow_module


class TestModflow:
    @pytest.fixture
    def empty_presets(self):
        return Presets()

    @pytest.mark.unittest
    def test_initialize_module(
        self, modflow_parameters: pd.DataFrame, modflow_executable: str
    ):
        module = Modflow(
            parameters=modflow_parameters, modflow_executable=modflow_executable
        )
        assert module.is_valid(module.name)
        assert isinstance(module, Modflow)
        assert isinstance(module.parameters, pd.DataFrame)
        assert isinstance(module.executable, Path)
        assert_array_equal(
            module.parameters.columns, ["runnr", "kh", "sy_peat", "sy_clay"]
        )
        assert module.aquifer_method == "flux"
        assert module.gw_recharge_method == "recharge"
        assert module.discretization is None
        assert module.recharge is None
        assert module.aquifer is None
        assert module.ditches is None
        assert module.trenches is None
        assert module.ssi is None
        assert module._success_and_failures is None

        module = Modflow(
            parameters=modflow_parameters,
            modflow_executable=modflow_executable,
            gw_recharge_method="precip_evap",
        )
        assert module.is_valid(module.name)
        assert isinstance(module, Modflow)
        assert module.gw_recharge_method == "precip_evap"

        with pytest.raises(ValidationError) as excinfo:
            Modflow(
                parameters=modflow_parameters,
                modflow_executable=modflow_executable,
                measure="ssi",
            )
            errors = excinfo.value.args[0]

            assert len(errors) == 1
            assert isinstance(errors[0], MissingColumnError)
            assert str(errors[0]) == (
                "Entry drain resistance is required for the measure: ssi. "
                "Please add column 'entry_drain_resistance (d)' to the parameter file."
            )

        modflow_parameters["entry_drain_resistance (d)"] = 1.0
        module = Modflow(
            parameters=modflow_parameters,
            modflow_executable=modflow_executable,
            measure="ssi",
        )
        assert isinstance(module, Modflow)

        with pytest.raises(ValidationError) as excinfo:
            Modflow(
                parameters=modflow_parameters,
                modflow_executable=modflow_executable,
                aquifer_method="invalid_method",
            )
            errors = excinfo.value.args[0]
            assert len(errors) == 1
            assert isinstance(errors[0], InvalidInputError)
            assert (
                str(errors[0])
                == "Aquifer method 'invalid_method' is not implemented for the Modflow module"
            )

        with pytest.raises(ValidationError) as excinfo:
            Modflow(
                parameters=modflow_parameters,
                modflow_executable=modflow_executable,
                measure="invalid_measure",
            )
            errors = excinfo.value.args[0]
            assert len(errors) == 1
            assert isinstance(errors[0], InvalidInputError)
            assert str(errors[0]) == (
                "Measure 'invalid_measure' is not a valid measure. "
                "Valid measures are: {'ref', 'ssi', 'pssi'}"
            )

        with pytest.raises(ValidationError) as excinfo:
            Modflow(
                parameters=modflow_parameters,
                modflow_executable=modflow_executable,
                gw_recharge_method="invalid_method",
            )
            errors = excinfo.value.args[0]
            assert len(errors) == 1
            assert isinstance(errors[0], InvalidInputError)
            assert str(errors[0]) == (
                "Recharge method 'invalid_method' is not valid. "
                "Valid methods are: {'precip_evap', 'recharge'}"
            )

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="recharge", measure="ref"),
            Params(gw_recharge_method="recharge", measure="ssi"),
            Params(gw_recharge_method="recharge", measure="pssi"),
            Params(gw_recharge_method="precip_evap", measure="ref"),
            Params(gw_recharge_method="precip_evap", measure="ssi"),
            Params(gw_recharge_method="precip_evap", measure="pssi"),
        ],
        ids=[
            "recharge-ref",
            "recharge-ssi",
            "recharge-pssi",
            "precip_evap-ref",
            "precip_evap-ssi",
            "precip_evap-pssi",
        ],
        indirect=True,
    )
    def test_discretize_parcel(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        lhm_data: GroundwaterData,
        empty_presets: Presets,
    ):
        modflow_module._discretize_parcel(
            parcel, lhm_data, preset_resistance=empty_presets.resistance
        )
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert_array_equal(
            modflow_module.discretization.thickness,
            [
                0.2,
                0.15,
                0.35,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
            ],
        )
        assert_array_equal(
            modflow_module.discretization.lithology,
            [3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4],
        )
        assert_array_equal(
            modflow_module.discretization.geology,
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
        )
        assert_array_equal(modflow_module.discretization.kvalues, [0.01, 2200.0])

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="recharge", measure="ref"),
            Params(gw_recharge_method="recharge", measure="ssi"),
            Params(gw_recharge_method="recharge", measure="pssi"),
            Params(gw_recharge_method="precip_evap", measure="ref"),
            Params(gw_recharge_method="precip_evap", measure="ssi"),
            Params(gw_recharge_method="precip_evap", measure="pssi"),
        ],
        ids=[
            "recharge-ref",
            "recharge-ssi",
            "recharge-pssi",
            "precip_evap-ref",
            "precip_evap-ssi",
            "precip_evap-pssi",
        ],
        indirect=True,
    )
    def test_discretize_parcel_with_presets(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        lhm_data: GroundwaterData,
        presets: Presets,
    ):
        modflow_module._discretize_parcel(
            parcel, lhm_data, preset_resistance=presets.resistance
        )
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert_array_equal(
            modflow_module.discretization.thickness,
            [
                0.2,
                0.15,
                0.35,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
            ],
        )
        assert_array_equal(
            modflow_module.discretization.lithology,
            [3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4],
        )
        assert_array_equal(
            modflow_module.discretization.geology,
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
        )
        assert_array_equal(modflow_module.discretization.kvalues, [0.0001, 2200.0])

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="precip_evap", measure="ref"),
            Params(gw_recharge_method="precip_evap", measure="ssi"),
            Params(gw_recharge_method="precip_evap", measure="pssi"),
        ],
        ids=["ref", "ssi", "pssi"],
        indirect=True,
    )
    def test_load_precip_evap_data(
        self, modflow_module, parcel, model_settings, weather_data
    ):
        modflow_module._load_precip_evap_data(parcel, weather_data, model_settings)
        assert isinstance(modflow_module.precipitation, components.ModflowInputSeries)
        assert isinstance(modflow_module.precipitation.start, float)
        assert isinstance(modflow_module.precipitation.series, np.ndarray)
        assert len(modflow_module.precipitation.series) == 7

        assert isinstance(
            modflow_module.evapotranspiration, components.ModflowInputSeries
        )
        assert isinstance(modflow_module.evapotranspiration.start, float)
        assert isinstance(modflow_module.evapotranspiration.series, np.ndarray)
        assert len(modflow_module.evapotranspiration.series) == 7

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="recharge", measure="ref"),
            Params(gw_recharge_method="recharge", measure="ssi"),
            Params(gw_recharge_method="recharge", measure="pssi"),
            Params(gw_recharge_method="precip_evap", measure="ref"),
            Params(gw_recharge_method="precip_evap", measure="ssi"),
            Params(gw_recharge_method="precip_evap", measure="pssi"),
        ],
        ids=[
            "recharge-ref",
            "recharge-ssi",
            "recharge-pssi",
            "precip_evap-ref",
            "precip_evap-ssi",
            "precip_evap-pssi",
        ],
        indirect=True,
    )
    def test_load_flux(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        lhm_data: GroundwaterData,
        model_settings: ModelSettings,
    ):
        modflow_module._load_aquifer(parcel, lhm_data, model_settings)

        assert isinstance(modflow_module.aquifer, components.ModflowInputSeries)
        assert modflow_module.aquifer.start == -0.000936158816
        assert isinstance(modflow_module.aquifer.series, np.ndarray)
        assert len(modflow_module.aquifer.series) == 7

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="recharge", measure="ref"),
            Params(gw_recharge_method="recharge", measure="ssi"),
            Params(gw_recharge_method="recharge", measure="pssi"),
            Params(gw_recharge_method="precip_evap", measure="ref"),
            Params(gw_recharge_method="precip_evap", measure="ssi"),
            Params(gw_recharge_method="precip_evap", measure="pssi"),
        ],
        ids=[
            "recharge-ref",
            "recharge-ssi",
            "recharge-pssi",
            "precip_evap-ref",
            "precip_evap-ssi",
            "precip_evap-pssi",
        ],
        indirect=True,
    )
    def test_load_ditches(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        empty_presets: Presets,
    ):
        settings = ModelSettings(workdir=".", start_date=start_date, end_date=end_date)

        modflow_module._load_ditches(parcel, settings, empty_presets)

        assert isinstance(modflow_module.ditches, components.Ditches)
        assert modflow_module.ditches.bottom == -2.8
        assert modflow_module.ditches.resistance == 1.0
        assert_array_equal(modflow_module.ditches.stage, [-2.5, -2.4, -2.5])
        assert_array_equal(
            modflow_module.ditches.dates,
            pd.DatetimeIndex(["2022-01-01", "2022-04-01", "2022-10-01"]),
        )

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="recharge", measure="ref"),
            Params(gw_recharge_method="recharge", measure="ssi"),
            Params(gw_recharge_method="recharge", measure="pssi"),
            Params(gw_recharge_method="precip_evap", measure="ref"),
            Params(gw_recharge_method="precip_evap", measure="ssi"),
            Params(gw_recharge_method="precip_evap", measure="pssi"),
        ],
        ids=[
            "recharge-ref",
            "recharge-ssi",
            "recharge-pssi",
            "precip_evap-ref",
            "precip_evap-ssi",
            "precip_evap-pssi",
        ],
        indirect=True,
    )
    def test_load_ditches_with_presets(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        presets: Presets,
    ):
        modflow_module._load_ditches(parcel, model_settings, presets)

        assert isinstance(modflow_module.ditches, components.Ditches)
        assert np.isclose(modflow_module.ditches.bottom, -3.074342881812138)
        assert modflow_module.ditches.resistance == 1.0
        assert_array_almost_equal(modflow_module.ditches.stage, [-2.63946319])
        assert_array_equal(
            modflow_module.ditches.dates, pd.DatetimeIndex(["2022-01-01"])
        )

    @pytest.mark.parametrize(
        "modflow_module",
        [Params(measure="ssi"), Params(measure="pssi")],
        ids=["ssi", "pssi"],
        indirect=True,
    )
    def test_load_ssi_pssi(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        empty_presets: Presets,
    ):
        settings = ModelSettings(workdir=".", start_date=start_date, end_date=end_date)
        modflow_module._load_ssi(parcel, settings, empty_presets)
        assert isinstance(modflow_module.ssi, components.SsiMeasure)
        assert modflow_module.ssi.drain_depth == -2.7
        assert modflow_module.ssi.drain_distance == 1
        assert_array_equal(modflow_module.ssi.drain_stage, [-2.2, -2.1, -2.2])
        assert_array_equal(
            modflow_module.ssi.time,
            pd.DatetimeIndex(["2022-01-01", "2022-04-01", "2022-10-01"]),
        )

    @pytest.mark.unittest
    def test_load_ssi_with_presets(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        presets: Presets,
    ):
        modflow_module.measure = "ssi"
        modflow_module._load_ssi(parcel, model_settings, presets)
        assert isinstance(modflow_module.ssi, components.SsiMeasure)
        assert modflow_module.ssi.drain_depth == -2.8743428818121384
        assert modflow_module.ssi.drain_distance == 1
        assert_array_almost_equal(
            modflow_module.ssi.drain_stage,
            [
                -2.60550536,
                -2.67214942,
                -2.67434288,
                -2.64338072,
                -2.62383385,
                -2.64847919,
                -2.60855089,
            ],
        )
        assert_array_equal(
            modflow_module.ssi.time,
            pd.DatetimeIndex(
                [
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-03",
                    "2022-01-04",
                    "2022-01-05",
                    "2022-01-06",
                    "2022-01-07",
                ]
            ),
        )

    @pytest.mark.unittest
    def test_load_pssi_with_presets(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        presets: Presets,
    ):
        modflow_module.measure = "pssi"
        modflow_module._load_ssi(parcel, model_settings, presets)
        assert isinstance(modflow_module.ssi, components.SsiMeasure)
        assert modflow_module.ssi.drain_depth == -2.7
        assert modflow_module.ssi.drain_distance == 1
        assert_array_almost_equal(
            modflow_module.ssi.drain_stage,
            [
                -2.30550536,
                -2.37214942,
                -2.37434288,
                -2.34338072,
                -2.32383385,
                -2.34847919,
                -2.30855089,
            ],
        )
        assert_array_equal(
            modflow_module.ssi.time,
            pd.DatetimeIndex(
                [
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-03",
                    "2022-01-04",
                    "2022-01-05",
                    "2022-01-06",
                    "2022-01-07",
                ]
            ),
        )

    @pytest.mark.unittest
    def test_initialize_recharge_ref(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: GroundwaterData,
    ):
        settings = model_settings.model_copy(update={"add_trenches": True})
        modflow_module.initialize(parcel, settings, lhm=lhm_data)
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert isinstance(modflow_module.recharge, components.ModflowInputSeries)
        assert isinstance(modflow_module.aquifer, components.ModflowInputSeries)
        assert isinstance(modflow_module.ditches, components.Ditches)
        assert isinstance(modflow_module.trenches, components.Trenches)
        assert modflow_module.ssi is None
        assert modflow_module.precipitation is None
        assert modflow_module.evapotranspiration is None

    @pytest.mark.parametrize(
        "modflow_module",
        [Params(measure="ssi"), Params(measure="pssi")],
        ids=["ssi", "pssi"],
        indirect=True,
    )
    def test_initialize_recharge_ssi_pssi(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: GroundwaterData,
    ):
        settings = model_settings.model_copy(update={"add_trenches": True})
        modflow_module.initialize(parcel, settings, lhm=lhm_data)
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert isinstance(modflow_module.recharge, components.ModflowInputSeries)
        assert isinstance(modflow_module.aquifer, components.ModflowInputSeries)
        assert isinstance(modflow_module.ditches, components.Ditches)
        assert isinstance(modflow_module.trenches, components.Trenches)
        assert isinstance(modflow_module.ssi, components.SsiMeasure)
        assert modflow_module.precipitation is None
        assert modflow_module.evapotranspiration is None

    @pytest.mark.parametrize(
        "modflow_module",
        [Params(gw_recharge_method="precip_evap")],
        ids=["precip_evap"],
        indirect=True,
    )
    def test_initialize_precip_evap_ref(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: GroundwaterData,
        weather_data: WeatherData,
    ):
        settings = model_settings.model_copy(update={"add_trenches": True})
        modflow_module.initialize(parcel, settings, lhm=lhm_data, weather=weather_data)
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert isinstance(modflow_module.precipitation, components.ModflowInputSeries)
        assert isinstance(
            modflow_module.evapotranspiration, components.ModflowInputSeries
        )
        assert isinstance(modflow_module.aquifer, components.ModflowInputSeries)
        assert isinstance(modflow_module.ditches, components.Ditches)
        assert isinstance(modflow_module.trenches, components.Trenches)
        assert modflow_module.recharge is None
        assert modflow_module.ssi is None

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="precip_evap", measure="ssi"),
            Params(gw_recharge_method="precip_evap", measure="pssi"),
        ],
        ids=["ssi", "pssi"],
        indirect=True,
    )
    def test_initialize_precip_evap_ssi_pssi(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: GroundwaterData,
        weather_data: WeatherData,
    ):
        settings = model_settings.model_copy(update={"add_trenches": True})
        modflow_module.initialize(parcel, settings, lhm=lhm_data, weather=weather_data)
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert isinstance(modflow_module.precipitation, components.ModflowInputSeries)
        assert isinstance(
            modflow_module.evapotranspiration, components.ModflowInputSeries
        )
        assert isinstance(modflow_module.aquifer, components.ModflowInputSeries)
        assert isinstance(modflow_module.ditches, components.Ditches)
        assert isinstance(modflow_module.trenches, components.Trenches)
        assert modflow_module.recharge is None
        assert isinstance(modflow_module.ssi, components.SsiMeasure)

    @pytest.mark.unittest
    def test_initialize_with_presets_recharge_ref(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: GroundwaterData,
        presets: Presets,
    ):
        modflow_module.initialize(parcel, model_settings, lhm=lhm_data, presets=presets)
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert isinstance(modflow_module.recharge, components.ModflowInputSeries)
        assert isinstance(modflow_module.aquifer, components.ModflowInputSeries)
        assert isinstance(modflow_module.ditches, components.Ditches)
        assert modflow_module.ssi is None
        assert modflow_module.trenches is None
        assert modflow_module.precipitation is None
        assert modflow_module.evapotranspiration is None

    @pytest.mark.parametrize(
        "modflow_module",
        [Params(measure="ssi"), Params(measure="pssi")],
        ids=["ssi", "pssi"],
        indirect=True,
    )
    def test_initialize_with_presets_recharge_ssi_pssi(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: GroundwaterData,
        presets: Presets,
    ):
        modflow_module.initialize(parcel, model_settings, lhm=lhm_data, presets=presets)
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert isinstance(modflow_module.recharge, components.ModflowInputSeries)
        assert isinstance(modflow_module.aquifer, components.ModflowInputSeries)
        assert isinstance(modflow_module.ditches, components.Ditches)
        assert isinstance(modflow_module.ssi, components.SsiMeasure)
        assert modflow_module.trenches is None
        assert modflow_module.precipitation is None
        assert modflow_module.evapotranspiration is None

    @pytest.mark.parametrize(
        "modflow_module",
        [Params(gw_recharge_method="precip_evap")],
        ids=["precip_evap"],
        indirect=True,
    )
    def test_initialize_with_presets_precip_evap_ref(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: GroundwaterData,
        weather_data: WeatherData,
        presets: Presets,
    ):
        modflow_module.initialize(
            parcel, model_settings, lhm=lhm_data, weather=weather_data, presets=presets
        )
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert isinstance(modflow_module.precipitation, components.ModflowInputSeries)
        assert isinstance(
            modflow_module.evapotranspiration, components.ModflowInputSeries
        )
        assert isinstance(modflow_module.aquifer, components.ModflowInputSeries)
        assert isinstance(modflow_module.ditches, components.Ditches)
        assert modflow_module.recharge is None
        assert modflow_module.trenches is None
        assert modflow_module.ssi is None

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="precip_evap", measure="ssi"),
            Params(gw_recharge_method="precip_evap", measure="pssi"),
        ],
        ids=["ssi", "pssi"],
        indirect=True,
    )
    def test_initialize_with_presets_precip_evap_ssi_pssi(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: GroundwaterData,
        weather_data: WeatherData,
        presets: Presets,
    ):
        modflow_module.initialize(
            parcel, model_settings, lhm=lhm_data, weather=weather_data, presets=presets
        )
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert isinstance(modflow_module.precipitation, components.ModflowInputSeries)
        assert isinstance(
            modflow_module.evapotranspiration, components.ModflowInputSeries
        )
        assert isinstance(modflow_module.aquifer, components.ModflowInputSeries)
        assert isinstance(modflow_module.ditches, components.Ditches)
        assert isinstance(modflow_module.ssi, components.SsiMeasure)
        assert modflow_module.trenches is None
        assert modflow_module.recharge is None

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="recharge", measure="ref"),
            Params(gw_recharge_method="recharge", measure="ssi"),
            Params(gw_recharge_method="recharge", measure="pssi"),
        ],
        ids=["recharge-ref", "recharge-ssi", "recharge-pssi"],
        indirect=True,
    )
    def test_create_modflow_model_recharge(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        settings_with_trenches: ModelSettings,
    ):
        model = initialized_modflow_module.create_modflow_model(
            parcel, settings_with_trenches, "simple"
        )
        assert isinstance(model, ModflowModel)
        assert not model.save_flows
        assert model.output_dir_runs.stem == "runs"
        assert model.output_dir_runs.parent.stem == "A_hVb"
        assert model.working_dir.stem == "modelfiles"
        assert model.working_dir.parent.stem == "A_hVb"
        assert model.start == settings_with_trenches.start_date - pd.Timedelta(days=1)
        assert model.end == settings_with_trenches.end_date + pd.Timedelta(days=1)
        assert_array_equal(
            model.time, settings_with_trenches.date_range.insert(0, model.start)
        )
        assert np.all(model.duration == 1)
        assert model.parcel_width == 2
        assert model.surface == -2.0
        assert model.nlayers == 35
        assert_array_equal(model.dz, np.repeat([0.05, 0.5, 1.0], [24, 10, 1]))
        assert_array_almost_equal(
            model.bottom,
            [
                -2.05,
                -2.1,
                -2.15,
                -2.2,
                -2.25,
                -2.3,
                -2.35,
                -2.4,
                -2.45,
                -2.5,
                -2.55,
                -2.6,
                -2.65,
                -2.7,
                -2.75,
                -2.8,
                -2.85,
                -2.9,
                -2.95,
                -3.0,
                -3.05,
                -3.1,
                -3.15,
                -3.2,
                -3.7,
                -4.2,
                -4.7,
                -5.2,
                -5.7,
                -6.2,
                -6.7,
                -7.2,
                -7.7,
                -8.2,
                -9.2,
            ],
        )
        assert_array_almost_equal(
            model.top,
            [
                -2.0,
                -2.05,
                -2.1,
                -2.15,
                -2.2,
                -2.25,
                -2.3,
                -2.35,
                -2.4,
                -2.45,
                -2.5,
                -2.55,
                -2.6,
                -2.65,
                -2.7,
                -2.75,
                -2.8,
                -2.85,
                -2.9,
                -2.95,
                -3.0,
                -3.05,
                -3.1,
                -3.15,
                -3.2,
                -3.7,
                -4.2,
                -4.7,
                -5.2,
                -5.7,
                -6.2,
                -6.7,
                -7.2,
                -7.7,
                -8.2,
            ],
        )
        assert_array_almost_equal(
            model.z,
            [
                -2.025,
                -2.075,
                -2.125,
                -2.175,
                -2.225,
                -2.275,
                -2.325,
                -2.375,
                -2.425,
                -2.475,
                -2.525,
                -2.575,
                -2.625,
                -2.675,
                -2.725,
                -2.775,
                -2.825,
                -2.875,
                -2.925,
                -2.975,
                -3.025,
                -3.075,
                -3.125,
                -3.175,
                -3.45,
                -3.95,
                -4.45,
                -4.95,
                -5.45,
                -5.95,
                -6.45,
                -6.95,
                -7.45,
                -7.95,
                -8.7,
            ],
        )
        assert_array_equal(model.x, [0.25, 0.75, 1.25, 1.75])
        assert model.ncol == 4
        assert model.dx == 0.5
        assert model.dy == 1
        assert_array_equal(
            model.vertical_index,
            [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
            ],
        )
        assert isinstance(model.sim, flopy.mf6.MFSimulation)
        assert isinstance(model.tdis, flopy.mf6.ModflowTdis)
        assert isinstance(model.solver, flopy.mf6.ModflowIms)
        assert isinstance(model.gwf, flopy.mf6.ModflowGwf)
        assert isinstance(model.dis, flopy.mf6.ModflowGwfdis)
        assert isinstance(model.ic, flopy.mf6.ModflowGwfic)
        assert isinstance(model.oc, flopy.mf6.ModflowGwfoc)
        assert model.kh is None
        assert model.kh_over_kv is None
        assert_array_almost_equal(
            model.recharge,
            [
                0.00048,
                0.0004812,
                0.002929,
                0.00003808,
                0.0003653,
                0.003971,
                0.0009328,
                0.001742,
            ],
        )
        assert model.evapotranspiration is None
        assert model.ditch_stage is None
        assert model.aquifer_chd is None
        assert isinstance(model.aquifer_wel, flopy.mf6.ModflowGwfwel)
        assert isinstance(model.riv, flopy.mf6.ModflowGwfriv)
        assert isinstance(model.riv_drn, flopy.mf6.ModflowGwfdrn)
        assert isinstance(model.trn, flopy.mf6.ModflowGwfdrn)
        assert model.ssi is None
        assert model.wel is None
        assert model.npf is None
        assert isinstance(model.rch, flopy.mf6.ModflowGwfrcha)
        assert model.sto is None
        assert model.evt is None
        assert model.head is None
        assert model.budgets is None

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="precip_evap", measure="ref"),
            Params(gw_recharge_method="precip_evap", measure="ssi"),
            Params(gw_recharge_method="precip_evap", measure="pssi"),
        ],
        ids=["precip_evap-ref", "precip_evap-ssi", "precip_evap-pssi"],
        indirect=True,
    )
    def test_create_modflow_model_precip_evap(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        settings_with_trenches: ModelSettings,
    ):
        model = initialized_modflow_module.create_modflow_model(
            parcel, settings_with_trenches, "simple"
        )
        assert isinstance(model, ModflowModel)
        assert not model.save_flows
        assert model.output_dir_runs.stem == "runs"
        assert model.output_dir_runs.parent.stem == "A_hVb"
        assert model.working_dir.stem == "modelfiles"
        assert model.working_dir.parent.stem == "A_hVb"
        assert model.start == settings_with_trenches.start_date - pd.Timedelta(days=1)
        assert model.end == settings_with_trenches.end_date + pd.Timedelta(days=1)
        assert_array_equal(
            model.time, settings_with_trenches.date_range.insert(0, model.start)
        )
        assert np.all(model.duration == 1)
        assert model.parcel_width == 2
        assert model.surface == -2.0
        assert model.nlayers == 35
        assert_array_equal(model.dz, np.repeat([0.05, 0.5, 1.0], [24, 10, 1]))
        assert_array_almost_equal(
            model.bottom,
            [
                -2.05,
                -2.1,
                -2.15,
                -2.2,
                -2.25,
                -2.3,
                -2.35,
                -2.4,
                -2.45,
                -2.5,
                -2.55,
                -2.6,
                -2.65,
                -2.7,
                -2.75,
                -2.8,
                -2.85,
                -2.9,
                -2.95,
                -3.0,
                -3.05,
                -3.1,
                -3.15,
                -3.2,
                -3.7,
                -4.2,
                -4.7,
                -5.2,
                -5.7,
                -6.2,
                -6.7,
                -7.2,
                -7.7,
                -8.2,
                -9.2,
            ],
        )
        assert_array_almost_equal(
            model.top,
            [
                -2.0,
                -2.05,
                -2.1,
                -2.15,
                -2.2,
                -2.25,
                -2.3,
                -2.35,
                -2.4,
                -2.45,
                -2.5,
                -2.55,
                -2.6,
                -2.65,
                -2.7,
                -2.75,
                -2.8,
                -2.85,
                -2.9,
                -2.95,
                -3.0,
                -3.05,
                -3.1,
                -3.15,
                -3.2,
                -3.7,
                -4.2,
                -4.7,
                -5.2,
                -5.7,
                -6.2,
                -6.7,
                -7.2,
                -7.7,
                -8.2,
            ],
        )
        assert_array_almost_equal(
            model.z,
            [
                -2.025,
                -2.075,
                -2.125,
                -2.175,
                -2.225,
                -2.275,
                -2.325,
                -2.375,
                -2.425,
                -2.475,
                -2.525,
                -2.575,
                -2.625,
                -2.675,
                -2.725,
                -2.775,
                -2.825,
                -2.875,
                -2.925,
                -2.975,
                -3.025,
                -3.075,
                -3.125,
                -3.175,
                -3.45,
                -3.95,
                -4.45,
                -4.95,
                -5.45,
                -5.95,
                -6.45,
                -6.95,
                -7.45,
                -7.95,
                -8.7,
            ],
        )
        assert_array_equal(model.x, [0.25, 0.75, 1.25, 1.75])
        assert model.ncol == 4
        assert model.dx == 0.5
        assert model.dy == 1
        assert_array_equal(
            model.vertical_index,
            [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
            ],
        )
        assert isinstance(model.sim, flopy.mf6.MFSimulation)
        assert isinstance(model.tdis, flopy.mf6.ModflowTdis)
        assert isinstance(model.solver, flopy.mf6.ModflowIms)
        assert isinstance(model.gwf, flopy.mf6.ModflowGwf)
        assert isinstance(model.dis, flopy.mf6.ModflowGwfdis)
        assert isinstance(model.ic, flopy.mf6.ModflowGwfic)
        assert isinstance(model.oc, flopy.mf6.ModflowGwfoc)
        assert model.kh is None
        assert model.kh_over_kv is None
        assert_array_almost_equal(
            model.recharge,
            [0.00214463, 0.0003, 0.0132, 0.0000025, 0.0003, 0.0054, 0.0000025, 0.0033],
        )
        assert_array_almost_equal(
            model.evapotranspiration,
            [0.00028525, 0.0003, 0.0003, 0.0003, 0.0002, 0.0002, 0.0004, 0.0002],
        )
        assert model.ditch_stage is None
        assert model.aquifer_chd is None
        assert isinstance(model.aquifer_wel, flopy.mf6.ModflowGwfwel)
        assert isinstance(model.riv, flopy.mf6.ModflowGwfriv)
        assert isinstance(model.riv_drn, flopy.mf6.ModflowGwfdrn)
        assert isinstance(model.trn, flopy.mf6.ModflowGwfdrn)
        assert model.ssi is None
        assert model.wel is None
        assert model.npf is None
        assert isinstance(model.rch, flopy.mf6.ModflowGwfrcha)
        assert model.sto is None
        assert isinstance(model.evt, flopy.mf6.ModflowGwfevt)
        assert model.head is None
        assert model.budgets is None

    @pytest.mark.skipif(
        not sys.platform.startswith("win"),
        reason="Can only run on Windows with .exe for now",
    )
    @pytest.mark.unittest
    def test_run_recharge_ref(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        settings_with_trenches: ModelSettings,
    ):
        ph = initialized_modflow_module.run(parcel, settings_with_trenches)
        assert isinstance(ph, xr.DataArray)
        assert_array_equal(ph["runs"], [1, 2])
        assert_array_equal(ph["time"], settings_with_trenches.date_range)
        assert_array_equal(ph["x"], [0.25, 0.75, 1.25, 1.75])
        assert_array_almost_equal(
            ph,
            [
                [
                    [-2.50107595, -2.50139723, -2.50139732, -2.50107601],
                    [-2.49710447, -2.49671107, -2.49671109, -2.49710448],
                    [-2.49965963, -2.49960781, -2.49960767, -2.49965964],
                    [-2.50035984, -2.50047761, -2.50047758, -2.50035987],
                    [-2.49490236, -2.49408216, -2.49408216, -2.49490236],
                    [-2.49695801, -2.49635534, -2.49635529, -2.49695801],
                    [-2.49668988, -2.49606285, -2.49606287, -2.49668996],
                ],
                [
                    [-2.50096249, -2.50136244, -2.50136252, -2.50096258],
                    [-2.49755454, -2.49725243, -2.49725245, -2.49755458],
                    [-2.49951376, -2.49944923, -2.49944918, -2.49951375],
                    [-2.50011754, -2.50019712, -2.50019711, -2.50011753],
                    [-2.49538123, -2.49455722, -2.49455717, -2.4953812],
                    [-2.49691206, -2.4961987, -2.49619868, -2.49691204],
                    [-2.49661409, -2.49584181, -2.49584181, -2.4966141],
                ],
            ],
        )
        assert_array_equal(
            initialized_modflow_module.success_and_failures.success_simple, [0, 1]
        )
        assert not initialized_modflow_module.success_and_failures.failure_simple
        assert not initialized_modflow_module.success_and_failures.success_complex
        assert not initialized_modflow_module.success_and_failures.failure_complex

    @pytest.mark.skipif(
        not sys.platform.startswith("win"),
        reason="Can only run on Windows with .exe for now",
    )
    @pytest.mark.parametrize(
        "modflow_module",
        [Params(measure="ssi"), Params(measure="pssi")],
        ids=["recharge-ssi", "recharge-pssi"],
        indirect=True,
    )
    def test_run_recharge_ssi_pssi(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        settings_with_trenches: ModelSettings,
    ):
        ph = initialized_modflow_module.run(parcel, settings_with_trenches)
        assert isinstance(ph, xr.DataArray)
        assert_array_equal(ph["runs"], [1, 2])
        assert_array_equal(ph["time"], settings_with_trenches.date_range)
        assert_array_equal(ph["x"], [0.25, 0.75, 1.25, 1.75])
        assert_array_almost_equal(
            ph,
            [
                [
                    [-2.44465405, -2.42978389, -2.42161424, -2.43481827],
                    [-2.44046105, -2.42542947, -2.41719975, -2.43079498],
                    [-2.44330441, -2.42838677, -2.42023568, -2.43356738],
                    [-2.44413824, -2.42925074, -2.42110156, -2.43435125],
                    [-2.43838415, -2.42277672, -2.41502787, -2.42881557],
                    [-2.44068185, -2.42566431, -2.41750342, -2.43107701],
                    [-2.44045865, -2.42543322, -2.41725941, -2.4308548],
                ],
                [
                    [-2.44762842, -2.42963348, -2.41996755, -2.43643021],
                    [-2.44383633, -2.42565944, -2.41590291, -2.43278434],
                    [-2.44611177, -2.42804689, -2.41839448, -2.43502253],
                    [-2.44689497, -2.42886709, -2.41922723, -2.43576622],
                    [-2.44176897, -2.42292092, -2.41371496, -2.43082283],
                    [-2.44348649, -2.42485539, -2.41562306, -2.43253649],
                    [-2.44323638, -2.4245773, -2.41535371, -2.43229749],
                ],
            ],
        )
        assert_array_equal(
            initialized_modflow_module.success_and_failures.success_simple, [0, 1]
        )
        assert not initialized_modflow_module.success_and_failures.failure_simple
        assert not initialized_modflow_module.success_and_failures.success_complex
        assert not initialized_modflow_module.success_and_failures.failure_complex

    @pytest.mark.parametrize(
        "modflow_module",
        [Params(gw_recharge_method="precip_evap", measure="ref")],
        ids=["precip_evap-ref"],
        indirect=True,
    )
    def test_run_precip_evap_ref(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        settings_with_trenches: ModelSettings,
    ):
        ph = initialized_modflow_module.run(parcel, settings_with_trenches)
        assert isinstance(ph, xr.DataArray)
        assert_array_equal(ph["runs"], [1, 2])
        assert_array_equal(ph["time"], settings_with_trenches.date_range)
        assert_array_equal(ph["x"], [0.25, 0.75, 1.25, 1.75])
        assert_array_almost_equal(
            ph,
            [
                [
                    [-2.49911341, -2.49889499, -2.49889462, -2.49911333],
                    [-2.47900918, -2.47395299, -2.47395285, -2.47900912],
                    [-2.49053326, -2.4884705, -2.48846945, -2.49053321],
                    [-2.49591953, -2.49491807, -2.49491736, -2.49591942],
                    [-2.49026438, -2.4884665, -2.48846604, -2.49026435],
                    [-2.4966863, -2.49588115, -2.49588056, -2.49668625],
                    [-2.49422469, -2.49312767, -2.4931275, -2.49422467],
                ],
                [
                    [-2.49848065, -2.49802573, -2.49802536, -2.49848031],
                    [-2.48105207, -2.47732623, -2.47732613, -2.48105194],
                    [-2.49013207, -2.48751648, -2.48751614, -2.49013201],
                    [-2.49485517, -2.49328968, -2.49328929, -2.49485512],
                    [-2.49020037, -2.48794201, -2.48794167, -2.49020036],
                    [-2.49570492, -2.49441937, -2.494419, -2.49570487],
                    [-2.49382813, -2.49234395, -2.49234386, -2.49382809],
                ],
            ],
        )
        assert_array_equal(
            initialized_modflow_module.success_and_failures.success_simple, [0, 1]
        )
        assert not initialized_modflow_module.success_and_failures.failure_simple
        assert not initialized_modflow_module.success_and_failures.success_complex
        assert not initialized_modflow_module.success_and_failures.failure_complex

    @pytest.mark.parametrize(
        "modflow_module",
        [
            Params(gw_recharge_method="precip_evap", measure="ssi"),
            Params(gw_recharge_method="precip_evap", measure="pssi"),
        ],
        ids=["precip_evap-ssi", "precip_evap-pssi"],
        indirect=True,
    )
    def test_run_precip_evap_ssi_pssi(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        settings_with_trenches: ModelSettings,
    ):
        ph = initialized_modflow_module.run(parcel, settings_with_trenches)
        assert isinstance(ph, xr.DataArray)
        assert_array_equal(ph["runs"], [1, 2])
        assert_array_equal(ph["time"], settings_with_trenches.date_range)
        assert_array_equal(ph["x"], [0.25, 0.75, 1.25, 1.75])
        assert_array_almost_equal(
            ph,
            [
                [
                    [-2.44305065, -2.42812788, -2.41999425, -2.43335067],
                    [-2.4207954, -2.40597247, -2.3996968, -2.41258681],
                    [-2.43502873, -2.41899526, -2.41129606, -2.42455926],
                    [-2.44038952, -2.42498675, -2.41721667, -2.43089842],
                    [-2.43439634, -2.41837451, -2.41090837, -2.42372038],
                    [-2.44095901, -2.42596455, -2.41786147, -2.43142858],
                    [-2.43827138, -2.42265922, -2.41497076, -2.42877587],
                ],
                [
                    [-2.44537456, -2.4272835, -2.41766375, -2.43437464],
                    [-2.42449045, -2.40686912, -2.39956099, -2.41506174],
                    [-2.43690833, -2.41738584, -2.40819104, -2.42477969],
                    [-2.44191633, -2.42306762, -2.41396387, -2.43121095],
                    [-2.43695863, -2.41749747, -2.40866989, -2.42453964],
                    [-2.44275383, -2.42403506, -2.41494798, -2.43196184],
                    [-2.44070648, -2.421731, -2.41269011, -2.42992241],
                ],
            ],
        )
        assert_array_equal(
            initialized_modflow_module.success_and_failures.success_simple, [0, 1]
        )
        assert not initialized_modflow_module.success_and_failures.failure_simple
        assert not initialized_modflow_module.success_and_failures.success_complex
        assert not initialized_modflow_module.success_and_failures.failure_complex

    @pytest.mark.skipif(
        not sys.platform.startswith("win"),
        reason="Can only run on Windows with .exe for now",
    )
    @pytest.mark.unittest
    def test_run_with_presets(
        self,
        initialized_modflow_with_presets: Modflow,
        parcel: Parcel,
        settings_with_trenches: ModelSettings,
    ):
        """
        NOTE: We only test the setup "recharge-ref" here because `Presets` will change
        the results of the other setups but not because of differences in the implementation
        of the Modflow module itself. Differences are due to differences in input values.
        """
        ph = initialized_modflow_with_presets.run(parcel, settings_with_trenches)
        assert isinstance(ph, xr.DataArray)
        assert_array_equal(ph["runs"], [1, 2])
        assert_array_equal(ph["time"], settings_with_trenches.date_range)
        assert_array_equal(ph["x"], [0.25, 0.75, 1.25, 1.75])
        assert_array_almost_equal(
            ph,
            [
                [
                    [-2.63981132, -2.63999452, -2.63999459, -2.63981135],
                    [-2.63669187, -2.63613402, -2.6361335, -2.63669203],
                    [-2.63895087, -2.63880254, -2.63880227, -2.6389509],
                    [-2.63952317, -2.6395713, -2.63957123, -2.63952316],
                    [-2.63519668, -2.63426225, -2.63426189, -2.63519672],
                    [-2.63709114, -2.63643668, -2.63643654, -2.63709115],
                    [-2.63690523, -2.63624388, -2.63624386, -2.63690522],
                ],
                [
                    [-2.63968697, -2.63989138, -2.63989138, -2.63968697],
                    [-2.63688741, -2.636326, -2.63632567, -2.63688748],
                    [-2.63872641, -2.63848821, -2.63848807, -2.6387264],
                    [-2.63927027, -2.63923751, -2.63923745, -2.63927026],
                    [-2.63543816, -2.63441977, -2.63441955, -2.63543819],
                    [-2.6369065, -2.63606137, -2.63606131, -2.63690649],
                    [-2.6367221, -2.63584732, -2.63584731, -2.63672209],
                ],
            ],
        )
        assert_array_equal(
            initialized_modflow_with_presets.success_and_failures.success_simple,
            [0, 1],
        )
        assert not initialized_modflow_with_presets.success_and_failures.failure_simple
        assert not initialized_modflow_with_presets.success_and_failures.success_complex
        assert not initialized_modflow_with_presets.success_and_failures.failure_complex

    @pytest.mark.unittest
    def test_success_and_failures_before_runs(
        self, initialized_modflow_module: Modflow
    ):
        with pytest.raises(ValueError, match="ModflowModel has not been run yet."):
            initialized_modflow_module.success_and_failures

    @pytest.mark.skipif(
        not sys.platform.startswith("win"),
        reason="Can only run on Windows with .exe for now",
    )
    @pytest.mark.unittest
    def test_run_1d(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        settings_with_trenches: ModelSettings,
    ):
        settings = settings_with_trenches.model_copy(update={"dimension": "1D"})
        with pytest.raises(
            NotImplementedError, match="1D Modflow model not implemented"
        ):
            initialized_modflow_module.run(parcel, settings)

    @pytest.mark.skipif(
        not sys.platform.startswith("win"),
        reason="Can only run on Windows with .exe for now",
    )
    @pytest.mark.integration
    def test_initialize_and_run_modflow(
        self,
        modflow_parameters: pd.DataFrame,
        modflow_executable: str,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: GroundwaterData,
    ):
        mf = Modflow(
            parameters=modflow_parameters, modflow_executable=modflow_executable
        )
        mf.initialize(parcel, model_settings, lhm=lhm_data)
        ph = mf.run(parcel, model_settings)
        assert isinstance(ph, xr.DataArray)
        # TODO: make sure a run with a "COMPLEX" modflow model is used and test result

    @pytest.mark.unittest
    def test_reset(self, modflow_module, initialized_modflow_module):
        """
        Test the reset method of the Modflow module from initial state and after running.

        """
        modflow_module.reset()
        assert modflow_module._discretization is None
        assert modflow_module._recharge is None
        assert modflow_module._aquifer is None
        assert modflow_module._ditches is None
        assert modflow_module._trenches is None
        assert modflow_module._ssi is None
        assert modflow_module._success_and_failures is None

        initialized_modflow_module.reset()
        assert initialized_modflow_module._discretization is None
        assert initialized_modflow_module._recharge is None
        assert initialized_modflow_module._aquifer is None
        assert initialized_modflow_module._ditches is None
        assert initialized_modflow_module._trenches is None
        assert initialized_modflow_module._ssi is None
        assert initialized_modflow_module._success_and_failures is None
