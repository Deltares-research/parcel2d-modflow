import re
import sys
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from parcel2d_modflow import components
from parcel2d_modflow.base import ModelSettings, Parcel
from parcel2d_modflow.mf.model import ModflowModel
from parcel2d_modflow.mf.module import Modflow
from parcel2d_modflow.modeldata import LhmData, Presets


@pytest.fixture
def start_date():
    return pd.to_datetime("01-01-2022", format="%d-%m-%Y")


@pytest.fixture
def end_date():
    return pd.to_datetime("31-12-2022", format="%d-%m-%Y")


@pytest.fixture
def modflow_module(modflow_parameters: pd.DataFrame, modflow_executable: str):
    """
    Empty (not-initialized) `Modflow` module with "flux" method and "ssi" measure.

    """
    modflow_parameters["entry_drain_resistance (d)"] = 1.0
    return Modflow(modflow_parameters, "flux", modflow_executable, "ssi")


@pytest.fixture
def initialized_modflow_module(
    modflow_parameters: pd.DataFrame, modflow_executable: str
):
    """
    Initialized `Modflow` module with "flux" method and "ssi" measure and containing all
    required components for a Modflow model run.

    """
    modflow_parameters["entry_drain_resistance (d)"] = 1.0
    mf = Modflow(modflow_parameters, "flux", modflow_executable, "ssi")
    mf.parameters.columns = [c.split(" ")[0] for c in mf.parameters.columns]
    mf._discretization = components.SubsurfaceStructure(
        np.array([0.2, 0.15, 0.35, 0.5, 0.5, 0.05, 1.0]),
        np.array([3, 2, 1, 1, 1, 4, 4]),
        np.array([1, 1, 1, 1, 1, 2, 2]),
        np.array([0.001, 2200.0]),
    )
    mf._recharge = components.Recharge(
        0.00048,
        np.array(
            [
                4.812e-04,
                2.929e-03,
                3.808e-05,
                3.653e-04,
                3.971e-03,
                9.328e-04,
                1.742e-03,
            ]
        ),
    )
    mf._aquifer = components.Aquifer(
        -0.000936,
        np.array(
            [
                -0.000936,
                -0.000951,
                -0.000949,
                -0.000910,
                -0.000929,
                -0.000924,
                -0.000922,
            ]
        ),
    )
    mf._ditches = components.Ditches(
        -2.8, 1.0, np.array([-2.5]), pd.DatetimeIndex(["2022-01-01"])
    )
    mf._ssi = components.SsiMeasure(
        -2.7, 1, np.array([-2.2]), pd.DatetimeIndex(["2022-01-01"])
    )
    mf._trenches = components.Trenches(-2.3, np.array([1.0]), 1.0)
    return mf


@pytest.fixture
def initialized_modflow_with_presets(
    modflow_module: Modflow,
    parcel: Parcel,
    model_settings: ModelSettings,
    lhm_data: LhmData,
    presets: Presets,
):
    """
    Initialized `Modflow` module with "flux" method and "ssi" measure and containing all
    required components for a Modflow model run, based on the fixture `Presets`.

    """
    settings = model_settings.model_copy(update={"add_trenches": True})
    modflow_module.initialize(parcel, settings, lhm_data, presets)
    return modflow_module


class TestModflow:
    @pytest.fixture
    def empty_presets(self):
        return Presets()

    @pytest.mark.unittest
    def test_initialize_module(
        self, modflow_parameters: pd.DataFrame, modflow_executable: str
    ):
        module = Modflow(modflow_parameters, "flux", modflow_executable)
        assert module.is_valid(module.name)
        assert isinstance(module, Modflow)
        assert isinstance(module.parameters, pd.DataFrame)
        assert isinstance(module.executable, Path)
        assert_array_equal(
            module.parameters.columns, ["runnr", "kh", "sy_peat", "sy_clay"]
        )
        assert module.aquifer_method == "flux"
        assert module.discretization is None
        assert module.recharge is None
        assert module.aquifer is None
        assert module.ditches is None
        assert module.trenches is None
        assert module.ssi is None
        assert module._success_and_failures is None

        expected_error = (
            "Entry drain resistance is required for the measure: ssi. "
            "Please add column 'entry_drain_resistance (d)' to the parameter file."
        )
        with pytest.raises(ValueError, match=re.escape(expected_error)):
            Modflow(modflow_parameters, "flux", modflow_executable, "ssi")

        modflow_parameters["entry_drain_resistance (d)"] = 1.0
        module = Modflow(modflow_parameters, "flux", modflow_executable, "ssi")
        assert isinstance(module, Modflow)

        expected_error = (
            "Aquifer method 'invalid_method' is not implemented for the Modflow module"
        )
        with pytest.raises(NotImplementedError, match=expected_error):
            Modflow(modflow_parameters, "invalid_method", modflow_executable)

        expected_error = "Measure 'invalid_measure' is not a valid measure. Valid measures are: {'ref', 'ssi', 'pssi'}"
        with pytest.raises(ValueError, match=expected_error):
            Modflow(modflow_parameters, "flux", modflow_executable, "invalid_measure")

    @pytest.mark.unittest
    def test_discretize_parcel(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        lhm_data: LhmData,
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

    @pytest.mark.unittest
    def test_discretize_parcel_with_presets(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        lhm_data: LhmData,
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

    @pytest.mark.unittest
    def test_load_recharge(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        lhm_data: LhmData,
        model_settings: ModelSettings,
        empty_presets: Presets,
    ):
        modflow_module._load_recharge(parcel, lhm_data, model_settings, empty_presets)

        assert isinstance(modflow_module.recharge, components.Recharge)
        assert modflow_module.recharge.start == 0.00048118845
        assert isinstance(modflow_module.recharge.series, np.ndarray)
        assert len(modflow_module.recharge.series) == 7

    @pytest.mark.unittest
    def test_load_recharge_with_presets(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        lhm_data: LhmData,
        model_settings: ModelSettings,
        presets: Presets,
    ):
        modflow_module._load_recharge(parcel, lhm_data, model_settings, presets)

        assert isinstance(modflow_module.recharge, components.Recharge)
        assert np.isclose(modflow_module.recharge.start, 0.00149386)
        assert isinstance(modflow_module.recharge.series, np.ndarray)
        assert len(modflow_module.recharge.series) == 7

    @pytest.mark.unittest
    def test_load_flux(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        lhm_data: LhmData,
        model_settings: ModelSettings,
        empty_presets: Presets,
    ):
        modflow_module._load_aquifer(parcel, lhm_data, model_settings, empty_presets)

        assert isinstance(modflow_module.aquifer, components.Aquifer)
        assert modflow_module.aquifer.start == -0.000936158816
        assert isinstance(modflow_module.aquifer.series, np.ndarray)
        assert len(modflow_module.aquifer.series) == 7

    @pytest.mark.unittest
    def test_load_flux_with_presets(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        lhm_data: LhmData,
        model_settings: ModelSettings,
        presets: Presets,
    ):
        modflow_module._load_aquifer(parcel, lhm_data, model_settings, presets)

        assert isinstance(modflow_module.aquifer, components.Aquifer)
        assert np.isclose(modflow_module.aquifer.start, -0.000931428)
        assert isinstance(modflow_module.aquifer.series, np.ndarray)
        assert len(modflow_module.aquifer.series) == 7

    @pytest.mark.unittest
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

    @pytest.mark.unittest
    def test_load_ditches_with_presets(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        presets: Presets,
    ):
        modflow_module._load_ditches(parcel, model_settings, presets)

        assert isinstance(modflow_module.ditches, components.Ditches)
        assert np.isclose(modflow_module.ditches.bottom, -2.9099, atol=1e-4)
        assert modflow_module.ditches.resistance == 1.0
        assert_array_almost_equal(modflow_module.ditches.stage, [-2.5014], decimal=4)
        assert_array_equal(
            modflow_module.ditches.dates,
            pd.DatetimeIndex(["2022-01-01"]),
        )

    @pytest.mark.unittest
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
        assert modflow_module.ssi.drain_depth == -2.71
        assert modflow_module.ssi.drain_distance == 1
        assert_array_almost_equal(
            modflow_module.ssi.drain_stage,
            [-2.51, -2.51, -2.51, -2.5, -2.5, -2.49, -2.49],
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
        assert modflow_module.ssi.drain_depth == -2.71
        assert modflow_module.ssi.drain_distance == 1
        assert_array_almost_equal(
            modflow_module.ssi.drain_stage,
            [-2.51, -2.51, -2.50, -2.5, -2.5, -2.49, -2.49],
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
    def test_initialize(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: LhmData,
    ):
        settings = model_settings.model_copy(update={"add_trenches": True})
        modflow_module.initialize(parcel, settings, lhm_data)
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert_array_equal(modflow_module.discretization.kvalues, [0.01, 2200.0])
        assert isinstance(modflow_module.recharge, components.Recharge)
        assert modflow_module.recharge.start == 0.00048118845
        assert isinstance(modflow_module.aquifer, components.Aquifer)
        assert modflow_module.aquifer.start == -0.000936158816
        assert isinstance(modflow_module.ditches, components.Ditches)
        assert modflow_module.ditches.bottom == -2.8
        assert isinstance(modflow_module.ssi, components.SsiMeasure)
        assert modflow_module.ssi.drain_depth == -2.7
        assert isinstance(modflow_module.trenches, components.Trenches)
        assert modflow_module.trenches.depth == -2.3

    @pytest.mark.unittest
    def test_initialize_with_presets(
        self,
        modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
        lhm_data: LhmData,
        presets: Presets,
    ):
        modflow_module.initialize(parcel, model_settings, lhm_data, presets)
        assert isinstance(modflow_module.discretization, components.SubsurfaceStructure)
        assert_array_equal(modflow_module.discretization.kvalues, [0.0001, 2200.0])
        assert isinstance(modflow_module.recharge, components.Recharge)
        assert np.isclose(modflow_module.recharge.start, 0.00149386)
        assert isinstance(modflow_module.aquifer, components.Aquifer)
        assert np.isclose(modflow_module.aquifer.start, -0.000931428)
        assert isinstance(modflow_module.ditches, components.Ditches)
        assert_array_almost_equal(modflow_module.ditches.stage, [-2.5014], decimal=4)
        assert isinstance(modflow_module.ssi, components.SsiMeasure)
        assert_array_almost_equal(
            modflow_module.ssi.drain_stage,
            [-2.51, -2.51, -2.51, -2.5, -2.5, -2.49, -2.49],
        )
        assert modflow_module.trenches is None

    @pytest.mark.unittest
    def test_create_modflow_model(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
    ):
        model = initialized_modflow_module.create_modflow_model(
            parcel, model_settings, "simple"
        )
        assert isinstance(model, ModflowModel)
        assert not model.save_flows
        assert model.output_dir_runs.stem == "runs"
        assert model.output_dir_runs.parent.stem == "A_hVb"
        assert model.working_dir.stem == "modelfiles"
        assert model.working_dir.parent.stem == "A_hVb"
        assert model.start == model_settings.start_date - pd.Timedelta(days=1)
        assert model.end == model_settings.end_date + pd.Timedelta(days=1)
        assert_array_equal(model.time, model_settings.date_range.insert(0, model.start))
        assert np.all(model.duration == 1)
        assert model.parcel_width == 2
        assert model.surface == -2.0
        assert model.nlayers == 27
        assert_array_equal(model.dz, [0.05] * 24 + [0.5, 0.05, 1.0])
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
                -3.75,
                -4.75,
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
                -3.75,
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
                -3.725,
                -4.25,
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
            ],
        )
        assert isinstance(model.sim, flopy.mf6.MFSimulation)
        assert isinstance(model.tdis, flopy.mf6.ModflowTdis)
        assert isinstance(model.solver, flopy.mf6.ModflowIms)
        assert isinstance(model.gwf, flopy.mf6.ModflowGwf)
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
        assert model.head is None
        assert model.budgets is None

    @pytest.mark.skipif(
        not sys.platform.startswith("win"),
        reason="Can only run on Windows with .exe for now",
    )
    @pytest.mark.unittest
    def test_run_modflow_model(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
    ):
        result = np.full(
            (
                len(initialized_modflow_module.parameters),
                len(model_settings.date_range),
                len(parcel.discretization.xcol),
            ),
            np.nan,
        )
        model = initialized_modflow_module.create_modflow_model(
            parcel, model_settings, "SIMPLE"
        )
        result, succes, failure = initialized_modflow_module.run_modflow_model(
            model,
            initialized_modflow_module.parameters,
            result,
            model_settings.start_date,
        )
        assert isinstance(result, np.ndarray)
        assert not np.isnan(result).all()  # Result should not contain any NaN values
        assert_array_equal(succes, [0, 1])
        assert not failure  # Should be empty list

    @pytest.mark.skipif(
        not sys.platform.startswith("win"),
        reason="Can only run on Windows with .exe for now",
    )
    @pytest.mark.unittest
    def test_run(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
    ):
        initialized_modflow_module._ssi = None  # Ensure no SSI measure is set
        ph = initialized_modflow_module.run(parcel, model_settings)
        assert isinstance(ph, xr.DataArray)
        assert_array_equal(ph["runs"], [1, 2])
        assert_array_equal(ph["time"], model_settings.date_range)
        assert_array_equal(ph["x"], [0.25, 0.75, 1.25, 1.75])
        assert_array_almost_equal(
            ph.sel(runs=1),
            [
                [-2.5010772, -2.50139866, -2.50139873, -2.50107714],
                [-2.49710463, -2.49671128, -2.49671129, -2.49710462],
                [-2.49965962, -2.49960775, -2.49960762, -2.49965964],
                [-2.50035993, -2.50047768, -2.50047766, -2.50035993],
                [-2.49490066, -2.4940802, -2.49408019, -2.49490068],
                [-2.49695734, -2.49635451, -2.49635446, -2.49695735],
                [-2.49668932, -2.4960621, -2.49606213, -2.49668941],
            ],
        )
        assert_array_almost_equal(
            ph.sel(runs=2),
            [
                [-2.50096376, -2.50136418, -2.5013643, -2.50096378],
                [-2.49755466, -2.49725276, -2.49725277, -2.49755452],
                [-2.49951371, -2.49944918, -2.49944911, -2.49951367],
                [-2.50011765, -2.50019716, -2.50019724, -2.50011764],
                [-2.49537993, -2.49455562, -2.49455556, -2.49537994],
                [-2.49691154, -2.49619796, -2.4961979, -2.49691142],
                [-2.49661376, -2.4958413, -2.49584127, -2.49661374],
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
    def test_run_with_ssi(
        self,
        initialized_modflow_module: Modflow,
        parcel: Parcel,
        model_settings: ModelSettings,
    ):
        ph = initialized_modflow_module.run(parcel, model_settings)
        assert isinstance(ph, xr.DataArray)
        assert_array_equal(ph["runs"], [1, 2])
        assert_array_equal(ph["time"], model_settings.date_range)
        assert_array_equal(ph["x"], [0.25, 0.75, 1.25, 1.75])
        assert_array_almost_equal(
            ph.sel(runs=1),
            [
                [-2.4446654, -2.42979146, -2.42161579, -2.43481534],
                [-2.44047038, -2.4254349, -2.41719933, -2.43079038],
                [-2.44331319, -2.42839159, -2.42023458, -2.43356218],
                [-2.44414608, -2.42925448, -2.42109936, -2.43434478],
                [-2.43839044, -2.42277886, -2.41502444, -2.42880819],
                [-2.4406892, -2.42566771, -2.41750098, -2.43107061],
                [-2.44046672, -2.42543738, -2.41725782, -2.4308492],
            ],
        )
        assert_array_almost_equal(
            ph.sel(runs=2),
            [
                [-2.4476403, -2.42964139, -2.4199689, -2.43642708],
                [-2.44384635, -2.42566551, -2.41590274, -2.43277981],
                [-2.44612103, -2.42805215, -2.41839353, -2.43501726],
                [-2.44690372, -2.4288718, -2.41922566, -2.43576035],
                [-2.44177618, -2.42292396, -2.41371213, -2.43081588],
                [-2.44349476, -2.42485957, -2.41562128, -2.43253072],
                [-2.44324485, -2.42458173, -2.41535224, -2.43229181],
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
        model_settings: ModelSettings,
    ):
        ph = initialized_modflow_with_presets.run(parcel, model_settings)
        assert isinstance(ph, xr.DataArray)
        assert_array_equal(ph["runs"], [1, 2])
        assert_array_equal(ph["time"], model_settings.date_range)
        assert_array_equal(ph["x"], [0.25, 0.75, 1.25, 1.75])
        assert_array_almost_equal(
            ph.sel(runs=1),
            [
                [-2.50158, -2.50198728, -2.50241515, -2.50195879],
                [-2.49904451, -2.49906222, -2.49956854, -2.49945739],
                [-2.50152638, -2.50186895, -2.50228086, -2.5018599],
                [-2.50172063, -2.50186309, -2.5018308, -2.50173075],
                [-2.49692264, -2.49624231, -2.49631785, -2.49698296],
                [-2.49811521, -2.49720866, -2.49683015, -2.49783992],
                [-2.49755809, -2.49650543, -2.49608378, -2.49722363],
            ],
        )
        assert_array_almost_equal(
            ph.sel(runs=2),
            [
                [-2.50093972, -2.50127647, -2.50190664, -2.50141068],
                [-2.49887416, -2.49888316, -2.49951574, -2.4993537],
                [-2.5009328, -2.50121775, -2.50172959, -2.50131577],
                [-2.50136088, -2.50148415, -2.50150534, -2.50142637],
                [-2.49716827, -2.49644758, -2.49652112, -2.49723229],
                [-2.49800519, -2.49693957, -2.49651785, -2.49773455],
                [-2.49747761, -2.49620739, -2.49571642, -2.49712773],
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
        model_settings: ModelSettings,
    ):
        settings = model_settings.model_copy(update={"dimension": "1D"})
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
        lhm_data: LhmData,
    ):
        mf = Modflow(modflow_parameters, "flux", modflow_executable)
        mf.initialize(parcel, model_settings, lhm_data)
        ph = mf.run(parcel, model_settings)
        assert isinstance(ph, xr.DataArray)
        # TODO: make sure a run with a "COMPLEX" modflow model is used and test result

    @pytest.mark.parametrize("module", ["modflow_module", "initialized_modflow_module"])
    def test_reset(self, module, request):
        """
        Test the reset method of the Modflow module from initial state and after running.

        """
        module = request.getfixturevalue(module)
        module.reset()
        assert module._discretization is None
        assert module._recharge is None
        assert module._aquifer is None
        assert module._ditches is None
        assert module._trenches is None
        assert module._ssi is None
        assert module._success_and_failures is None
