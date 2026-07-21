import numpy as np
import pandas as pd
import pytest
from numpy.testing import (
    assert_approx_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from parcel2d_modflow import components
from parcel2d_modflow.base import Parcel
from parcel2d_modflow.config import ModelSettings


class TestParcel:
    @pytest.mark.unittest
    def test_parcel_init(self):
        parcel = Parcel("A", 1.0, 1.0, 2, -2.0)
        assert parcel.name == "A"
        assert parcel.x == 1.0
        assert parcel.y == 1.0
        assert parcel.width == 2
        assert parcel.surface_level == -2.0
        assert parcel.soilcode is None
        assert parcel.summer_stage is None
        assert parcel.winter_stage is None
        assert parcel.trench_depth is None
        assert parcel.trench_locations is None
        assert parcel.drain_depth is None
        assert parcel.drain_distance is None
        assert parcel.pssi_summer_stage is None
        assert parcel.pssi_winter_stage is None

    @pytest.mark.unittest
    def test_discretize_soildepth(
        self, empty_parcel: Parcel, model_settings: ModelSettings
    ):
        empty_parcel.discretize_soildepth(model_settings)
        assert isinstance(empty_parcel.discretization, components.SoilDiscretization)
        assert empty_parcel.discretization.nlayers == 24
        assert (
            empty_parcel.discretization.nlayers
            == len(empty_parcel.discretization.zmid)
            == len(empty_parcel.discretization.zbot)
        )
        assert_array_almost_equal(
            empty_parcel.discretization.zmid, np.arange(0.025, 1.2, 0.05)
        )
        assert_array_almost_equal(
            empty_parcel.discretization.zbot, np.arange(0.05, 1.25, 0.05)
        )
        assert_array_almost_equal(
            empty_parcel.discretization.xcol, np.arange(0.25, 20, 0.5)
        )

    @pytest.mark.unittest
    def test_load_ditches(self, parcel: Parcel):
        settings = ModelSettings(
            workdir=".",
            start_date=pd.to_datetime("2022-01-01"),
            end_date=pd.to_datetime("2022-12-31"),
        )

        ditches = parcel.load_ditches(settings.date_range, settings.winter_period)
        assert isinstance(ditches, components.Ditches)
        assert ditches.bottom == -2.8
        assert ditches.resistance == 1.0
        assert_array_equal(ditches.stage, [-2.5, -2.4, -2.5])
        assert_array_equal(
            ditches.dates, pd.DatetimeIndex(["2022-01-01", "2022-04-01", "2022-10-01"])
        )

        # Test that ValueError is raised when summer_stage or winter_stage is None
        parcel.summer_stage = None
        parcel.winter_stage = None

        expected_error = (
            "Cannot load ditch information for parcel. Summer and winter stage must "
            "be defined."
        )
        with pytest.raises(ValueError, match=expected_error):
            parcel.load_ditches(settings.date_range, settings.winter_period)

    @pytest.mark.unittest
    def test_ditch_water_depth(self, parcel: Parcel):
        ditch_depth = parcel.ditch_water_depth()
        assert ditch_depth == 0.4

        ditch_depth = parcel.ditch_water_depth(min_water_depth=0.1)
        assert_approx_equal(ditch_depth, 0.3)

        parcel.winter_stage = None
        ditch_depth = parcel.ditch_water_depth()
        assert ditch_depth == 0.4

    @pytest.mark.unittest
    def test_get_forganic(self, parcel: Parcel):
        forganic = parcel.get_forganic()
        assert_array_almost_equal(
            forganic,
            [
                0.35,
                0.35,
                0.35,
                0.35,
                0.25,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.7,
                0.7,
                0.7,
                0.7,
                0.7,
                0.7,
                0.7,
                0.7,
                0.7,
                0.7,
            ],
        )

        soilprofile = parcel.soilprofile.copy()
        parcel.soilprofile = None
        parcel._discretization = None

        with pytest.raises(TypeError, match="No soilprofile loaded for parcel."):
            parcel.get_forganic()  # Misses `self.soilprofile` and `self.discretization`

        parcel.soilprofile = soilprofile
        with pytest.raises(
            TypeError, match="Soilprofile not discretized yet for parcel."
        ):
            parcel.get_forganic()  # Misses `self.discretization`

    @pytest.mark.unittest
    def test_load_ssi_measure(self, parcel: Parcel):
        parcel.drain_depth = 0.7
        parcel.drain_distance = 4
        parcel.pssi_summer_stage = -2.1
        parcel.pssi_winter_stage = -2.2

        settings = ModelSettings(
            workdir=".",
            start_date=pd.to_datetime("2022-01-01"),
            end_date=pd.to_datetime("2022-12-31"),
        )

        ssi = parcel.load_ssi_measure(
            settings.date_range, settings.winter_period, settings.min_drain_depth
        )
        assert isinstance(ssi, components.SsiMeasure)
        assert ssi.drain_depth == -2.7
        assert ssi.drain_distance == 4
        assert_array_equal(ssi.drain_stage, [-2.2, -2.1, -2.2])
        assert_array_equal(
            ssi.time, pd.DatetimeIndex(["2022-01-01", "2022-04-01", "2022-10-01"])
        )

    @pytest.mark.unittest
    def test_load_trenches(self, parcel: Parcel):
        resistance = 1.0
        trenches = parcel.load_trenches(resistance)
        assert isinstance(trenches, components.Trenches)
        assert trenches.depth == -2.3
        assert trenches.resistance == resistance
        assert_array_equal(trenches.locations, [1.0])

    @pytest.mark.unittest
    def test_load_trenches_specified_distance(self, parcel: Parcel):
        resistance = 1.0
        distance = [0.5, 1.5]
        parcel.trench_locations = distance

        trenches = parcel.load_trenches(resistance)
        assert_array_equal(trenches.locations, distance)
