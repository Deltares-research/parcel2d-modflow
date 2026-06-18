from pathlib import Path

import pandas as pd
import pytest

from parcel2d_modflow.base import ModelSettings


class TestPresets:  # TODO: Move this to parcel2d-modflow
    """
    This class only tests error behaviour for loading of Preset data. The correct
    behaviour is testing model run components.

    """

    @pytest.fixture
    def settings_for_error(self):
        start_date = pd.to_datetime("2022-01-01")
        end_date = pd.to_datetime("2022-12-31")
        return ModelSettings(
            workdir=Path("."), start_date=start_date, end_date=end_date
        )

    @pytest.mark.unittest
    def test_load_recharge_with_error(self, presets, settings_for_error):
        expected_error = (
            f"{presets.__class__.__name__}.recharge does not have daily data for the "
            "required modelling period"
        )
        with pytest.raises(KeyError, match=expected_error):
            presets.load_recharge(settings_for_error)

    @pytest.mark.unittest
    def test_load_flux_with_error(self, presets, settings_for_error):
        expected_error = (
            f"{presets.__class__.__name__}.aquifer_flux does not have daily data for the "
            "required modelling period"
        )
        with pytest.raises(KeyError, match=expected_error):
            presets.load_aquifer_flux(settings_for_error)

    @pytest.mark.unittest
    def test_load_ditches_with_error(self, presets, settings_for_error):
        expected_error = (
            f"{presets.__class__.__name__}.ditch_stage does not have daily data for the "
            "required modelling period"
        )
        surface_level = -2.0
        with pytest.raises(KeyError, match=expected_error):
            presets.load_ditches(settings_for_error, surface_level)

    @pytest.mark.unittest
    def test_load_ssi_measure_with_error(self, presets, settings_for_error):
        expected_error = (
            f"{presets.__class__.__name__} does not have daily data for SSI/PSSI in the "
            "required modelling period"
        )
        with pytest.raises(KeyError, match=expected_error):
            presets.load_ssi_measure(
                "ssi", settings_for_error.date_range, 0.7, 4, -2.0, 0.2
            )
