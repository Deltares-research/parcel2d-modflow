from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from parcel2d_modflow import components
from parcel2d_modflow.base import ModelSettings
from parcel2d_modflow.modeldata import LhmData, Soilmap


@pytest.fixture
def start_date():
    return pd.Timestamp("2022-01-01")


@pytest.fixture
def end_date():
    return pd.Timestamp("2022-02-01")


@pytest.fixture
def soilmap_files(tmp_path, soilmap):
    """
    Create individual files to test Soilmap.from_files.

    """
    soilmap_file = tmp_path / r"soilmap.geoparquet"
    soilprofiles_file = tmp_path / r"soilprofiles.parquet"
    soilmap.soilmap.to_parquet(soilmap_file)

    sp = soilmap.soilprofiles
    # Convert to percentage. Soilmap gives organic matter content in percentage.
    sp["organicmattercontent"] = sp["organicmattercontent"] * 100
    # Columns "sand", "lithology", and "thickness" are generated in Soilmap.from_files
    sp.drop(columns=["sand", "lithology", "thickness"]).to_parquet(soilprofiles_file)
    return soilmap_file, soilprofiles_file


class TestLhmData:
    @pytest.mark.unittest
    def test_lhm_data(self, lhm_data):
        assert isinstance(lhm_data, LhmData)
        assert isinstance(lhm_data.confining, xr.DataArray | xr.Dataset)
        assert isinstance(lhm_data.flux, xr.DataArray | xr.Dataset)
        assert isinstance(lhm_data.recharge, xr.DataArray | xr.Dataset)
        assert isinstance(lhm_data.head, xr.DataArray | xr.Dataset)
        assert lhm_data.cell_area == 62500

        lhm = LhmData(None, None, None, None)
        assert isinstance(lhm, LhmData)
        assert lhm.confining is None
        assert lhm.flux is None
        assert lhm.recharge is None
        assert lhm.head is None
        assert lhm.cell_area is None

    @pytest.mark.unittest
    def test_load_recharge(self, lhm_data, parcel, start_date, end_date):
        recharge = lhm_data.load_recharge(parcel, start_date, end_date)
        assert isinstance(recharge, components.Recharge)
        assert isinstance(recharge.start, float)
        assert isinstance(recharge.series, np.ndarray)
        assert recharge.series.size == 32

        lhm_data.recharge = None
        with pytest.raises(
            AttributeError, match="Cannot load recharge from LhmData. LhmData.recharge"
        ):
            lhm_data.load_recharge(parcel, start_date, end_date)

    @pytest.mark.unittest
    def test_load_phreatic_head(self, lhm_data, parcel, model_settings):
        head = lhm_data.load_phreatic_head(parcel, model_settings.date_range)
        assert isinstance(head, xr.DataArray)
        assert head.size == 7

        invalid_date_range = model_settings.date_range - pd.Timedelta(2)
        with pytest.raises(
            KeyError, match="Phreatic head does not have data for the modelling period"
        ):
            lhm_data.load_phreatic_head(parcel, invalid_date_range)

        lhm_data.head = None
        with pytest.raises(
            AttributeError, match="Cannot load phreatic head from LhmData. LhmData.head"
        ):
            lhm_data.load_phreatic_head(parcel, model_settings.date_range)

    @pytest.mark.unittest
    def test_load_aquifer_flux(self, lhm_data, parcel, start_date, end_date):
        aquifer = lhm_data.load_aquifer_flux(parcel, start_date, end_date)
        assert isinstance(aquifer, components.Aquifer)
        assert isinstance(aquifer.start, float)
        assert isinstance(aquifer.series, np.ndarray)
        assert aquifer.series.size == 32

        lhm_data.flux = None
        with pytest.raises(
            AttributeError, match="Cannot load aquifer flux from LhmData. LhmData.flux"
        ):
            lhm_data.load_aquifer_flux(parcel, start_date, end_date)

    @pytest.mark.unittest
    def test_load_confining_layer(self, lhm_data, parcel):
        confining, thin_confining_layer = lhm_data.load_confining_layer(
            parcel, 1.2
        )  # Assume all Holocene
        assert isinstance(confining, components.SubsurfaceStructure)
        assert not thin_confining_layer
        assert_array_equal(
            confining.thickness,
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
        )
        assert_array_equal(confining.lithology, [1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4])
        assert_array_equal(confining.geology, [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2])
        assert_array_equal(confining.kvalues, [0.01, 2200.0])

        confining, thin_confining_layer = lhm_data.load_confining_layer(parcel, 0.5)
        assert thin_confining_layer
        assert_array_equal(confining.thickness, [1.0])
        assert_array_equal(confining.lithology, [4])
        assert_array_equal(confining.geology, [2])
        assert_array_equal(confining.kvalues, [70.0, 2200.0])

        # Test different thickness of resistance layer
        confining, thin_confining_layer = lhm_data.load_confining_layer(
            parcel, 1.2, 0.15
        )
        assert_array_equal(
            confining.thickness,
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.15, 1.0],
        )

        lhm_data.confining = None
        with pytest.raises(
            AttributeError,
            match="Cannot load confining layer from LhmData. LhmData.confining",
        ):
            lhm_data.load_confining_layer(parcel, 1.2)


class TestSoilmap:
    @pytest.mark.unittest
    def test_initialize_soilmap(self, soilmap):
        assert isinstance(soilmap, Soilmap)
        assert isinstance(soilmap.soilmap, gpd.GeoDataFrame)
        assert isinstance(soilmap.soilprofiles, pd.DataFrame)

    @pytest.mark.unittest
    def test_from_files(self, soilmap_files):
        soilmap_file, soilprofiles_file = soilmap_files
        s = Soilmap.from_files(soilmap_file, soilprofiles_file)
        assert isinstance(s, Soilmap)
        assert all(
            col in s.soilprofiles.columns for col in ["sand", "lithology", "thickness"]
        )
        assert_array_equal(s.soilprofiles["sand"], [20, 5, 5, 5, 20, 5, 25, 25])
        assert_array_equal(s.soilprofiles["lithology"], [3, 2, 1, 1, 3, 1, 1, 1])
        assert_array_almost_equal(
            s.soilprofiles["thickness"],
            [0.2, 0.15, 0.35, 0.5, 0.15, 0.15, 0.2, 0.7],
        )
        assert_array_almost_equal(
            s.soilprofiles["organicmattercontent"],
            [0.35, 0.25, 0.5, 0.7, 0.35, 0.5, 0.75, 0.8],
        )

    @pytest.mark.parametrize("x, y, expected", [(1, 1, "hVb"), (3, 1, "hVc")])
    def test_soilcode_at(self, x, y, expected, soilmap):
        soilunit_code = soilmap.soilcode_at(x, y)
        assert soilunit_code == expected

    @pytest.mark.unittest
    def test_load_soilprofile(self, empty_parcel, soilmap):
        profile = soilmap.load_soilprofile(empty_parcel)
        assert isinstance(profile, pd.DataFrame)
        assert_array_equal(profile["normalsoilprofile_id"], 1010)
        assert_array_equal(
            profile.columns,
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
                "geology",
            ],
        )

        empty_parcel.soilcode = "hVc"
        profile = soilmap.load_soilprofile(empty_parcel)
        assert_array_equal(profile["normalsoilprofile_id"], 1050)


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
