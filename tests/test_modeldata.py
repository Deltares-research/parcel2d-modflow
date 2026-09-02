from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from parcel2d_modflow import components
from parcel2d_modflow.config import ModelSettings
from parcel2d_modflow.exceptions import InvalidPresetDataError, MissingDataError
from parcel2d_modflow.modeldata import GroundwaterData, Presets, Soilmap


@pytest.fixture
def start_date():
    return pd.Timestamp("2022-01-01")


@pytest.fixture
def end_date():
    return pd.Timestamp("2022-02-01")


@pytest.fixture
def settings(tmp_path, start_date, end_date):
    return ModelSettings(
        workdir=tmp_path, start_date=start_date, end_date=end_date, clean_workdir=True
    )


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


class TestGroundwaterData:
    @staticmethod
    def _as_dataarray(obj):
        if isinstance(obj, xr.Dataset) and len(obj.data_vars) == 1:
            return obj[next(iter(obj.data_vars))]
        return obj

    @pytest.mark.unittest
    def test_lhm_data(self, lhm_data):
        assert isinstance(lhm_data, GroundwaterData)
        assert isinstance(lhm_data.confining, xr.DataArray | xr.Dataset)
        assert isinstance(lhm_data.flux, xr.DataArray | xr.Dataset)
        assert isinstance(lhm_data.recharge, xr.DataArray | xr.Dataset)
        assert isinstance(lhm_data.head, xr.DataArray | xr.Dataset)
        assert lhm_data.cell_area == 62500

        lhm = GroundwaterData(None, None, None, None)
        assert isinstance(lhm, GroundwaterData)
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
        with pytest.raises(AttributeError, match="Cannot load recharge from NoneType."):
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
            AttributeError, match="Cannot load phreatic head from NoneType."
        ):
            lhm_data.load_phreatic_head(parcel, model_settings.date_range)

    @pytest.mark.unittest
    def test_load_aquifer_flux(self, lhm_data, parcel, settings):
        aquifer = lhm_data.load_aquifer_flux(parcel, settings)
        assert isinstance(aquifer, components.Aquifer)
        assert isinstance(aquifer.start, float)
        assert isinstance(aquifer.series, np.ndarray)
        assert aquifer.series.size == 32

        lhm_data.flux = None
        with pytest.raises(
            AttributeError, match="Cannot load aquifer flux from NoneType."
        ):
            lhm_data.load_aquifer_flux(parcel, settings)

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
            AttributeError, match="Cannot load confining layer from NoneType."
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
        start_date = pd.to_datetime("1918-01-01")
        end_date = pd.to_datetime("1918-12-31")
        return ModelSettings(
            workdir=Path("."), start_date=start_date, end_date=end_date
        )

    @pytest.fixture
    def parcel_for_error(self, parcel):
        parcel.name = "Unknown Parcel"
        return parcel

    @pytest.mark.unittest
    def test_post_init_validation(self, presets):
        ditch_stage = presets.ditch_stage.rename(
            {"name": "invalid_name", "time": "invalid_time"}
        )
        pssi_stage = presets.pssi_stage.rename(
            {"name": "invalid_name", "time": "invalid_time"}
        )
        with pytest.raises(InvalidPresetDataError):
            Presets(ditch_stage=ditch_stage, pssi_stage=pssi_stage)

    @pytest.mark.unittest
    def test_load_ditches(self, presets, parcel, settings):
        result = presets.load_ditches(parcel, settings)
        assert isinstance(result, components.Ditches)
        assert np.isclose(result.bottom, -3.104884115944994)
        assert result.resistance == 1
        assert_array_almost_equal(
            result.stage,
            [-2.63946319, -2.62906901, -2.6584295, -2.64365931, -2.60707662],
        )
        assert_array_equal(
            result.dates, pd.date_range("2022-01-01", "2022-01-29", freq="7D")
        )

    @pytest.mark.unittest
    def test_load_ditches_error(self, presets, parcel_for_error, settings_for_error):
        with pytest.raises(MissingDataError):
            presets.load_ditches(parcel_for_error, settings_for_error)

    def test_load_ssi_measure(self, presets, parcel, settings):
        result = presets.load_ssi_measure(parcel, settings, "ssi")
        assert isinstance(result, components.SsiMeasure)
        assert np.isclose(result.drain_depth, -2.904884115944994)
        assert result.drain_distance == 1
        assert_array_almost_equal(
            result.drain_stage,
            presets.ditch_stage.sel(name=parcel.name, time=settings.date_range),
        )
        assert_array_equal(result.time, settings.date_range)

        result = presets.load_ssi_measure(parcel, settings, "pssi")
        assert isinstance(result, components.SsiMeasure)
        assert np.isclose(result.drain_depth, -2.7)
        assert result.drain_distance == 1
        assert_array_almost_equal(
            result.drain_stage,
            presets.pssi_stage.sel(name=parcel.name, time=settings.date_range),
        )
        assert_array_equal(result.time, settings.date_range)

    @pytest.mark.unittest
    def test_load_ssi_measure_with_error(
        self, presets, parcel_for_error, settings_for_error
    ):
        with pytest.raises(MissingDataError):
            presets.load_ssi_measure(parcel_for_error, settings_for_error, "ssi")
        with pytest.raises(MissingDataError):
            presets.load_ssi_measure(parcel_for_error, settings_for_error, "pssi")
