import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from parcel2d_modflow._io import read
from parcel2d_modflow.modeldata import LhmData, Soilmap
from parcel2d_modflow.validation import ValidationError


@pytest.fixture
def modflow_parameter_file(tmp_path, modflow_parameters):
    file = tmp_path / "modflow_parameters.csv"
    modflow_parameters.to_csv(file, index=False)
    return file


@pytest.fixture
def modflow_parameter_file_missing_columns(tmp_path, modflow_parameters):
    modflow_parameters.rename(columns={"kh (m/d)": "wrong"}, inplace=True)
    file = tmp_path / "modflow_parameters_missing_columns.csv"
    modflow_parameters.to_csv(file)
    return file


@pytest.fixture
def modflow_parameter_file_wrong_index(tmp_path, modflow_parameters):
    modflow_parameters["wrong_index"] = modflow_parameters["runnr"] + 10
    file = tmp_path / "modflow_parameters_wrong_index.csv"
    modflow_parameters.to_csv(file)
    return file


@pytest.fixture
def invalid_bro_soilmap(testdatadir):
    return testdatadir / "test_invalid_soilmap_v2023.gpkg"


@pytest.mark.unittest
def test_read_lhm_data(
    lhm_confining_nc, lhm_flux_nc, lhm_recharge_nc, lhm_phreatic_head_nc
):
    lhm_data = read.read_lhm_data(
        lhm_confining_nc,
        lhm_flux_nc,
        lhm_recharge_nc,
        lhm_phreatic_head_nc,
    )
    assert isinstance(lhm_data, LhmData)
    assert isinstance(lhm_data.confining, xr.Dataset)
    assert isinstance(lhm_data.flux, xr.DataArray)
    assert isinstance(lhm_data.recharge, xr.DataArray)
    assert isinstance(lhm_data.head, xr.DataArray)
    assert lhm_data.confining.sizes == {"x": 1, "y": 1}
    assert lhm_data.flux.sizes == {"x": 1, "y": 1, "time": 304}
    assert lhm_data.recharge.sizes == {"x": 1, "y": 1, "time": 365}
    assert lhm_data.head.sizes == {"x": 1, "y": 1, "time": 365}

    assert_array_equal(
        lhm_data.confining.data_vars,
        ["bottom", "thickness", "resistance", "k_value_1aq", "kd_value_1aq"],
    )

    # Test with all inputs None
    lhm_data = read.read_lhm_data(None, None, None, None)
    assert isinstance(lhm_data, LhmData)
    assert lhm_data.confining is None
    assert lhm_data.flux is None
    assert lhm_data.recharge is None
    assert lhm_data.head is None


@pytest.mark.unittest
def test_read_bro_soilmap(simple_bro_soilmap):
    bro_soilmap = read.read_bro_soilmap(simple_bro_soilmap)

    assert isinstance(bro_soilmap, Soilmap)
    assert isinstance(bro_soilmap.soilmap, gpd.GeoDataFrame)
    assert isinstance(bro_soilmap.soilprofiles, pd.DataFrame)

    assert_array_equal(
        bro_soilmap.soilmap.columns,
        ["maparea_id", "geometry", "normalsoilprofile_id", "soilunit_code"],
    )
    assert_array_equal(
        bro_soilmap.soilprofiles.columns,
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
        ],
    )


@pytest.mark.unittest
def test_read_invalid_soilmap(invalid_bro_soilmap):
    with pytest.raises(ValidationError):
        read.read_bro_soilmap(invalid_bro_soilmap)


@pytest.mark.unittest
def test_read_modflow_parameters(
    modflow_parameter_file,
    modflow_parameter_file_wrong_index,
    modflow_parameter_file_missing_columns,
):
    modflow_parameters = read.read_modflow_parameters(modflow_parameter_file)
    assert isinstance(modflow_parameters, pd.DataFrame)
    assert_array_equal(
        modflow_parameters.columns, ["runnr", "kh (m/d)", "sy_peat (-)", "sy_clay (-)"]
    )

    error = "Index of modflow parameters DataFrame is not correct. Expected a RangeIndex starting from 0 with step 1"
    with pytest.raises(ValidationError, match=error):
        read.read_modflow_parameters(
            modflow_parameter_file_wrong_index, index_col="wrong_index"
        )

    with pytest.raises(
        ValidationError, match="Modflow parameters DataFrame is missing columns:"
    ):
        read.read_modflow_parameters(modflow_parameter_file_missing_columns)
