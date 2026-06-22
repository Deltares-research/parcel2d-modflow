import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from parcel2d_modflow import Soilmap, utils


@pytest.mark.unittest
def test_strip_column_units():
    df = pd.DataFrame(columns=["column1 (m)", "column2 (cm)", "column3 (kg)"])
    stripped_df = utils.strip_column_units(df)
    assert_array_equal(stripped_df.columns, ["column1", "column2", "column3"])

    df = pd.DataFrame(columns=["column1", "column2"])
    stripped_df = utils.strip_column_units(df)
    assert_array_equal(stripped_df.columns, ["column1", "column2"])


@pytest.mark.unittest
def test_invalid_read_geopandas(tmp_path):
    invalid_file = tmp_path / r"invalid_file.csv"
    pd.DataFrame({"a": [1]}).to_csv(invalid_file)
    with pytest.raises(
        ValueError, match="File type .csv is not supported by geopandas"
    ):
        utils.geopandas_read(invalid_file)


@pytest.fixture
def dummy_soilprofiles():
    return pd.DataFrame(
        {
            "organicmattercontent": [0.55, 0.3, 0.75, 0.03, 0.018, 0.003, 0.008],
            "loamcontent": [95, 88, 95, 50, 97, 2, 30],
            "lutitecontent": [60, 45, 45, 20, 55, 1, 15],
            "siltcontent": [35, 43, 50, 30, 42, 1, 15],
            "peattype": [None, "verweerdKleirijk", "zeggeveen", None, None, None, None],
        }
    )


@pytest.mark.parametrize(
    "sp, expected",
    [
        ("dummy_soilprofiles", [1, 3, 1, 3, 2, 4, 4]),
        ("soilmap", [3, 2, 1, 1, 3, 1, 1, 1]),
    ],
)
def test_determine_lithology_from(request, sp, expected):
    sp = request.getfixturevalue(sp)
    if isinstance(sp, Soilmap):
        sp = sp.soilprofiles
    lithology = utils.determine_lithology_from(sp)
    assert_array_equal(lithology, expected)


@pytest.mark.parametrize(
    "lith, expected",
    [
        (np.array([1, 2, 3, 4]), np.array([1, 1, 1, 2])),
        (np.array([4, 4, 4, 4]), np.array([1, 2, 2, 2])),
        (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])),
        (np.array([1, 4, 1, 4]), np.array([1, 1, 1, 2])),
        (np.array([1, 4, 1, 2]), np.array([1, 1, 1, 1])),
        (np.array([1, 2, 4, 2]), np.array([1, 1, 1, 1])),
        (np.array([1, 2, 4, 4]), np.array([1, 1, 2, 2])),
    ],
)
def test_lithology_to_geology(lith, expected):
    result = utils.lithology_to_geology(lith)
    assert_array_equal(result, expected)
