import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from parcel2d_modflow import utils


@pytest.mark.unittest
def test_strip_column_units():
    df = pd.DataFrame(columns=["column1 (m)", "column2 (cm)", "column3 (kg)"])
    stripped_df = utils.strip_column_units(df)
    assert_array_equal(stripped_df.columns, ["column1", "column2", "column3"])

    df = pd.DataFrame(columns=["column1", "column2"])
    stripped_df = utils.strip_column_units(df)
    assert_array_equal(stripped_df.columns, ["column1", "column2"])
