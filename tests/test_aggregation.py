import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from parcel2d_modflow import aggregation


@pytest.fixture
def phreatic_head():
    time = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    runs = [1, 2, 3]
    x = np.arange(0.25, 4, 0.25)

    rng = np.random.default_rng(seed=42)
    return xr.DataArray(
        rng.random((len(runs), len(time), len(x))),
        coords={"runs": runs, "time": time, "x": x},
        dims=("runs", "time", "x"),
    )


@pytest.mark.unittest
def test_calculate_lg3(phreatic_head):
    lg3 = aggregation.calculate_lg3(phreatic_head)
    assert isinstance(lg3, xr.DataArray)
    assert_array_equal(lg3.coords["year"], [2020, 2021, 2022, 2023])
    assert_array_equal(lg3.coords["runs"], [1, 2, 3])
    assert_array_almost_equal(
        lg3,
        [
            [0.00570127, 0.0044956, 0.00529893, 0.00477711],
            [0.00408223, 0.00571486, 0.00475136, 0.00572175],
            [0.00491208, 0.00454694, 0.00595689, 0.00595483],
        ],
    )
