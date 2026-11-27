import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

pytest.importorskip("hydroeval")

analysis = importlib.import_module(
    "parcel2d_modflow.postprocessing.calibration.analyze_calibration"
)
plots = importlib.import_module(
    "parcel2d_modflow.postprocessing.calibration.plots_compare_calimethods"
)


@pytest.mark.unittest
def test_imports_do_not_read_data_or_write_files(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Import performed file or database access")

    monkeypatch.setattr(pd, "read_csv", unexpected)
    monkeypatch.setattr(xr, "open_dataarray", unexpected)
    monkeypatch.setattr(analysis.gpd, "read_parquet", unexpected)
    monkeypatch.setattr("pathlib.Path.mkdir", unexpected)
    importlib.reload(analysis)
    importlib.reload(plots)


@pytest.mark.unittest
def test_calculate_weighted_performance():
    performance = pd.DataFrame(
        {
            "r": [0.8, 1.0],
            "alpha": [1.2, 0.8],
            "beta": [1.0, 1.4],
            "rmse": [0.1, 0.5],
            "count": [1, 3],
        },
        index=pd.MultiIndex.from_tuples(
            [("well1", 1), ("well2", 1)], names=["piez_name", "runs"]
        ),
    )

    result = analysis.calculate_weighted_performance(performance)

    assert_array_equal(result.index, [1])
    assert result.loc[1, "r"] == pytest.approx(0.95)
    assert result.loc[1, "alpha"] == pytest.approx(0.9)
    assert result.loc[1, "beta"] == pytest.approx(1.3)
    assert result.loc[1, "rmse"] == pytest.approx(0.4)
    assert result.loc[1, "kge"] == pytest.approx(1 - np.sqrt(0.1025))


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        (np.nan, []),
        ("well1", ["well1"]),
        (["well1", None, "well2"], ["well1", "well2"]),
        (np.array(["well1", "well2"]), ["well1", "well2"]),
    ],
)
def test_as_well_list(value, expected):
    assert analysis._as_well_list(value) == expected
