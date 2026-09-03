from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from parcel2d_modflow.preprocessing.calibration.time_range import (
    update_time_range_from_inputdata,
    update_time_range_from_measurements,
)


@pytest.fixture
def measurement_bounds():
    start = pd.Series([pd.Timestamp("2019-01-01")], index=[42])
    end = pd.Series([pd.Timestamp("2021-01-01")], index=[42])
    return start, end


@pytest.fixture
def valid_input_data():
    return SimpleNamespace(
        recharge=SimpleNamespace(time=pd.to_datetime(["2020-02-01", "2020-10-01"])),
        flux=SimpleNamespace(time=pd.to_datetime(["2020-03-01", "2020-09-01"])),
    )


@pytest.fixture
def valid_measurements():
    return xr.DataArray(
        np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, 3.0]], dtype=float),
        dims=["time", "well_id"],
        coords={
            "time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "well_id": ["well_a", "well_b"],
        },
    )


@pytest.fixture
def empty_measurements():
    return xr.DataArray(
        np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=float),
        dims=["time", "well_id"],
        coords={
            "time": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "well_id": ["well_a", "well_b"],
        },
    )


@pytest.mark.unittest
def test_update_time_range_from_inputdata_raises_attribute_error_and_limits_bounds(
    valid_input_data,
):
    start = pd.Series(pd.to_datetime(["2020-01-01"]), index=[7])
    end = pd.Series(pd.to_datetime(["2020-12-31"]), index=[7])

    with pytest.raises(AttributeError):
        update_time_range_from_inputdata(
            valid_input_data,
            start.copy(),
            end.copy(),
            attrs=["recharge", "missing_attr"],
        )

    result_start, result_end = update_time_range_from_inputdata(
        valid_input_data, start.copy(), end.copy(), attrs=["recharge", "flux"]
    )
    assert result_start.loc[7] == pd.Timestamp("2020-03-01")
    assert result_end.loc[7] == pd.Timestamp("2020-09-01")


@pytest.mark.unittest
def test_update_time_range_from_measurements_checks_column_warning_and_bounds(
    measurement_bounds, empty_measurements, valid_measurements
):
    parcel_df = pd.DataFrame({"well_id": [{"well_a", "well_b"}]}, index=[42])
    valid_start, valid_end = measurement_bounds

    with pytest.raises(ValueError, match="well_id"):
        update_time_range_from_measurements(
            parcel_df.rename(columns={"well_id": "other_id"}),
            xr.DataArray(
                np.ones((2, 2)),
                dims=["time", "well_id"],
                coords={"time": [0, 1], "well_id": ["well_a", "well_b"]},
            ),
            "well_id",
            valid_start.copy(),
            valid_end.copy(),
        )

    with pytest.warns(UserWarning, match="No measurements found"):
        start_result, end_result = update_time_range_from_measurements(
            parcel_df,
            empty_measurements,
            "well_id",
            valid_start.copy(),
            valid_end.copy(),
        )
    assert start_result.loc[42] == pd.Timestamp("2019-01-01")
    assert end_result.loc[42] == pd.Timestamp("2021-01-01")

    result_start, result_end = update_time_range_from_measurements(
        parcel_df, valid_measurements, "well_id", valid_start.copy(), valid_end.copy()
    )
    assert result_start.loc[42] == pd.Timestamp("2020-01-01")
    assert result_end.loc[42] == pd.Timestamp("2020-01-03")
