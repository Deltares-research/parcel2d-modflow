import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from parcel2d_modflow.preprocessing.calibration import ditches


@pytest.mark.unittest
def test_interpolate_ditch_values_interpolates_between_known_values_only():
    time_index = pd.date_range("2024-01-01", periods=5, freq="D")
    values = pd.DataFrame(
        {"ditch": [np.nan, 1.0, np.nan, 3.0, np.nan]}, index=time_index
    )

    result = ditches.interpolate_ditch_values(values, method="linear")
    assert_array_equal(result["ditch"], [np.nan, 1.0, 2.0, 3.0, np.nan])


@pytest.mark.unittest
def test_load_parcel_ditches_from_db_has_expected_dimensions_and_coords(monkeypatch):
    def fake_read(**kwargs):
        location_id = kwargs["user_query"].split("locationkey=")[1].split("\n")[0]
        values = [1.0, 2.0, 3.0] if location_id == "48" else [10.0, 20.0]
        return pd.DataFrame(
            {
                "scalarvalue": values,
                "datetime": pd.date_range("2020-01-01", periods=len(values), freq="D"),
            }
        )

    monkeypatch.setattr(ditches, "read_timeseries_from_database", fake_read)
    gdf = gpd.GeoDataFrame(
        {"ditch_id": ["nobv_48", "waterschappen_1056"], "name": ["name_1", "name_2"]}
    )

    result = ditches.load_parcel_ditches_from_db(gdf, "connection")
    assert result.sizes == {"name": 2, "time": 3}
    assert_array_equal(result["name"], ["name_1", "name_2"])
    assert_array_equal(
        result["time"], pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    )
