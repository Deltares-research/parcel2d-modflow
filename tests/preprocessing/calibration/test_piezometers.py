import geopandas as gpd
import pandas as pd
import pytest

from parcel2d_modflow.preprocessing.calibration import piezometers


@pytest.mark.unittest
def test_load_parcel_piezometers_from_db_has_expected_dimensions_and_coords(
    monkeypatch,
):
    def fake_read(**kwargs):
        location_id = kwargs["user_query"].split("locationkey=")[1].split("\n")[0]
        values = [1.5, 2.5] if location_id == "56" else [7.0, 8.0]
        return pd.DataFrame(
            {
                "scalarvalue": values,
                "datetime": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            }
        )

    monkeypatch.setattr(piezometers, "read_timeseries_from_database", fake_read)
    gdf = gpd.GeoDataFrame({"well_id": ["bro_56", "waterschappen_1056"]})

    result = piezometers.load_parcel_piezometers_from_db(gdf, "connection")

    assert result.dims == ("time", "well_id")
    assert set(result["well_id"].values) == {"bro_56", "waterschappen_1056"}
    assert all(
        f"{source}_{well_id}" in result["well_id"].values
        for source, well_id in [("bro", "56"), ("waterschappen", "1056")]
    )
    time_index = pd.DatetimeIndex(result["time"].values)
    assert time_index.is_monotonic_increasing
    assert time_index.inferred_freq in {"D", None}
