from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from parcel2d_modflow.preprocessing.calibration import ditches, parcels, piezometers
from parcel2d_modflow.preprocessing.calibration.time_range import (
    update_time_range_from_inputdata,
    update_time_range_from_measurements,
)


@pytest.fixture
def parcel_data(testdatadir):
    data = pd.read_csv(testdatadir / "test_parcel_df.csv")
    data["start_date"] = pd.to_datetime(data["start_date"])
    data["end_date"] = pd.to_datetime(data["end_date"])
    return data


@pytest.fixture
def calibration_timeseries(testdatadir):
    piezometer_data = pd.read_csv(testdatadir / "test_piez_df.csv")
    ditch_data = pd.read_csv(testdatadir / "test_ditch_df.csv")
    return piezometer_data, ditch_data


@pytest.fixture
def valid_input_data():
    return SimpleNamespace(
        recharge=SimpleNamespace(time=pd.to_datetime(["2020-02-01", "2020-10-01"])),
        flux=SimpleNamespace(time=pd.to_datetime(["2020-03-01", "2020-09-01"])),
    )


@pytest.fixture
def measurement_bounds():
    start = pd.Series([pd.Timestamp("2019-01-01")], index=[42])
    end = pd.Series([pd.Timestamp("2021-01-01")], index=[42])
    return start, end


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


@pytest.mark.unittest
def test_interpolate_ditch_values_interpolates_between_known_values_only():
    time_index = pd.date_range("2024-01-01", periods=5, freq="D")
    values = pd.DataFrame(
        {"ditch": [np.nan, 1.0, np.nan, 3.0, np.nan]}, index=time_index
    )

    result = ditches.interpolate_ditch_values(values, method="linear")

    assert pd.isna(result.loc[time_index[0], "ditch"])
    assert result.loc[time_index[1], "ditch"] == pytest.approx(1.0)
    assert result.loc[time_index[2], "ditch"] == pytest.approx(2.0)
    assert result.loc[time_index[3], "ditch"] == pytest.approx(3.0)
    assert pd.isna(result.loc[time_index[4], "ditch"])


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
    print(result)
    assert result.dims == ("name", "time")
    assert set(result["name"].values) == {"name_1", "name_2"}
    time_index = pd.DatetimeIndex(result["time"].values)
    assert (time_index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


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


@pytest.mark.unittest
def test_create_parcel_gdf_keeps_parcel_geometry(parcel_data):
    parcel_gdf = parcels.create_parcel_gdf(parcel_data.iloc[[0]].copy())

    assert "parcel_geom" in parcel_gdf.columns
    assert parcel_gdf.geometry.name == "parcel_geom"
    assert isinstance(parcel_gdf, gpd.GeoDataFrame)
    assert parcel_gdf.crs.to_epsg() == 28992


@pytest.mark.unittest
def test_rename_parcel_columns_contains_surface_level_fields(parcel_data):
    assert "z_surface_level_m_nap" in parcel_data.columns
    assert "ahn4_m_nap" in parcel_data.columns

    renamed = parcels.rename_parcel_columns(parcel_data.iloc[[0]].copy())

    assert "surface_level" in renamed.columns
    assert renamed.loc[0, "surface_level"] == pytest.approx(
        parcel_data.loc[0, "ahn4_m_nap"]
    )


@pytest.mark.unittest
def test_rename_parcel_columns_raises_for_nan_surface_levels(parcel_data):
    valid = parcel_data.iloc[[0]].copy()
    valid["z_surface_level_m_nap"] = np.nan
    valid["ahn4_m_nap"] = np.nan

    with pytest.raises(ValueError, match="Surface level is missing"):
        parcels.rename_parcel_columns(valid)

    valid["z_surface_level_m_nap"] = np.nan
    valid["ahn4_m_nap"] = -1.5
    renamed = parcels.rename_parcel_columns(valid)
    assert renamed.loc[0, "surface_level"] == pytest.approx(-1.5)


@pytest.mark.unittest
def test_rename_parcel_columns_raises_for_missing_original_columns(parcel_data):
    valid = parcel_data.iloc[[0]].copy()
    with pytest.raises(ValueError, match="Missing expected columns"):
        parcels.rename_parcel_columns(valid.drop(columns=["soil_class"]))

    renamed = parcels.rename_parcel_columns(valid)
    assert "soilcode" in renamed.columns
