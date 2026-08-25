from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from parcel2d_modflow.preprocessing.calibration import ditches, parcels, piezometers
from parcel2d_modflow.preprocessing.calibration.time_range import (
    select_time_range,
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


@pytest.mark.unittest
def test_create_parcel_gdf_uses_parcel_geometry(parcel_data):
    parcel_gdf = parcels.create_parcel_gdf(parcel_data.iloc[[0]].copy())

    assert isinstance(parcel_gdf, gpd.GeoDataFrame)
    assert parcel_gdf.crs.to_epsg() == 28992
    assert parcel_gdf.geometry.iloc[0].geom_type == "MultiPolygon"
    assert parcel_gdf.loc[0, "parcel_x"] == pytest.approx(
        parcel_gdf.geometry.iloc[0].centroid.x
    )
    assert parcel_gdf.loc[0, "parcel_y"] == pytest.approx(
        parcel_gdf.geometry.iloc[0].centroid.y
    )


@pytest.mark.unittest
def test_rename_parcel_columns_fills_surface_level(parcel_data):
    renamed = parcels.rename_parcel_columns(parcel_data.iloc[[0]].copy())

    assert renamed.loc[0, "surface_level"] == pytest.approx(
        parcel_data.loc[0, "ahn4_m_nap"]
    )
    assert renamed.loc[0, "measure"] == "ref"
    assert renamed.loc[0, "soilcode"] == "pVc"
    assert renamed.loc[0, "width"] == pytest.approx(136.5)


@pytest.mark.unittest
def test_preprocess_calibration_parcels_groups_wells(parcel_data):
    result = parcels.preprocess_calibration_parcels(parcel_data)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.geometry.name == "parcel_geom"
    assert set(result.index) == {14419, 4935}
    assert result.loc[14419, "well_id"] == {"bro_56", "bro_137"}
    assert result.loc[14419, "start_date"] == pd.Timestamp("2018-06-01 09:00")
    assert result.loc[14419, "end_date"] == pd.Timestamp("2023-12-11 12:54")


@pytest.mark.unittest
def test_interpolate_ditch_values_only_fills_interior():
    index = pd.date_range("2024-01-01", periods=4)
    values = pd.DataFrame({"ditch": [np.nan, 1.0, np.nan, np.nan]}, index=index)

    result = ditches.interpolate_ditch_values(values, method="linear")

    assert pd.isna(result.loc[index[0], "ditch"])
    assert pd.isna(result.loc[index[2], "ditch"])
    assert result.loc[index[1], "ditch"] == 1.0


@pytest.mark.unittest
def test_update_time_range_from_inputdata_uses_all_attributes():
    start = pd.Series(pd.to_datetime(["2020-01-01"]), index=[7])
    end = pd.Series(pd.to_datetime(["2020-12-31"]), index=[7])
    data = SimpleNamespace(
        recharge=SimpleNamespace(time=pd.to_datetime(["2020-02-01", "2020-10-01"])),
        flux=SimpleNamespace(time=pd.to_datetime(["2020-03-01", "2020-09-01"])),
    )

    result_start, result_end = update_time_range_from_inputdata(
        data, start, end, attrs=["recharge", "flux"]
    )

    assert result_start.loc[7] == pd.Timestamp("2020-03-01")
    assert result_end.loc[7] == pd.Timestamp("2020-09-01")


@pytest.mark.unittest
def test_update_time_range_from_measurements_handles_sets():
    parcel_df = pd.DataFrame({"well_id": [{"well_a", "well_b"}]}, index=[42])
    measurements = pd.DataFrame(
        {
            "well_a": [np.nan, 1.0, 2.0, np.nan],
            "well_b": [np.nan, np.nan, 3.0, 4.0],
        },
        index=pd.date_range("2020-01-01", periods=4),
    )
    start = pd.Series(pd.Timestamp("2019-01-01"), index=[42])
    end = pd.Series(pd.Timestamp("2021-01-01"), index=[42])

    result_start, result_end = update_time_range_from_measurements(
        parcel_df, measurements, "well_id", start, end
    )

    assert result_start.loc[42] == pd.Timestamp("2020-01-02")
    assert result_end.loc[42] == pd.Timestamp("2020-01-04")


@pytest.mark.unittest
def test_select_time_range_intersects_weather_and_measurements():
    parcel_df = pd.DataFrame(
        {
            "start_date": ["2020-01-01"],
            "end_date": ["2020-12-31"],
            "well_id": [{"well_a"}],
            "ditch_id": [{"ditch_a"}],
        },
        index=[1],
    )
    weather = pd.DataFrame({"YYYYMMDD": pd.to_datetime(["2020-02-01", "2020-11-30"])})
    input_data = SimpleNamespace(time=pd.to_datetime(["2020-03-01", "2020-10-31"]))
    piez = pd.DataFrame(
        {"well_a": [np.nan, 1.0, np.nan]},
        index=pd.date_range("2020-04-01", periods=3),
    )
    ditch = pd.DataFrame(
        {"ditch_a": [2.0, np.nan, 3.0]},
        index=pd.date_range("2020-05-01", periods=3),
    )

    result = select_time_range(parcel_df, input_data, None, weather, piez, ditch)

    assert result.loc[1, "start_date"] == pd.Timestamp("2020-05-01")
    assert result.loc[1, "end_date"] == pd.Timestamp("2020-04-02")
    assert "time" in weather.columns


@pytest.mark.unittest
def test_load_parcel_ditches_from_db_resamples_and_interpolates(
    monkeypatch, calibration_timeseries
):
    calls = []
    _, ditch_data = calibration_timeseries
    ditch_values = ditch_data["nobv_48"].dropna().iloc[:3].tolist()

    def fake_read(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "scalarvalue": [ditch_values[0], np.nan, ditch_values[2]],
                "datetime": pd.to_datetime(
                    ["2020-01-01 12:00", "2020-01-02 12:00", "2020-01-03 12:00"]
                ),
            }
        )

    monkeypatch.setattr(ditches, "read_timeseries_from_database", fake_read)
    gdf = gpd.GeoDataFrame({"ditch_id": ["nobv_48"]})

    result = ditches.load_parcel_ditches_from_db(gdf, "connection")

    assert list(result.columns) == ["nobv_48"]
    assert result.loc["2020-01-02", "nobv_48"] == pytest.approx(
        np.mean([ditch_values[0], ditch_values[2]])
    )
    assert calls[0]["schema_name"] == "nobv_timeseries"
    assert "locationkey=48" in calls[0]["user_query"]


@pytest.mark.unittest
def test_load_parcel_piezometers_from_db_resamples(monkeypatch, calibration_timeseries):
    piezometer_data, _ = calibration_timeseries
    piezometer_values = piezometer_data["bro_56"].dropna().iloc[:2].tolist()

    def fake_read(**kwargs):
        return pd.DataFrame(
            {
                "scalarvalue": piezometer_values,
                "datetime": pd.to_datetime(["2020-01-01 12:00", "2020-01-02 12:00"]),
            }
        )

    monkeypatch.setattr(piezometers, "read_timeseries_from_database", fake_read)
    gdf = gpd.GeoDataFrame({"well_id": ["bro_56"]})

    result = piezometers.load_parcel_piezometers_from_db(gdf, "connection")

    assert list(result.columns) == ["bro_56"]
    assert result.loc["2020-01-01", "bro_56"] == pytest.approx(piezometer_values[0])
    assert result.loc["2020-01-02", "bro_56"] == pytest.approx(piezometer_values[1])


@pytest.mark.integrationtest
def test_run_preprocessing_workflow_from_repository_data(testdatadir, monkeypatch):
    parcel_df = pd.read_csv(testdatadir / "test_parcel_df.csv")
    piezometer_data = pd.read_csv(testdatadir / "test_piez_df.csv")
    ditch_data = pd.read_csv(testdatadir / "test_ditch_df.csv")

    def fake_read_timeseries(**kwargs):
        source = kwargs["schema_name"].removesuffix("_timeseries")
        identifier = kwargs["user_query"].split("locationkey=")[1].split("\n")[0]
        column = f"{source}_{identifier}"
        measurement_data = piezometer_data if column in piezometer_data else ditch_data
        values = measurement_data[column].to_numpy()
        measurement_index = pd.date_range("2022-01-01", periods=len(values), freq="D")
        return pd.DataFrame(
            {
                "scalarvalue": values,
                "datetime": measurement_index,
            }
        )

    monkeypatch.setattr(ditches, "read_timeseries_from_database", fake_read_timeseries)
    monkeypatch.setattr(
        piezometers, "read_timeseries_from_database", fake_read_timeseries
    )
    parcel_gdf = parcels.preprocess_calibration_parcels(
        parcel_df,
        flux_data=xr.open_dataarray(testdatadir / "lhm_flux.nc"),
        recharge_data=xr.open_dataarray(testdatadir / "lhm_recharge.nc"),
        piez_df=piezometers.load_parcel_piezometers_from_db(parcel_df, "connection"),
        ditch_df=ditches.load_parcel_ditches_from_db(parcel_df, "connection"),
    )

    assert set(parcel_gdf.index) == {14419, 4935}
    assert parcel_gdf.geometry.name == "parcel_geom"
    assert parcel_gdf["start_date"].min() >= pd.Timestamp("2022-01-01")
    assert parcel_gdf["end_date"].max() <= pd.Timestamp("2022-10-31")
