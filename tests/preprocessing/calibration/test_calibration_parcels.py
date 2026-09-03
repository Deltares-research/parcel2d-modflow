import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from parcel2d_modflow.preprocessing.calibration import calibration_parcels


@pytest.fixture
def parcel_data(testdatadir):
    data = pd.read_csv(testdatadir / "test_parcel_df.csv")
    data["start_date"] = pd.to_datetime(data["start_date"])
    data["end_date"] = pd.to_datetime(data["end_date"])
    return data


@pytest.mark.unittest
def test_create_parcel_gdf_keeps_parcel_geometry(parcel_data):
    parcel_gdf = calibration_parcels.create_parcel_gdf(parcel_data.iloc[[0]].copy())

    assert "parcel_geom" in parcel_gdf.columns
    assert parcel_gdf.geometry.name == "parcel_geom"
    assert isinstance(parcel_gdf, gpd.GeoDataFrame)
    assert parcel_gdf.crs.to_epsg() == 28992


@pytest.mark.unittest
def test_rename_parcel_columns_contains_surface_level_fields(parcel_data):
    assert "z_surface_level_m_nap" in parcel_data.columns
    assert "ahn4_m_nap" in parcel_data.columns

    renamed = calibration_parcels.rename_parcel_columns(parcel_data.iloc[[0]].copy())

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
        calibration_parcels.rename_parcel_columns(valid)

    valid["z_surface_level_m_nap"] = np.nan
    valid["ahn4_m_nap"] = -1.5
    renamed = calibration_parcels.rename_parcel_columns(valid)
    assert renamed.loc[0, "surface_level"] == pytest.approx(-1.5)


@pytest.mark.unittest
def test_rename_parcel_columns_raises_for_missing_original_columns(parcel_data):
    valid = parcel_data.iloc[[0]].copy()
    with pytest.raises(ValueError, match="Missing expected columns"):
        calibration_parcels.rename_parcel_columns(valid.drop(columns=["soil_class"]))

    renamed = calibration_parcels.rename_parcel_columns(valid)
    assert "soilcode" in renamed.columns
