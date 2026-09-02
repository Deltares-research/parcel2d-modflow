# %%
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import from_wkt

from parcel2d_modflow.preprocessing.calibration.time_range import select_time_range


def preprocess_calibration_parcels(
    calibration_wells_df: pd.DataFrame,
    flux_data=None,
    recharge_data=None,
    weather_data=None,
    piez_da=None,
    ditch_da=None,
) -> gpd.GeoDataFrame:
    """
    Preprocess calibration parcels by renaming columns, selecting the valid time range,
    and creating a GeoDataFrame.

    """
    agg_funcs = {col: "first" for col in calibration_wells_df.columns}
    agg_funcs.update({"start_date": "min", "end_date": "max", "well_id": set})
    parcel_df = calibration_wells_df.groupby(
        ["aan_id", "transect", "parcel_type"], as_index=False
    ).agg(agg_funcs)

    parcel_df = select_time_range(
        parcel_df, flux_data, recharge_data, weather_data, piez_da, ditch_da
    )

    parcel_df = select_correct_time_range(parcel_df)
    parcel_gdf = create_parcel_gdf(parcel_df)
    parcel_gdf = rename_parcel_columns(parcel_gdf)
    parcel_gdf = parcel_gdf.drop(columns=["geometry"]).reset_index(drop=True)
    return parcel_gdf


def select_correct_time_range(
    parcel_df: pd.DataFrame, min_period="365D"
) -> pd.DataFrame:
    """
    Filter parcels by the minimum required measurement duration.

    Parameters
    ----------
    parcel_df : pd.DataFrame
        Parcel table containing parcel data and ``start_date`` and ``end_date`` columns.
    min_period : str, optional
        Minimum valid time span for a parcel, by default "365D".

    Returns
    -------
    pd.DataFrame
        Parcel rows whose time range is at least the requested minimum period.
    """
    return parcel_df[
        parcel_df.end_date >= parcel_df.start_date + pd.Timedelta(min_period)
    ]


def create_parcel_gdf(parcel_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Construct a GeoDataFrame from a parcel table using the parcel geometry column.

    Parameters
    ----------
    parcel_df : pd.DataFrame
        Parcel table with a WKT-based ``parcel_geom`` column and parcel attributes.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with ``parcel_geom`` converted to Shapely geometries, assigned the
        Dutch national CRS (EPSG:28992), and with centroid-based ``parcel_x`` and
        ``parcel_y`` columns added.

    """
    if (
        not parcel_df["parcel_geom"]
        .map(lambda geom: geom is not None and hasattr(geom, "geom_type"))
        .all()
    ):
        parcel_df["parcel_geom"] = parcel_df["parcel_geom"].apply(from_wkt)
    parcel_gdf = gpd.GeoDataFrame(parcel_df, geometry="parcel_geom", crs="EPSG:28992")

    parcel_gdf["parcel_x"] = parcel_gdf.centroid.x
    parcel_gdf["parcel_y"] = parcel_gdf.centroid.y

    return parcel_gdf


def rename_parcel_columns(parcel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename parcel columns to the canonical schema used by the model.

    Parameters
    ----------
    parcel_df : pd.DataFrame
        Input parcel table containing the original column names from the source data.

    Returns
    -------
    pd.DataFrame
        A copy of the input dataframe with columns renamed according to the internal
        parcel naming convention used downstream in the preprocessing pipeline.

    """
    required_columns = {
        "name",
        "parcel_type",
        "parcel_width_m",
        "soil_class",
        "summer_stage_m_nap",
        "winter_stage_m_nap",
        "trench_depth_m_sfl",
        "trenches",
        "wis_depth_m_sfl",
        "wis_distance_m",
    }

    if not required_columns.issubset(parcel_df.columns):
        missing_cols = required_columns - set(parcel_df.columns)
        raise ValueError(
            f"Missing expected columns in parcel_df: {missing_cols}. Please check the input data."
        )

    if "parcel_x" not in parcel_df.columns or "parcel_y" not in parcel_df.columns:
        if "parcel_geom" in parcel_df.columns:
            parcel_df = create_parcel_gdf(parcel_df)

    if "surface_level" not in parcel_df.columns:
        if (
            "z_surface_level_m_nap" not in parcel_df.columns
            and "ahn4_m_nap" not in parcel_df.columns
        ):
            raise ValueError("Surface level columns are missing from parcel_df.")
        parcel_df["surface_level"] = parcel_df.get("z_surface_level_m_nap").fillna(
            parcel_df.get("ahn4_m_nap")
        )

    if parcel_df["surface_level"].isnull().any():
        null_parcels = parcel_df[parcel_df["surface_level"].isnull()].name
        raise ValueError(
            f"Surface level is missing for some parcels: {null_parcels.tolist()}. Please check the input data."
        )

    rename_dict = {
        "name": "name",
        "parcel_x": "x",
        "parcel_y": "y",
        "parcel_type": "measure",
        "parcel_width_m": "width",
        "surface_level": "surface_level",
        "soil_class": "soilcode",
        "summer_stage_m_nap": "summer_stage",
        "winter_stage_m_nap": "winter_stage",
        "trench_depth_m_sfl": "trench_depth",
        "trenches": "trench_locations",
        "wis_depth_m_sfl": "drain_depth",
        "wis_distance_m": "drain_distance",
    }

    return parcel_df.rename(columns=rename_dict)


# %%
