from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely import from_wkt

from parcel2d_modflow.preprocessing.time_range import select_time_range


# %%
def preprocess_calibration_parcels(
    calibration_wells_df: pd.DataFrame,
    flux_data=None,
    recharge_data=None,
    weather_data=None,
) -> gpd.GeoDataFrame:
    """ """

    parcel_df = calibration_wells_df.groupby("aan_id").first().reset_index()
    parcel_df = rename_parcel_columns(parcel_df)

    parcel_df = select_time_range(
        parcel_df,
        flux_data,
        recharge_data,
        weather_data,
    )
    parcel_gdf = create_parcel_gdf(parcel_df)
    return parcel_gdf


def create_parcel_gdf(parcel_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Construct a GeoDataFrame from a parcel table using the parcel geometry column.

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

    parcel_df["parcel_geom"] = parcel_df["parcel_geom"].apply(from_wkt)
    parcel_gdf = gpd.GeoDataFrame(parcel_df, geometry="parcel_geom", crs="EPSG:28992")

    parcel_gdf["parcel_x"] = parcel_gdf.centroid.x
    parcel_gdf["parcel_y"] = parcel_gdf.centroid.y

    return parcel_gdf


def rename_parcel_columns(parcel_df: pd.DataFrame) -> pd.DataFrame:
    """Rename parcel columns to the canonical schema used by the model.

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

    # fill surface level
    parcel_df["surface_level"] = parcel_df["z_surface_level_m_nap"].fillna(
        parcel_df["ahn4_m_nap"]
    )

    return parcel_df.rename(columns=rename_dict)


# %%
