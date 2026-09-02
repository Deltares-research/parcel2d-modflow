import pandas as pd
import xarray as xr

from parcel2d_modflow.io.postgis import read_timeseries_from_database


# %%
def load_parcel_ditches_from_db(gdf, connection) -> pd.DataFrame:
    """
    Load ditch stage data for the ditches associated with the parcels in the provided GeoDataFrame

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing parcel information, including ditch identifiers.
    connection : sqlalchemy.engine.Engine
        SQLAlchemy engine for connecting to the database.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ditch stage data for the associated ditches, resampled to daily frequency and interpolated.
    """
    ditch_df = []
    for name, (source, ditch_id) in zip(gdf.name, gdf.ditch_id.str.split("_")):
        ditch_stage = read_timeseries_from_database(
            engine=connection,
            select_name="tsv.scalarvalue, tsv.datetime",
            schema_name=f"{source}_timeseries_2024",
            table_name="timeseriesvaluesandflags tsv",
            user_query=f"""
                JOIN {source}_timeseries_2024.timeseries ts ON ts.timeserieskey = tsv.timeserieskey
                JOIN {source}_timeseries_2024.location l ON l.locationkey = ts.locationkey
                JOIN {source}_timeseries_2024.parameter p ON p.parameterkey = ts.parameterkey
                WHERE l.locationkey={ditch_id}
            """,
        )
        ditch_stage.set_index("datetime", inplace=True)
        ditch_stage.rename(
            columns={"scalarvalue": name, "datetime": "time"}, inplace=True
        )
        ditch_df.append(ditch_stage)
    ditches_df = pd.concat(ditch_df)
    daily_ditches_df = ditches_df.resample("D").mean().astype(float)
    daily_ditches_df.index.name = "time"
    interpolated_ditches_df = interpolate_ditch_values(
        daily_ditches_df, method="linear"
    )
    interpolated_ditches_df.columns.name = "name"
    interpolated_ditches_da = interpolated_ditches_df.stack().to_xarray()
    interpolated_ditches_da = interpolated_ditches_da.transpose("name", "time")
    return interpolated_ditches_da


def interpolate_ditch_values(df, method) -> pd.DataFrame:
    """
    Function to interpolate missing values in the ditch stage DataFrame using the specified method. Is done only inside known values, no extrapolation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ditch stage data with potential missing values.
    method : str
        Interpolation method to use. Options include 'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'polynomial', 'barycentric', 'krog', 'pchip', and 'akima'. Check https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html for documentation.
    """
    return df.interpolate(method=method, limit_direction="both", limit_area="inside")
