import pandas as pd

from parcel2d_modflow.io.postgis import read_timeseries_from_database


def load_parcel_piezometers_from_db(gdf, connection) -> pd.DataFrame:
    """
    Load piezometer stage data for the piezometers associated with the parcels in the provided GeoDataFrame

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing parcel information, including piezometer identifiers.
    connection : sqlalchemy.engine.Engine
        SQLAlchemy engine for connecting to the database.

    Returns
    -------
    pd.DataFrame
        DataFrame containing piezometer stage data for the associated piezometers, resampled to daily frequency and interpolated.
    """
    piez_df = []
    for source, piez_id in gdf.well_id.str.split("_"):
        piez_stage = read_timeseries_from_database(
            engine=connection,
            select_name="tsv.scalarvalue, tsv.datetime",
            schema_name=f"{source}_timeseries_2024",
            table_name="timeseriesvaluesandflags tsv",
            user_query=f"""
                JOIN {source}_timeseries_2024.timeseries ts ON ts.timeserieskey = tsv.timeserieskey
                JOIN {source}_timeseries_2024.location l ON l.locationkey = ts.locationkey
                WHERE l.locationkey={piez_id}
            """,
        )
        piez_stage.set_index("datetime", inplace=True)
        piez_stage.rename(
            columns={"scalarvalue": f"{source}_{piez_id}", "datetime": "time"},
            inplace=True,
        )
        piez_df.append(piez_stage)
    piezs_df = pd.concat(piez_df)
    daily_piezs_df = piezs_df.resample("D").mean().astype(float)
    daily_piezs_df.index.name = "time"
    daily_piezs_df.columns.name = "well_id"
    daily_piezs_da = daily_piezs_df.stack().to_xarray()
    return daily_piezs_da
