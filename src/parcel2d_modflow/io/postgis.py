import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import sqlalchemy
from dotenv import find_dotenv, load_dotenv


def initialize_database(dotenv_path: str | Path = None) -> sqlalchemy.engine.Engine:
    """
    Establish a SQLAlchemy engine for connecting to a PostgreSQL database.

    Parameters
    ----------
    dotenv_path : str or Path, optional
        Path to the .env file containing database connection parameters. If None, the
        function will attempt to find a .env file in the current working directory. The
        default is None.

    Returns
    -------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine connected to the PostgreSQL database.

    Raises
    ------
    FileNotFoundError
        If the .env file cannot be found or loaded.

    """
    dotenv_path = dotenv_path or find_dotenv(usecwd=True)

    if not load_dotenv(dotenv_path):
        raise FileNotFoundError(
            """
            Failed to load database connection variables from .env file. Please ensure
            a .env file exists in the current working directory or give a valid path to
            the .env file to load the database connection parameters.

            Make sure the .env file has the following contents:

            DB_HOST=your_database_host
            PORT=your_database_port
            DB_USER=your_database_user
            DB_PASSWORD=your_database_password
            DB_NAME=your_database_name
            """
        )

    host = os.environ["DB_HOST"]
    port = int(os.environ["PORT"])
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    database = os.environ["DB_NAME"]

    engine = sqlalchemy.create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
    )

    return engine


def read_table_from_database(
    engine: sqlalchemy.engine.Engine,
    schema_name: str,
    table_name: str,
    user_query: str = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Read a table from the database with additional filtering based on a user-defined
    SQL query.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine for connecting to the database. See also `somers.io.initialize_somers_database`.
    schema_name : str
        Name of the database schema containing the output table.
    table_name : str
        Name of the database table containing output data for filtering.
    user_query : str, default None
        SQL query defining additional filtering logic.
    **kwargs
        Additional keyword arguments passed to `geopandas.read_postgis`.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing entries that meet the query.
    """
    query = f"SELECT * FROM {schema_name}.{table_name}" + (
        f" {user_query}" if user_query else ""
    )

    with engine.connect() as connection:
        table_data = gpd.read_postgis(
            query,
            con=connection,
            geom_col="geometry",
            **kwargs,
        )

    return table_data


def read_timeseries_from_database(
    engine: sqlalchemy.engine.Engine,
    select_name: str,
    schema_name: str,
    table_name: str,
    user_query: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a table from the database with additional filtering based on a user-defined
    SQL query.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine for connecting to the database. See also `somers.io.initialize_somers_database`.
    select_name: str
        Name of the column to select from the d atabase table.
    schema_name : str
        Name of the database schema containing the output table.
    table_name : str
        Name of the database table containing output data for filtering.
    user_query : str, default None
        SQL query defining additional filtering logic.

    Returns
    -------
    pd.DataFrame
        DataFrame containing entries that meet the query.
    """
    query = f"SELECT {select_name} FROM {schema_name}.{table_name}" + (
        f" {user_query}" if user_query else ""
    )

    with engine.connect() as connection:
        table_data = pd.read_sql_query(query, con=connection)

    return table_data
