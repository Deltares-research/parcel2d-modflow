from pathlib import Path
from typing import Iterator

import geopandas as gpd
import numpy as np

from parcel2d_modflow.base import Parcel
from parcel2d_modflow.config import ModelSettings
from parcel2d_modflow.modeldata import Soilmap, WeatherData
from parcel2d_modflow.validation import validate_parcels


@validate_parcels
def prepare_parcels(
    parcels: gpd.GeoDataFrame,
    settings: ModelSettings,
    soilmap: Soilmap,
    weather: WeatherData = None,
) -> Iterator[Parcel]:
    """
    Prepare a GeoDataFrame with parcel information for a SOMERS modelling run. This returns
    a generator that yields :class:`~somers.base.Parcel` objects. The yielded `Parcel` objects
    contain discretized soil depth information and soil profile information. Also, for each
    parcel a dedicated directory is created in the workdir to store the results of the model
    run for the parcel.

    Parameters
    ----------
    parcels : gpd.GeoDataFrame
        GeoDataFrame with parcel information to prepare the `Parcel` objects for.
    settings : :class:`~somers.base.ModelSettings`
        General model settings for the SOMERS model run.
    soilmap : :class:`~somers.modeldata.Soilmap`
        Soilmap data container to select all relevant soilmap information for each parcel
        from.
    weather : :class:`~somers.modeldata.WeatherData`
        Weather data container containing locations of KNMI weather stations to select the
        station for each parcel.

    Returns
    -------
    Iterator[Parcel]
        Generator that yields :class:`~somers.base.Parcel` objects.

    Yields
    ------
    :class:`~somers.base.Parcel`

    Examples
    --------
    Prepare the parcels by reading the required input data and finally using:

    >>> parcels = gpd.read_file("path/to/parcels.shp")
    ... settings = ModelSettings("path/to/workdir", "2022-01-01", "2022-12-31")
    ... soilmap = read_soilmap("path/to/soilmap.gpkg")
    ... weather = read_weather_data("path/to/weather.shp", "path/to/weather.txt")
    ... prepared_parcels = prepare_parcels(parcels, settings, soilmap, weather)

    This returns a generator that yields parcels. You can loop over the parcels and print
    the name of each parcel:

    >>> for parcel in prepared_parcels: # Loop over parcels
    ...     print(parcel.name)

    """
    parcels = parcels.copy()

    if weather is not None and not {"nearest_weather_station", "weather_rg"}.issubset(parcels.columns):
        parcels = add_weather_station_info(parcels, weather.stations)

    parcel_attributes = parcels.columns
    for p in parcels.itertuples(index=False):
        temp_dir_name = p.name + "_" + p.soilcode
        Path(settings.workdir / temp_dir_name).mkdir(exist_ok=True)
        parcel = Parcel(**dict(zip(parcel_attributes, p)))

        if parcel.soilcode is None:
            parcel.soilcode = soilmap.soilcode_at(parcel.x, parcel.y)

        parcel.discretize_soildepth(settings)
        parcel.soilprofile = soilmap.load_soilprofile(parcel)
        yield parcel


def add_weather_station_info(
    parcels: gpd.GeoDataFrame, stations: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Add the nearest weather station and the associated weather region to a GeoDataFrame of
    parcels. The nearest weather station is determined by the Euclidean distance between the
    centroid of the parcel and the location of each weather station.

    Parameters
    ----------
    parcels : gpd.GeoDataFrame
        GeoDataFrame containing the parcels.
    stations : gpd.GeoDataFrame
        GeoDataFrame containing the weather stations. The GeoDataFrame should have columns
        "id" and "weather_rg" that contain the ID and weather region the station lies in,
        respectively.

    Returns
    -------
    gpd.GeoDataFrame
        The input GeoDataFrame with two new columns: "nearest_weather_station" and "weather_rg",
        containing the name of the closest weather station to each parcel and each associated
        weather region, respectively.

    """
    index = closest_weather_station(parcels, stations)
    parcels["nearest_weather_station"] = stations["id"].values[index]
    parcels["weather_rg"] = stations["weather_rg"].values[index]
    return parcels


def closest_weather_station(
    parcels: gpd.GeoDataFrame, stations: gpd.GeoDataFrame
) -> np.ndarray:
    """
    Find the closest weather station to each parcel. The closest weather station is determined
    by the Euclidean distance between the centroid of the parcel and the location of each
    weather station.

    Parameters
    ----------
    parcels : gpd.GeoDataFrame
        GeoDataFrame containing the parcels.
    stations : gpd.GeoDataFrame
        GeoDataFrame containing the weather stations. The GeoDataFrame should have columns
        "id" and "weather_rg" that contain the ID and weather region the station lies in,
        respectively.

    Returns
    -------
    np.ndarray
        Array with the indices of the closest weather station to each parcel.

    """
    from scipy.spatial import KDTree

    if "x" not in parcels or "y" not in parcels:
        centroids = parcels.centroid
        p1 = np.c_[centroids.x, centroids.y]
    else:
        p1 = np.c_[parcels["x"], parcels["y"]]

    tree = KDTree(np.c_[stations["geometry"].x, stations["geometry"].y])
    _, index = tree.query(p1)

    return index


def parcel_check(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardizes and prepares a GeoDataFrame of parcels for analysis.
    This function ensures that the column names are lowercase, sets the CRS to EPSG:28992
    if not already set, and adds centroid coordinates ('x', 'y') as columns if they do not
    exist.

    Parameters
    ----------
    parcels : gpd.GeoDataFrame
        A GeoDataFrame containing parcel geometries and attributes.

    Returns
    -------
    gpd.GeoDataFrame
        The input GeoDataFrame with standardized columns, CRS set to EPSG:28992, and centroid
        coordinates added.

    """
    parcels.columns = parcels.columns.str.lower()

    if parcels.crs is None:
        parcels.set_crs(epsg=28992, inplace=True)
    elif parcels.crs != 28992:
        parcels.to_crs(epsg=28992, inplace=True)

    if "x" not in parcels.columns:
        parcels["x"] = parcels["geometry"].centroid.x
    if "y" not in parcels.columns:
        parcels["y"] = parcels["geometry"].centroid.y

    return parcels