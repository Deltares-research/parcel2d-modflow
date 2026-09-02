from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from pydantic import ValidationError

from parcel2d_modflow import modeldata, utils
from parcel2d_modflow.config import Config
from parcel2d_modflow.constants import BestKappa, ParameterCorrectionCurve
from parcel2d_modflow.exceptions import ConfigError
from parcel2d_modflow.io.soilmap import BroSoilmap
from parcel2d_modflow.modeldata import ModelData
from parcel2d_modflow.validation import validate_modflow_parameters, validate_soilmap

if TYPE_CHECKING:
    import geopandas as gpd


def read_config(config_file: str | Path) -> Config:
    """
    Read a TOML configuration file containing all settings and inputs for a modelling
    run. TODO: add explanation where to find all configurable options in the TOML.

    Parameters
    ----------
    config_file : str | Path
        Path to the TOML configuration file containing all settings, inputs and output for
        a modelling run.

    Returns
    -------
    :class:`~parcel2d_modflow.config.Config`
        Configuration istance containing all settings and inputs.

    """

    def _normalize_keys(obj: dict) -> dict:
        """
        Recursively replace dashes with underscores in dictionary keys of a parsed TOML
        file.

        """
        if isinstance(obj, dict):
            return {k.replace("-", "_"): _normalize_keys(v) for k, v in obj.items()}
        return obj

    with open(config_file, "rb") as f:
        toml = tomllib.load(f)

    try:
        return Config(**_normalize_keys(toml))
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration file: {e}")


def read_data_from_config(config: Config) -> ModelData:
    """
    Read all input data for a modelling run from the configuration object.

    Parameters
    ----------
    config : :class:`~parcel2d_modflow.config.Config`
        Configuration instance containing all input data for a modelling run.

    Returns
    -------
    :class:`~parcel2d_modflow.modeldata.ModelData`
        ModelData instance containing all input data for a modelling run.

    """
    parcels = read_parcels(config.data.parcels)
    gw_data = read_groundwater_data(
        confining_nc=config.data.confining_nc,
        flux_nc=config.data.flux_nc,
        recharge_nc=config.data.recharge_nc,
    )
    soilmap = read_bro_soilmap(config.data.soilmap_gpkg)
    parameters = read_modflow_parameters(config.modflow_settings.parameters)
    presets = read_presets(
        ditch_stage_nc=config.data.ditch_level_nc,
        ssi_stage_nc=config.data.ssi_stage_nc,
    )
    return ModelData(parcels, gw_data, soilmap, parameters, presets)


def read_parcels(file: str | Path, **gpd_kwargs) -> gpd.GeoDataFrame:
    """
    Read a shapefile or geoparquet file containing parcel data and return a GeoDataFrame.

    Parameters
    ----------
    parcels : str | Path
        Path to the parcel data file.
    **gpd_kwargs
        Keyword arguments passed to `geopandas.read_file` or `geopandas.read_parquet`.
        See the relevant Geopandas documentation for details.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the parcel data.

    """
    parcels = utils.geopandas_read(file, **gpd_kwargs)

    # We need to ensure that CRS is set to EPSG:28992 (Amersfoort / RD New) because of
    # the BRO Soilmap and the groundwater data, which are both in this CRS.
    if parcels.crs is None:
        parcels.set_crs(epsg=28992, inplace=True)
    elif parcels.crs != 28992:
        parcels.to_crs(epsg=28992, inplace=True)

    parcels["x"] = parcels.geometry.centroid.x
    parcels["y"] = parcels.geometry.centroid.y

    return parcels


def read_groundwater_data(
    confining_nc: str | Path = None,
    flux_nc: str | Path = None,
    recharge_nc: str | Path = None,
    head_nc: str | Path = None,
) -> modeldata.GroundwaterData:
    """
    Read NetCDF files containing confining layer, flux, recharge, and head data for the
    required groundwater data for modelling runs.

    Parameters
    ----------
    confining_nc : str | Path
        NetCDF file containing confining layer data.
    flux_nc : str | Path
        NetCDF file containing a time-series of flux data.
    recharge_nc : str | Path
        NetCDF file containing a time-series of recharge data.
    head_nc : str | Path
        NetCDF file containing a time-series of phreatic head data.

    Returns
    -------
    :class:`~parcel2d_modflow.modeldata.GroundwaterData`
        `GroundwaterData` instance containing the confining layer, flux, recharge, and head
        data.

    """
    confining = (
        xr.open_dataset(confining_nc)  # , engine="netcdf4")
        if confining_nc is not None
        else None
    )
    flux = xr.open_dataarray(flux_nc) if flux_nc is not None else None
    recharge = xr.open_dataarray(recharge_nc) if recharge_nc is not None else None
    head = xr.open_dataarray(head_nc) if head_nc is not None else None

    return modeldata.GroundwaterData(confining, flux, recharge, head)


def read_presets(
    ditch_stage_nc: str | Path = None,
    ssi_stage_nc: str | Path = None,
) -> modeldata.Presets:
    """
    Read NetCDF files containing ditch stage and piezometer head data for the required
    presets for modelling runs.

    Parameters
    ----------
    ditch_stage_nc : str | Path
        NetCDF file containing a time-series of ditch stage data.
    ssi_stage_nc : str | Path
        NetCDF file containing a time-series of SSI stage data.

    Returns
    -------
    :class:`~parcel2d_modflow.modeldata.Presets`
        `Presets` instance containing the ditch stage and SSI stage data.

    """
    ditch_stage = (
        xr.open_dataarray(ditch_stage_nc) if ditch_stage_nc is not None else None
    )
    ssi_stage = xr.open_dataarray(ssi_stage_nc) if ssi_stage_nc is not None else None

    return modeldata.Presets(ditch_stage=ditch_stage, pssi_stage=ssi_stage)


@validate_soilmap
def read_bro_soilmap(soilmap_path: str | Path, **gpd_kwargs) -> modeldata.Soilmap:
    """
    Read and merge the relevant tables from the BRO Soilmap into a `Soilmap` instance.

    The BRO Soilmap can be downloaded from PDOK with the following url:
    https://service.pdok.nl/bzk/bro-bodemkaart/atom/downloads/BRO_DownloadBodemkaart.gpkg

    Parameters
    ----------
    soilmap_path : str | Path
        Path to GeoPackage of the BRO Soilmap.
    gpd_kwargs
        `gpd.read_file` keyword arguments. See the relevant GeoPandas documentation.

    Returns
    -------
    :class:`~parcel2d_modflow.modeldata.Soilmap`
        A `Soilmap` dataclass containing geometries with `soilunit_code` attributes
        and a standardized table of soil profiles.

    """
    bro_soilmap = BroSoilmap.from_geopackage(soilmap_path, **gpd_kwargs)
    soilmap = bro_soilmap.create_soilmap_with_units()
    soilprofiles = bro_soilmap.create_soilprofile_table()

    id_code_mapping = soilmap[
        ["normalsoilprofile_id", "soilunit_code"]
    ].drop_duplicates()

    soilprofiles = soilprofiles.merge(
        id_code_mapping, on="normalsoilprofile_id", how="left"
    )
    soilprofiles["lithology"] = utils.determine_lithology_from(soilprofiles)
    soilprofiles["thickness"] = soilprofiles["uppervalue"] - soilprofiles["lowervalue"]

    to_fraction = 100
    soilprofiles["organicmattercontent"] /= to_fraction

    return modeldata.Soilmap(soilmap, soilprofiles)


@validate_modflow_parameters
def read_modflow_parameters(file: str | Path, **pd_kwargs) -> pd.DataFrame:
    """
    Read and validate the stochastic Modflow parameters for a Modflow model run from a
    CSV file.

    Parameters
    ----------
    file : str | Path
        Path to the CSV file containing the stochastic Modflow parameters.
    **pd_kwargs
        Keyword arguments passed to pandas read_csv. See the relevant pandas
        documentation.

    Returns
    -------
    pd.DataFrame

    """
    return pd.read_csv(file, **pd_kwargs)


def read_weather_data(
    weather_stations: str | Path,
    knmi_measurements: str | Path,
    weather_regions: str | Path,
    correction_params: dict[str, float] = None,
    kappa: dict[str, float] = None,
) -> modeldata.WeatherData:
    """
    Read all the required weather data for SOMERS modelling runs. This consists of weather
    stations, time series of temperature data, and weather regions.

    Parameters
    ----------
    weather_stations : str | Path
        Shapefile like file containing the weather stations.
    knmi_measurements : str | Path
        Text file containing the KNMI measurement data. The file should be downloaded from
        the KNMI website. The downloaded file should be converted to plain csv format without
        the header and comments that are present in the downloaded file. See the format below::

            STN,YYYYMMDD,TG,RH,...,EV24
            260,20220101,5.0,0.0,...,0.
    weather_regions : str | Path
        Shapefile like file containing the weather regions.
    correction_params : dict[str, float], optional
        Dictionary containing the correction parameters for temperature data. The default
        is None, then an instance of :class:`~parcel2d_modflow.constants.ParameterCorrectionCurve`
        with default values is used.
    kappa : dict[str, float], optional
        Dictionary containing best kappa parameters for the soil temperature module. The
        default is None, then an instance of :class:`~parcel2d_modflow.constants.BestKappa`
        with default values is used.

    Returns
    -------
    :class:`~parcel2d_modflow.modeldata.WeatherData`
        `WeatherData` instance containing the weather stations, temperature data, weather
        regions, correction parameters and kappa.

    """
    correction_params = ParameterCorrectionCurve(**(correction_params or {}))
    kappa = BestKappa(**(kappa or {}))

    stations = utils.geopandas_read(weather_stations)
    measurements = read_knmi_measurements(knmi_measurements)
    regions = utils.geopandas_read(weather_regions)

    regions.set_crs(stations.crs, inplace=True)
    stations = stations.sjoin(regions, predicate="within")

    return modeldata.WeatherData(
        stations, measurements, regions, correction_params, kappa
    )


def read_knmi_measurements(file: str | Path) -> pd.DataFrame:
    """
    Read downloaded KNMI measurement data from a text-file. Measurement data can be downloaded
    from the KNMI website from the following url: https://daggegevens.knmi.nl/klimatologie/daggegevens

    Parameters
    ----------
    file : str | Path
        Path to the KNMI temperature data text-file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the measurement data.

    """
    measurements = pd.read_csv(
        file,
        comment="#",
        skipinitialspace=True,
        parse_dates=["YYYYMMDD"],
        index_col="YYYYMMDD",
    )
    if "TG" in measurements.columns:
        to_degree_celsius = 10
        measurements["TG"] = measurements["TG"] / to_degree_celsius

    to_m_per_day = 1e4
    if "RH" in measurements.columns:
        # -1 is code for <0.05 mm precipitation
        measurements["RH"] = measurements["RH"].astype(float).replace(-1, 0.025)
        measurements["RH"] /= to_m_per_day
    if "EV24" in measurements.columns:
        measurements["EV24"] /= to_m_per_day

    return measurements
