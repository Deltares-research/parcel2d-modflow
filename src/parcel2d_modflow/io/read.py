from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from pydantic import ValidationError

from parcel2d_modflow import modeldata, utils
from parcel2d_modflow.config import Config
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
        ditch_stage_nc=config.data.ditchlvl_nc,
        ssi_stage_nc=config.data.ssi_stage_nc,
    )
    # Read presets if available
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


if __name__ == "__main__":
    config_file = r"c:\src\somers\parcel2d-modflow\dev\config_parcel2d.toml"
    config = read_config(config_file)
    print()
