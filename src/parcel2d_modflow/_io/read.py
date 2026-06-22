from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from parcel2d_modflow import modeldata, utils
from parcel2d_modflow._io.soilmap import BroSoilmap
from parcel2d_modflow.validation import validate_modflow_parameters, validate_soilmap


def read_lhm_data(
    confining_nc: str | Path = None,
    flux_nc: str | Path = None,
    recharge_nc: str | Path = None,
    head_nc: str | Path = None,
) -> modeldata.LhmData:
    """
    Read NetCDF files containing confining layer, flux, and recharge data for the required
    LHM data for SOMERS modelling runs.

    Parameters
    ----------
    confining_nc : str | Path
        NetCDF file containing LHM confining layer data.
    flux_nc : str | Path
        NetCDF file containing LHM flux data.
    recharge_nc : str | Path
        NetCDF file containing LHM recharge data.
    head_nc : str | Path
        NetCDF file containing LHM phreatic head data.

    Returns
    -------
    :class:`~parcel2d_modflow.modeldata.LhmData`
        `LhmData` instance containing the confining layer, flux, and recharge data.

    """
    confining = (
        xr.open_dataset(confining_nc)  # , engine="netcdf4")
        if confining_nc is not None
        else None
    )
    flux = xr.open_dataarray(flux_nc) if flux_nc is not None else None
    recharge = xr.open_dataarray(recharge_nc) if recharge_nc is not None else None
    head = xr.open_dataarray(head_nc) if head_nc is not None else None

    return modeldata.LhmData(confining, flux, recharge, head)


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
