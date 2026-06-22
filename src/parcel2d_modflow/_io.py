from __future__ import annotations

from pathlib import Path

import xarray as xr

from parcel2d_modflow import modeldata


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
