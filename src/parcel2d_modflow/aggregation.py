import numpy as np
import xarray as xr


def calculate_mean(head: xr.DataArray):
    raise NotImplementedError("This function is not yet implemented.")


def calculate_lg3(head: xr.DataArray):
    """
    Calculate the mean of the lowest 3 groundwater levels per year from the modelled
    daily time series of phreatic heads.

    Parameters
    ----------
    head : xr.DataArray
        Modelled daily time series of phreatic heads.

    Returns
    -------
    xr.DataArray
        DataArray with dimensions (runs, year) containing the mean of the lowest 3
        groundwater levels per year.

    """

    def mean_lowest3(da: xr.DataArray) -> xr.DataArray:
        time_axis = da.get_axis_num("time")
        lowest3 = da.copy(data=np.partition(da.data, 3, axis=time_axis)).isel(
            time=slice(3)
        )
        return lowest3.mean(dim=("time", "x"))

    return head.groupby("time.year").map(mean_lowest3)
