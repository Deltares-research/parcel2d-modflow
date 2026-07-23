import numpy as np
import pandas as pd
import xarray as xr


def calculate_mean(head: xr.DataArray):
    raise NotImplementedError("This function is not yet implemented.")


def calculate_lg3(head: xr.DataArray):
    def mean_lowest3(da: xr.DataArray) -> xr.DataArray:
        time_axis = da.get_axis_num("time")
        lowest3 = da.copy(data=np.partition(da.data, 3, axis=time_axis)).isel(
            time=slice(3)
        )
        return lowest3.mean(dim=("time", "x"))

    lg3 = head.groupby("time.year").map(mean_lowest3)

    return lg3.to_dataframe(name="phreatic_head")
