# %%
import warnings
from typing import Any, Optional, Sequence

import pandas as pd
import xarray as xr


def select_time_range(
    parcel_df: pd.DataFrame,
    flux_data: Optional[Any],
    recharge_data: Optional[Any],
    weather_data: Optional[Any],
    piez_da: Optional[Any],
    ditch_da: Optional[Any],
) -> pd.DataFrame:
    """
    Select the valid time range based on available data sources.

    Parameters
    ----------
    parcel_df : pd.DataFrame
        Parcel data with start_time and end_time columns
    flux_data : xr.DataArray or None
        Flux data object containing flux time series
    recharge_data : xr.DataArray or None
        Recharge data object containing recharge time series
    weather_data : pd.DataFrame or None
        Weather data object containing precipitation and evapotranspiration attributes
    piez_da : xr.DataArray or None
        Piezometer data array containing piezometer measurements
    ditch_da : xr.DataArray or None
        Ditch data array containing ditch measurements
    Returns
    -------
    pd.DataFrame
        Parcel data with start_time and end_time columns representing the valid time range
    """

    start_time = pd.to_datetime(parcel_df.start_date)
    end_time = pd.to_datetime(parcel_df.end_date)

    if flux_data is not None:
        start_time, end_time = update_time_range_from_inputdata(
            flux_data, start_time, end_time
        )
    if recharge_data is not None:
        start_time, end_time = update_time_range_from_inputdata(
            recharge_data, start_time, end_time
        )
    if weather_data is not None:
        weather_data.rename(columns={"YYYYMMDD": "time"}, inplace=True)
        start_time, end_time = update_time_range_from_inputdata(
            weather_data, start_time, end_time
        )
    if piez_da is not None:
        start_time, end_time = update_time_range_from_measurements(
            parcel_df, piez_da, "well_id", start_time, end_time
        )
    if ditch_da is not None:
        start_time, end_time = update_time_range_from_measurements(
            parcel_df, ditch_da, "name", start_time, end_time
        )

    parcel_df["start_date"] = start_time.dt.normalize()
    parcel_df["end_date"] = end_time.dt.normalize()
    return parcel_df


def update_time_range_from_measurements(
    parcel_df: pd.DataFrame,
    measurements_da: xr.DataArray,
    meas_name_column: str,
    start_time: pd.Series,
    end_time: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Update time range based on available measurements data.
    """
    if meas_name_column not in parcel_df.columns:
        raise ValueError(f"{meas_name_column} not found in parcel_df columns")
    for aan_id, meas_id in parcel_df[meas_name_column].items():
        if isinstance(meas_id, set):
            meas_id = list(meas_id)
        measurement_data = measurements_da.sel({meas_name_column: meas_id})
        measurement_data = measurement_data.dropna(dim="time", how="all")
        if measurement_data.size == 0:
            warnings.warn(
                f"No measurements found for {meas_name_column} '{meas_id}' in parcel '{aan_id}'"
            )
            continue
        st = pd.Timestamp(measurement_data.time.min().values)
        et = pd.Timestamp(measurement_data.time.max().values)
        start_time[aan_id] = max(start_time[aan_id], st)
        end_time[aan_id] = min(end_time[aan_id], et)
    return start_time, end_time


def update_time_range_from_inputdata(
    data: Any,
    start_time: pd.Series,
    end_time: pd.Series,
    attrs: Optional[Sequence[str]] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Update time range based on available data attributes.

    Parameters
    ----------
    data : object
        Data object containing time series attributes
    start_time : pd.Series
        Current latest start time
    end_time : pd.Series
        Current earliest end time
    attrs : list, optional
        List of attribute names to extract time bounds from

    Returns
    -------
    tuple
        Tuple of (start_time, end_time) as pd.Series objects
    """
    if attrs:
        if not all(hasattr(data, attr) for attr in attrs):
            missing_attrs = [attr for attr in attrs if not hasattr(data, attr)]
            raise AttributeError(f"Data object is missing attributes: {missing_attrs}")
        for attr in attrs:
            attr_data = getattr(data, attr)
            attr_time = pd.to_datetime(attr_data.time)
            start_time = start_time.combine(
                pd.Series(attr_time.min(), index=start_time.index), max
            )
            end_time = end_time.combine(
                pd.Series(attr_time.max(), index=end_time.index), min
            )
    else:
        data_time = pd.to_datetime(data.time)
        start_time = start_time.combine(
            pd.Series(data_time.min(), index=start_time.index), max
        )
        end_time = end_time.combine(
            pd.Series(data_time.max(), index=end_time.index), min
        )
    return start_time, end_time


# %%
