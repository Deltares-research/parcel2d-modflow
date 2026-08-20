# %%
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


def select_time_range(
    parcel_df: pd.DataFrame,
    flux_data: Optional[Any],
    recharge_data: Optional[Any],
    weather_data: Optional[Any],
) -> pd.DataFrame:
    """
    Select the valid time range based on available data sources.

    Parameters
    ----------
    parcel_df : pd.DataFrame
        Parcel data with start_time and end_time columns
    flux_data : object or None
        Flux data object containing flux time series
    recharge_data : object or None
        Recharge data object containing recharge time series
    weather_data : object or None
        Weather data object containing precipitation and evapotranspiration attributes
    Returns
    -------
    pd.DataFrame
        Parcel data with start_time and end_time columns representing the valid time range
    """

    start_time = pd.to_datetime(parcel_df.start_date)
    end_time = pd.to_datetime(parcel_df.end_date)

    if flux_data is not None:
        start_time, end_time = update_time_range(flux_data, start_time, end_time)
    if recharge_data is not None:
        start_time, end_time = update_time_range(recharge_data, start_time, end_time)
    if weather_data is not None:
        weather_data.rename(columns={"YYYYMMDD": "time"}, inplace=True)
        start_time, end_time = update_time_range(weather_data, start_time, end_time)
    parcel_df["start_date"] = start_time
    parcel_df["end_date"] = end_time
    return parcel_df


def update_time_range(
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
