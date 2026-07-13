from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def check_mandatory_columns(df: pd.DataFrame, columns: set[str]) -> None:
    """
    Check if a DataFrame contains mandatory columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    columns : set[str]
        Set of mandatory column names.

    Returns
    -------
    List of the mandatory columns that are not present in the DataFrame. Returns an
    empty list if all mandatory columns are present.

    """
    return [c for c in columns if c not in df.columns]


def check_attributes(
    attributes: Iterable[str], check_attributes: Iterable[str]
) -> tuple[list[str], list[str]]:
    """
    Check if attributes are in the check_attributes set.

    Parameters
    ----------
    attributes : Iterable[str]
        Attributes to check.
    check_attributes : Iterable[str]
        Set of attributes to check against.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing two lists:
        - The first list contains attributes that are in the check_attributes set.
        - The second list contains attributes that are not in the check_attributes set.

    """
    overlapping = [attr for attr in attributes if attr in check_attributes]
    nonoverlapping = [attr for attr in attributes if attr not in check_attributes]
    return overlapping, nonoverlapping


def validate_min_max(array: pd.Series | np.ndarray, min_: float, max_: float) -> bool:
    """
    Validate if all values in an array are between a minimum and maximum value.

    Parameters
    ----------
    array : pd.Series | np.array
        Array with values to validate.
    min : float
        Minimum value of the array.
    max : float
        Maximum value of the array.

    Returns
    -------
    bool
        True, if all values are between the minimum and maximum value.

    """
    return (array >= min_) & (array <= max_)
