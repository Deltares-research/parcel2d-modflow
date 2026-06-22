from functools import wraps
from typing import TYPE_CHECKING

import pandas as pd

from parcel2d_modflow._exceptions import ValidationError

if TYPE_CHECKING:
    import numpy as np


def validate_modflow_parameters(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        mf = func(*args, **kwargs)
        missing_cols = [
            c
            for c in ["runnr", "kh (m/d)", "sy_peat (-)", "sy_clay (-)"]
            if c not in mf.columns
        ]
        if missing_cols:
            raise ValidationError(
                f"Modflow parameters DataFrame is missing columns: {missing_cols}"
            )

        correct_index = pd.RangeIndex(start=0, stop=len(mf), step=1)
        if not mf.index.equals(correct_index):
            raise ValidationError(
                f"Index of modflow parameters DataFrame is not correct. Expected "
                f"a RangeIndex starting from 0 with step 1, but got {mf.index}. "
                "Use index_col=None when reading the modflow parameters CSV file to use "
                "a default Pandas RangeIndex starting at 0."
            )
        return mf

    return wrapper


def validate_soilmap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        sp = func(*args, **kwargs)

        errors = []

        valid_upper_bounds = validate_min_max(
            sp.soilprofiles["lowervalue"].round(2), 0.0, 1.2
        ).all()
        if not valid_upper_bounds:
            errors.append(ValueError("Lowervalues soilprofiles are not valid."))

        valid_lower_bounds = validate_min_max(
            sp.soilprofiles["uppervalue"].round(2), 0.0, 1.20
        ).all()
        if not valid_lower_bounds:
            errors.append(ValueError("Uppervalues soilprofiles are not valid."))

        valid_organic_matter = validate_min_max(
            sp.soilprofiles["organicmattercontent"].round(2), 0.0, 1.0
        ).all()
        if not valid_organic_matter:
            errors.append(
                ValueError("Organic matter content soilprofiles are not valid.")
            )

        if errors:
            raise ValidationError(errors)
        return sp

    return wrapper


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
