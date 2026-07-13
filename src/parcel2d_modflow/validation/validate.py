import warnings
from functools import wraps

import pandas as pd

from parcel2d_modflow._exceptions import ValidationError
from parcel2d_modflow.validation import helpers


class ValidationWarning(Warning):
    """Warning raised when a validation check fails but does not prevent further processing."""

    pass


def validate_parcels(func):
    """
    Validate the input parcels GeoDataFrame before processing. This checks if the mandatory
    columns are present in the DataFrame and selects the columns from the GeoDataFrame that
    can be used for processing. Raises a ValidationWarning for the columns that cannot be
    used.
    """

    @wraps(func)
    def wrapper(parcels, *args, **kwargs):
        mandatory_columns = {"name", "x", "y", "width", "surface_level", "soilcode"}
        if missing_mandatory_columns := helpers.check_mandatory_columns(
            parcels, mandatory_columns
        ):
            raise ValidationError(
                "Parcels DataFrame is missing mandatory columns: "
                f"{', '.join(missing_mandatory_columns)}."
            )

        valid_attrs, unknown_attrs = helpers.check_attributes(
            parcels.columns.drop("geometry", errors="ignore"),
            {
                "name",
                "x",
                "y",
                "width",
                "surface_level",
                "soilcode",
                "summer_stage",
                "winter_stage",
                "trench_depth",
                "trench_locations",
                "drain_depth",
                "drain_distance",
                "pssi_summer_stage",
                "pssi_winter_stage",
                "nearest_weather_station",
                "weather_rg",
                "soilprofile",
            },
        )
        if unknown_attrs:
            warnings.warn(
                (
                    "Input parcels GeoDataFrame contains unknown parcel attributes: "
                    f"{', '.join(unknown_attrs)}.\n Only the following attributes will be "
                    f"used for running the model: \n\n{', '.join(valid_attrs)}"
                ),
                ValidationWarning,
            )
        return func(parcels[valid_attrs], *args, **kwargs)

    return wrapper


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

        valid_upper_bounds = helpers.validate_min_max(
            sp.soilprofiles["lowervalue"].round(2), 0.0, 1.2
        ).all()
        if not valid_upper_bounds:
            errors.append(ValueError("Lowervalues soilprofiles are not valid."))

        valid_lower_bounds = helpers.validate_min_max(
            sp.soilprofiles["uppervalue"].round(2), 0.0, 1.20
        ).all()
        if not valid_lower_bounds:
            errors.append(ValueError("Uppervalues soilprofiles are not valid."))

        valid_organic_matter = helpers.validate_min_max(
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
