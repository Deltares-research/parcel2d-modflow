from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from parcel2d_modflow import utils

if TYPE_CHECKING:
    import numpy as np

type Dimension = Literal["1D", "2D"]


class Config(BaseModel):
    settings: ModelSettings
    modflow_settings: ModflowSettings
    run_settings: RunSettings
    data: InputData
    output: OutputSettings


class ModelSettings(BaseModel):
    """
    General settings for the SOMERS model. This includes the working directory, start and
    end date of the simulation and settings that are used for the discretization of the
    model.

    Parameters
    ----------
    workdir : str | Path
        Working directory for the model. This is where the output and modelling files for
        individual parcels are saved.
    start_date : pd.Timestamp
        Start date of the simulation. Required to be a :class:`~datetime.date` or
        :class:`~pd.Timestamp` object".
    end_date : pd.Timestamp
        End date of the simulation. Required to be a :class:`~datetime.date` or
        :class:`~pd.Timestamp` object".
    include_leap_days : bool, optional
        Whether to include leap days in the date range of the simulation. The default is
        True.
    stress_frequency : str, optional
        Frequency of the stress periods accepted by :func:`pd.to_datetime`. This is used
        to discretize the modelling period between the start and end dates. See relevant
        Pandas documentation for options. The default is "d" (daily).
    summer_start : int, optional
        Integer value of the month where the summer starts. The default is 4 (April).
    winter_start : int, optional
        Integer value of the month where the winter starts. The default is 10 (October).
    dimension : str, optional
        Model dimension: 1D or 2D. The default is "2D".
    ditch_depth : int | float, optional
        Depth of a ditch in meters. The default is 0.7.
    add_trenches : bool, optional
        Add trenches in the 2D section of a Modflow model. The default is False. If True,
        the `trench_depth` and `trench_locations` parameters must be available in a
        :class:`~somers.base.Parcel` instance.
    trench_resistance : int | float, optional
        Resistance of a trench in days. The default is 1.0.
    min_drain_depth : int | float, optional
        Minimum drainage depth for SSI or PSSI measure in meters. The default is 0.2.
    soilprofile_thickness : int | float, optional
        Thickness of the soil profile in meters. The default is 1.2, this corresponds with
        the traditional depth of soil profiles in the Dutch soil map.
    soil_layer_thickness : int | float, optional
        Layer thickness (in meters) in which the soil profile depth interval is discretized
        in. The default is 0.05.
    dx : int | float, optional
        Horizontal discretization of the model in meters when the modelling dimension is
        2D. The default is 0.5.
    dz_resistance_layer : int | float, optional
        Thickness of the resistance layer in meters used in the Modflow groundwater model
        at the top of the aquifer. The default is 0.5.
    save_flopy : bool, optional
        Save the `Flopy` model files in the working directory. The default is False.
    save_phreatic_head : bool, optional
        Save the phreatic head results in a NetCDF file in the working directory. The
        default is False.
    clean_workdir : bool, optional
        Clean the working directory after running the model. The default is False. You
        may want to use this when running many parcels to avoid filling up the disk space
        with temporary Modflow input and output files.

    Raises
    ------
    TypeError
        If ``start_date`` or ``end_date`` are not a :class:`~pd.Timestamp` object.
    ValidationError
        If ``start_date`` is after ``end_date``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    workdir: str | Path = Field(default_factory=utils.create_workdir)
    start_date: datetime.date | pd.Timestamp
    end_date: datetime.date | pd.Timestamp
    stress_frequency: str = "D"
    include_leap_days: bool = True
    summer_start: int = Field(default=4, ge=1, le=12)
    winter_start: int = Field(default=10, ge=1, le=12)
    dimension: Dimension = "2D"
    ditch_depth: int | float = Field(default=0.7, ge=0.0)
    ditch_resistance: int | float = Field(default=1.0, ge=0.0)
    min_water_depth: int | float = Field(default=0.4, ge=0.0)
    add_trenches: bool = False
    trench_resistance: int | float = Field(default=1.0, ge=0.0)
    min_drain_depth: int | float = Field(default=0.2, ge=0.0)
    soilprofile_thickness: int | float = Field(default=1.2, ge=0.0, le=1.2)
    soil_layer_thickness: int | float = Field(default=0.05, gt=0.0)
    dx: int | float = Field(default=0.5, gt=0.0)
    dz_resistance_layer: int | float = Field(default=0.5, gt=0.0)
    save_flopy: bool = False
    save_phreatic_head: bool = False
    clean_workdir: bool = False

    @field_validator("workdir", mode="before")
    @classmethod
    def _cast_to_path(cls, v):
        return Path(v)

    @field_validator("start_date", "end_date", mode="after")
    @classmethod
    def _cast_to_timestamp(cls, v):
        if isinstance(v, datetime.date):
            return pd.Timestamp(v)
        elif isinstance(v, str):
            try:
                v = datetime.datetime.fromisoformat(v)
                return pd.Timestamp(v)
            except ValueError:
                return v  # Let Pydantic handle the validation error for invalid date strings
        return v

    @model_validator(mode="after")
    def _validate_dates(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date.")
        return self

    def _create_date_range(self) -> pd.DatetimeIndex:
        """
        Create a date range for the modelling period based on the start and end dates and
        the stress frequency. This is used in the `date_range` property. Leap days are
        included or excluded based on the `include_leap_days` attribute.

        Returns
        -------
        pd.DatetimeIndex
            Date range from start_date to end_date with frequency defined by stress_frequency.

        """
        date_range = pd.date_range(
            self.start_date,
            self.end_date,
            freq=self.stress_frequency,
            name="time",
        )
        if not self.include_leap_days:
            date_range = date_range[~((date_range.month == 2) & (date_range.day == 29))]

        self._date_range = date_range

    @property
    def date_range(self) -> pd.DatetimeIndex:
        """
        Date range of the modelling period derived from the specified "start_date" and
        "end_date" attributes. Leap days are included or excluded based on the
        `include_leap_days` attribute.

        Returns
        -------
        pd.DatetimeIndex
            Date range from "start_date" to "end_date" with frequency defined by the
            "stress_frequency" attribute.

        """
        if not hasattr(self, "_date_range"):
            self._create_date_range()
        return self._date_range

    def _get_winter_period(self) -> np.ndarray:
        """
        Create a boolean array which indicates dates that are in winter. This is derived
        from the "date_range" and the specified "summer_start" and "winter_start
        attributes.

        """
        self._winter_period = (self.date_range.month < self.summer_start) | (
            self.date_range.month >= self.winter_start
        )

    @property
    def winter_period(self) -> np.ndarray:
        """
        Boolean array which indicates dates that are in winter. Is derived from the
        "date_range" and the specified "summer_start" and "winter_start attributes of
        the `ModelSettings` instance.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates that the corresponding date in the
            `date_range` attribute is in the winter period.

        """
        if not hasattr(self, "_winter_period"):
            self._get_winter_period()
        return self._winter_period


class ModflowSettings(BaseModel):
    modflow_executable: Path
    parameters: Path
    aquifer_method: Literal["flux"] = "flux"
    measure: Literal["ref", "ssi", "pssi"] = "ref"
    modflow_kwargs: dict[str, Any] = Field(default_factory=dict)


class InputData(BaseModel):
    parcels: Path
    confining_nc: Path
    flux_nc: Path
    recharge_nc: Path
    soilmap_gpkg: Path


class OutputSettings(BaseModel):
    directory: Path
    file: Path | None = None
    format: Literal["parquet", "geoparquet"] = "geoparquet"
    prefix: str = "batch"

    @model_validator(mode="after")
    def _validate_file_format(self):
        if self.file is not None:
            file_format = self.file.suffix.lstrip(".").lower()
            if file_format not in {"parquet", "geoparquet"}:
                raise ValueError("File extension must be parquet or geoparquet.")

        return self


class RunSettings(BaseModel):
    multiprocessing: bool = True
    batch_size: int = 100
    multiprocess_scale: float = Field(default=1.0, ge=0.0, le=1.0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
