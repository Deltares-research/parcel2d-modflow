from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from parcel2d_modflow import components, utils

type Dimension = Literal["1D", "2D"]


class AbstractModule(ABC):
    """
    Abstract base class for modelling components in SOMERS. Inheritance makes sure that
    all subclasses have an initialize and run method.

    Attributes
    ----------
    available_modules : set
        Set of all available modules in SOMERS. Each module that inherits from `AbstractModule`
        is automatically added to the set. This is used to check if a module is valid.
    """

    available_modules = set()

    def __init_subclass__(cls, **kwargs):
        """
        Register any Module subclass in SOMERS that inherits from this abstract base class.
        """
        super().__init_subclass__(**kwargs)
        cls.available_modules.add(cls.__name__)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @classmethod
    def is_valid(cls, module: str) -> bool:
        """
        Check if a module is valid Model component in SOMERS.

        Parameters
        ----------
        module : str
            Name of module to check.

        Returns
        -------
        bool
            True, if the module exists in the registry.

        """
        return module in cls.available_modules


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


class Parcel:
    """
    Container with all relevant information of each parcel that is needed for SOMERS runs.

    Parameters
    ----------
    name : str
        Name of the parcel.
    x : int | float
        x-coordinate (m) of the parcel in Rijksdriehoekstelsel coordinates.
    y : int | float
        y-coordinate (m) of the parcel in Rijksdriehoekstelsel coordinates.
    width : int | float
        Width of the parcel (m).
    surface_level : int | float
        Surface level of the parcel (m +NAP).
    soilcode : str, optional
        Soil code of the parcel as derived from the BRO Bodemkaart. The default is None,
        then it must be derived by location from the BRO Bodemkaart.
    summer_stage : int | float, optional
        Summer stage of the parcel in meters +NAP. The default is None.
    winter_stage : int | float, optional
        Winter stage of the parcel in meters +NAP. The default is None.
    trench_depth: int | float, optional
        Depth of trenches in meters. Must be a value between 0.1 and 0.8 meters. The default
        is None.
    trench_locations : int | array-like, optional
        Locations of the trenches in the parcel. This can be an integer value for the number
        of trenches or an array-like object with the locations of the trenches in meters. If
        the input is an integer, the trenches are evenly distributed over the width of a
        parcel.
    drain_depth : int | float, optional
        Depth of a drain in meters below surface level, used with SSI or PSSI measure. Depth
        must be between 0 and 1.2 meters below surface level. The default is None.
    drain_distance : int | float, optional
        Distance between drains in meters. Drain distance must be between 3 and 10 meters.
        The default is None.
    pssi_summer_stage : int | float, optional
        Summer stage of the parcel for PSSI measure in meters +NAP). The default is None.
    pssi_winter_stage : int | float, optional
        Winter stage of the parcel for PSSI measure in meters +NAP. The default is None.
    nearest_weather_station : int, optional
        ID of the nearest weather station to the parcel. The default is None.
    weather_rg : str, optional
        Weather region the parcel is located in. The default is None.
    """

    def __init__(
        self,
        name: str,
        x: int | float,
        y: int | float,
        width: int | float,
        surface_level: int | float,
        soilcode: str = None,
        summer_stage: int | float = None,
        winter_stage: int | float = None,
        trench_depth: int | float = None,
        trench_locations: int | np.ndarray = None,
        drain_depth: int | float = None,
        drain_distance: int | float = None,
        pssi_summer_stage: int | float = None,
        pssi_winter_stage: int | float = None,
        nearest_weather_station: int = None,
        weather_rg: str = None,
    ) -> None:
        if not (1 <= width <= 300):
            raise ValueError(
                f"Parcel width must be between 1 and 300 meters. Got {width}."
            )
        if trench_depth is not None and not (0.1 <= trench_depth <= 0.8):
            raise ValueError(
                f"Trench depth must be between 0.1 and 0.8 meters. Got {trench_depth}."
            )
        if drain_depth is not None and not (0 <= drain_depth <= 1.2):
            raise ValueError(
                f"Drain depth must be between 0 and 1.2 meters. Got {drain_depth}."
            )

        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.surface_level = surface_level
        self.soilcode = soilcode
        self.summer_stage = summer_stage
        self.winter_stage = winter_stage
        self.trench_depth = trench_depth
        self.trench_locations = trench_locations
        self.drain_depth = drain_depth
        self.drain_distance = drain_distance
        self.pssi_summer_stage = (
            pssi_summer_stage if pssi_summer_stage is not None else summer_stage
        )
        self.pssi_winter_stage = (
            pssi_winter_stage if pssi_winter_stage is not None else winter_stage
        )
        # Weather attributes below are set in `preprocessing.prepare_parcels`
        self.nearest_weather_station = nearest_weather_station
        self.weather_rg = weather_rg  # Attribute name is set to `weather_rg` to match the column name in the weather regions GeoDataFrame
        self._soilprofile = None  # Is set in `somers.preprocessing.prepare_parcels`
        self._discretization = None  # Is set in `discretize_soildepth`

    def __repr__(self):
        name = self.__class__.__name__
        wspace = " "
        items = [
            (
                f"{wspace * 4}{key.lstrip('_')}={type(value)},"
                if key == "soilprofile"
                else f"{wspace * 4}{key.lstrip('_')}={value},"
            )
            for key, value in self.__dict__.items()
        ]
        return f"{name}(\n{'\n'.join(items)}\n)"

    @property
    def discretization(self) -> components.SoilDiscretization:
        """
        Discretization of the soil profile for a Modflow model run. This is set in
        `discretize_soildepth`.

        Returns
        -------
        :class:`~somers.components.SoilDiscretization`
            Discretization of the soil profile.

        """
        return self._discretization

    @property
    def soilprofile(self) -> pd.DataFrame:
        """
        Soil profile of the parcel as derived from the BRO Bodemkaart. This is set in
        `somers.preprocessing.prepare_parcels`.

        Returns
        -------
        pd.DataFrame
            Soil profile of the parcel.

        """
        return self._soilprofile

    @soilprofile.setter
    def soilprofile(self, soilprofile: pd.DataFrame) -> None:
        self._soilprofile = soilprofile

    def discretize_soildepth(self, settings: ModelSettings) -> None:
        """
        Discretize the upper part of the subsurface which corresponds with the depth of
        the soil profile.

        Parameters
        ----------
        settings : :class:`~somers.base.ModelSettings`
            `ModelSettings` instance with the required discretization settings.

        """
        total_thickness = settings.soilprofile_thickness
        layer_thickness = settings.soil_layer_thickness

        nlayers = int(round(total_thickness / layer_thickness))
        zmid = np.arange(0.0, total_thickness, layer_thickness) + 0.5 * layer_thickness
        zbot = np.cumsum(np.repeat(layer_thickness, nlayers))

        ncolumns = int(self.width // settings.dx)
        xcol = np.arange(0.0, ncolumns * settings.dx, settings.dx) + 0.5 * settings.dx
        self._discretization = components.SoilDiscretization(nlayers, zmid, zbot, xcol)

    def load_ditches(
        self,
        date_range: pd.DatetimeIndex,
        winter: np.ndarray,
        ditch_depth: int | float = 0.7,
        ditch_resistance: int | float = 1.0,  # d
        min_water_depth: int | float = 0.4,
    ) -> components.Ditches:
        """
        Load ditch input for the Modflow model for a given time period.

        Parameters
        ----------
        date_range : pd.DatetimeIndex
            Time period for which the ditch information is loaded.
        winter : np.ndarray
            Boolean array which is True for the winter period.
        ditch_depth : int | float, optional
            Depth of a ditch in meters. The default is 0.7.
        ditch_resistance : int | float, optional
            Resistance of ditch in days. The default is 1.0.
        min_water_depth : int | float, optional
            Minimum water depth that is assumed for a ditch. The default is 0.4.

        Returns
        -------
        :class:`~somers.components.Ditches`
            Ditch component for the Modflow model.

        Raises
        ------
        ValueError
            When summer_stage or winter_stage is not defined in the `Parcel`.

        """
        if self.winter_stage is None or self.summer_stage is None:
            raise ValueError(
                "Cannot load ditch information for parcel. Summer and winter stage must "
                "be defined."
            )

        season_changes = np.insert(
            np.asarray(np.nonzero(np.diff(winter, n=1))) + 1, 0, 0
        )
        change_dates = date_range[season_changes]

        ditch_stage = np.where(
            winter[season_changes], self.winter_stage, self.summer_stage
        )

        water_depth = self.ditch_water_depth(ditch_depth, min_water_depth)
        ditch_bottom = np.max([self.winter_stage, self.summer_stage]) - water_depth
        return components.Ditches(
            ditch_bottom, ditch_resistance, ditch_stage, change_dates
        )

    def ditch_water_depth(
        self, ditch_depth: int | float = 0.7, min_water_depth: int | float = 0.4
    ) -> int | float:
        """
        Calculate the absolute ditch water depth in the parcel from the summer and winter
        stage.

        Parameters
        ----------
        ditch_depth : int | float, optional
            Depth of a ditch in meters. The default is 0.7.
        min_water_depth : int | float, optional
            Minimum water depth that is assumed for a ditch. The default is 0.4.

        Returns
        -------
        int | float
            Ditch water depth in the parcel.

        """
        try:
            # Round result to 2 decimal places to avoid floating point precision issues
            return np.max(
                [
                    (self.winter_stage - (self.surface_level - ditch_depth)),
                    (self.summer_stage - (self.surface_level - ditch_depth)),
                    min_water_depth,
                ]
            ).round(2)
        except TypeError:
            return min_water_depth

    def get_forganic(self) -> np.ndarray:
        """
        Get the organic matter content of the discretized soilprofile of a parcel. This
        uses the `Parcel.discretization.zmid` to find the layer index each discretized
        layer falls in.

        Returns
        -------
        np.ndarray
            Organic matter content of the discretized soilprofile.

        """
        if self.soilprofile is None:
            raise TypeError("No soilprofile loaded for parcel.")

        if self.discretization is None:
            raise TypeError("Soilprofile not discretized yet for parcel.")

        layer_indices = np.searchsorted(
            self.soilprofile["uppervalue"],
            self.discretization.zmid,
            sorter=self.soilprofile["uppervalue"].argsort(),
        )
        forganic = self.soilprofile["organicmattercontent"].values[layer_indices]
        return forganic

    def load_trenches(self, resistance: int | float = 1.0):
        """
        Load optional trench input for the Modflow model.

        Parameters
        ----------
        resistance : int | float, optional
            Resistance of the trench in days. The default is 1.0.

        Returns
        -------
        :class:`~somers.components.Trenches`
            Trenches component for the Modflow model.

        """
        depth = self.surface_level - self.trench_depth

        if isinstance(self.trench_locations, int):
            locations = np.round(np.linspace(0, self.width, self.trench_locations + 2))[
                1:-1
            ]
        else:
            locations = np.array(self.trench_locations)

        return components.Trenches(depth, locations, resistance)

    def load_ssi_measure(
        self,
        date_range: pd.DatetimeIndex,
        winter: np.ndarray,
        min_drain_depth: int | float = 0.2,
    ):
        """
        Load optional ssi input for the Modflow model.

        Parameters
        ----------
        date_range : pd.DatetimeIndex
            Time period for which the ssi information is loaded.
        winter : np.ndarray
            Boolean array which is True for the winter period of in the input `dater_range`.
        min_drain_depth : int | float, optional
            Minimum drainage depth for SSI or PSSI measure in meters below surface level.
            The default is 0.2.

        """
        season_changes = np.flatnonzero(np.diff(winter, prepend=0))
        change_dates = date_range[season_changes]

        drain_stage = np.where(
            winter[season_changes], self.pssi_winter_stage, self.pssi_summer_stage
        )
        drain_depth = np.min(
            [
                (self.surface_level - self.drain_depth),
                np.min(drain_stage) - min_drain_depth,
            ]
        )
        return components.SsiMeasure(
            drain_depth, self.drain_distance, drain_stage, change_dates
        )
