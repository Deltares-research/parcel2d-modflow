from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from shapely import geometry as gmt

from parcel2d_modflow import components, utils
from parcel2d_modflow.exceptions import InvalidPresetDataError, MissingDataError
from parcel2d_modflow.validation import validate_soilmap

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd
    import xarray as xr

    from parcel2d_modflow.base import Parcel
    from parcel2d_modflow.config import ModelSettings
    from parcel2d_modflow.constants import BestKappa, ParameterCorrectionCurve


class ModelData(NamedTuple):
    """
    Container for all data that is needed for modelling runs.

    Parameters
    ----------
    parcels : gpd.GeoDataFrame
        GeoDataFrame with all parcels that are modelled.
    groundwater : :class:`~parcel2d_modflow.modeldata.GroundwaterData`
        Container for all groundwater data that is needed for modelling runs.
    soilmap : :class:`~parcel2d_modflow.modeldata.Soilmap`
        Data container to retrieve all soilmap information (soilcodes and soilprofiles)
        for individual parcels that is needed for modelling runs.
    weather : :class:`~parcel2d_modflow.modeldata.WeatherData`
        Container for all weather data that is needed for modelling runs.
    parameters : pd.DataFrame
        DataFrame with all stochastic Modflow parameters that are needed for modelling
        runs.
    presets : :class:`~parcel2d_modflow.modeldata.Presets`
        Container for all preset data that is needed for modelling runs.
    """

    parcels: gpd.GeoDataFrame
    groundwater: GroundwaterData
    soilmap: Soilmap
    weather: WeatherData
    parameters: pd.DataFrame
    presets: Presets = None


@dataclass(repr=False, slots=True)
class GroundwaterData:
    """
    Container for all LHM data that is needed for SOMERS runs.

    Parameters
    ----------
    confining : xr.Dataset
        Dataset with LHM confining layer information.
    flux : xr.Dataset
        Dataset with LHM flux information.
    recharge : xr.DataArray
        DataArray with LHM recharge information.
    head: xr.DataArray
        DataArray with the LHM phreatic head information.
    """

    confining: xr.Dataset = None
    flux: xr.DataArray = None
    recharge: xr.DataArray = None
    head: xr.DataArray = None
    cell_area: tuple[int | float, int | float] = field(init=False, default=None)

    @staticmethod
    def _as_dataarray(data):
        import xarray as xr

        if isinstance(data, xr.Dataset) and len(data.data_vars) == 1:
            return data[next(iter(data.data_vars))]
        return data

    def __post_init__(self):
        import rioxarray

        if self.flux is not None:
            self.flux = self._as_dataarray(self.flux)
            try:
                xsize, ysize = self.flux.rio.resolution()
                self.cell_area = abs(xsize) * abs(ysize)
            except rioxarray.exceptions.OneDimensionalRaster:
                self.cell_area = 250 * 250  # Default cell size for LHM data

    def __repr__(self):
        confining = type(self.confining)
        flux = type(self.flux)
        recharge = type(self.recharge)
        head = type(self.head)
        return f"{self.__class__.__name__}({confining=}, {flux=}, {recharge=}, {head=})"

    def load_confining_layer(
        self, parcel: Parcel, thickness_holocene: float, dz_resistance: float = 0.5
    ) -> tuple[components.SubsurfaceStructure, bool]:
        """
        Load confining layer input for the Modflow model for a given parcel.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the confining layer is loaded at xy-location.
        thickness_holocene : float
            Thickness of the Holocene (confining) layer.
        dz_resistance : int | float, optional
            Thickness of the resistance layer in meters used in the Modflow groundwater
            model at the top of the aquifer. The default is 0.5.

        Returns
        -------
        tuple[components.SubsurfaceStructure, bool]
            Tuple with the confining layer structure and a boolean indicating if the
            confining layer is thin.

        """
        if self.confining is None:
            raise AttributeError("Cannot load confining layer from NoneType.")

        confining = self.confining.sel(x=parcel.x, y=parcel.y, method="nearest")

        remaining_confining_thickness = np.max(
            [0.0, confining["thickness"] - thickness_holocene]
        )
        ncells = np.round(remaining_confining_thickness / 0.5).astype(int) - 1

        k = confining["k_value_1aq"]
        kd = confining["kd_value_1aq"]

        pleistocene = 2
        sand = 4

        thin_confining_layer = thickness_holocene < 1.2 or ncells < 0

        if thin_confining_layer:
            thickness = np.array([1.0])
            lithology = np.array([sand])
            geology = np.array([pleistocene])
            kvalues = np.array([k, kd])
        else:
            thickness = np.append(
                np.repeat(0.5, ncells), np.array([dz_resistance, 1.0])
            )
            lithology = np.append(np.repeat(1, ncells), np.array([sand, sand]))
            geology = np.append(
                np.repeat(1, ncells), np.array([pleistocene, pleistocene])
            )
            kvalues = np.array([dz_resistance / confining["resistance"], kd])

        structure = components.SubsurfaceStructure(
            thickness, lithology, geology, kvalues
        )
        return structure, thin_confining_layer

    def load_aquifer_flux(
        self, parcel: Parcel, settings: ModelSettings
    ) -> components.ModflowInputSeries:
        """
        Load LHM aquifer flux data for a given parcel and time period.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the recharge data is loaded at xy-location.
        settings : :class:`~parcel2d_modflow.ModelSettings`
            Model settings containing the start and end dates of the time period.

        Returns
        -------
        :class:`~parcel2d_modflow.components.ModflowInputSeries`
            Aquifer flux component for Modflow model containing the start flux for the
            time period and the flux through time.

        """
        if self.flux is None:
            raise AttributeError("Cannot load aquifer flux from NoneType.")

        self.flux = self._as_dataarray(self.flux)
        flux_xy = self.flux.sel(x=parcel.x, y=parcel.y, method="nearest")

        start_date = settings.start_date
        flux_start = flux_xy.sel(
            time=slice(start_date - pd.Timedelta(days=60), start_date)
        )
        flux_start = flux_start.mean().item() / self.cell_area
        flux = flux_xy.sel(time=settings.date_range).values / self.cell_area
        return components.ModflowInputSeries(flux_start, flux)

    def load_recharge(
        self, parcel: Parcel, settings: ModelSettings
    ) -> components.ModflowInputSeries:
        """
        Load LHM recharge data for a given parcel and time period.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the recharge data is loaded at xy-location.
        settings : :class:`~parcel2d_modflow.ModelSettings`
            Model settings containing the start and end dates of the time period.

        Returns
        -------
        :class:`~parcel2d_modflow.components.ModflowInputSeries`
            Recharge component for Modflow model containing the start recharge for the
            time period and the recharge through time.

        """
        if self.recharge is None:
            raise AttributeError("Cannot load recharge from NoneType.")

        self.recharge = self._as_dataarray(self.recharge)
        recharge = self.recharge.sel(x=parcel.x, y=parcel.y, method="nearest")

        mm_to_m = 1000

        start_date = settings.start_date
        recharge_start = (
            recharge.sel(time=slice(start_date - pd.Timedelta(days=60), start_date))
            .mean()
            .item()
            / mm_to_m
        )
        recharge_series = recharge.sel(time=settings.date_range) / mm_to_m
        return components.ModflowInputSeries(recharge_start, recharge_series.values)

    def load_phreatic_head(
        self, parcel: Parcel, date_range: pd.DatetimeIndex
    ) -> xr.DataArray:
        """
        Load LHM phreatic head data for a given parcel and time period.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the phreatic head data is loaded at xy-location.
        date_range : pd.DatetimeIndex
            Date range to load the data for. Will raise an error if no daily data is
            present for the entire date range.

        Returns
        -------
        phreatic_head: xr.DataArray
            Daily time series of phreatic head data for the Measurements module.

        """
        if self.head is None:
            raise AttributeError("Cannot load phreatic head from NoneType.")

        self.head = self._as_dataarray(self.head)
        head = self.head.sel(x=parcel.x, y=parcel.y, method="nearest")

        try:
            head = head.sel(time=date_range)
        except KeyError as e:
            raise MissingDataError(
                "Phreatic head does not have data for the modelling period."
            ) from e

        return head


@dataclass(repr=False, slots=True)
class Soilmap:
    """
    Data container to retrieve all soilmap information (soilcodes and soilprofiles) for
    individual parcels that is needed for SOMERS runs.

    Parameters
    ----------
    soilmap : gpd.GeoDataFrame
        `GeoDataFrame` with polygons indicating the spatial extents of different soiltypes.
        and associated information to use for spatial selections.
    soilprofiles : pd.DataFrame
        `DataFrame` containing associated soilprofile information for the polygons in the
        soilmap `GeoDataFrame`.
    """

    soilmap: gpd.GeoDataFrame
    soilprofiles: pd.DataFrame

    def __repr__(self):
        soilmap = type(self.soilmap)
        soilprofiles = type(self.soilprofiles)
        return f"{self.__class__.__name__}({soilmap=}, {soilprofiles=})"

    @classmethod
    @validate_soilmap
    def from_files(cls, soilmap: str | Path, soilprofiles: str | Path):
        """
        Create a `Soilmap` instance from individual files containing the soilmap and
        soilprofiles information. The files should be in the correct format (e.g.
        Geoparquet for the soilmap and csv or parquet for the soilprofiles).

        Parameters
        ----------
        soilmap : str | Path
            Path to the file containing the soilmap information. The file should be
            readable by geopandas.
        soilprofiles : str | Path
            Path to the file containing the soilprofiles information. The file should be
            in csv-like or parquet format.

        Returns
        -------
        :class:`~somers.modeldata.Soilmap`
            `Soilmap` instance containing the soilmap and soilprofiles information.
        """
        soilmap = utils.geopandas_read(soilmap)
        soilprofiles = utils.pandas_read(soilprofiles)

        soilprofiles["thickness"] = (
            soilprofiles["uppervalue"] - soilprofiles["lowervalue"]
        )

        to_fraction = 100
        soilprofiles["organicmattercontent"] /= to_fraction
        soilprofiles["lithology"] = utils.determine_lithology_from(soilprofiles)

        return cls(soilmap, soilprofiles)

    def _contains(self, x, y):
        """
        Select the soilmap polygons that contain a given point (x, y). This first selects
        polygons using the spatial index (i.e. bbox contains the point) to speed up the
        selection process because .contains() is called on less polygons.

        """
        sel = self.soilmap.cx[x, y]
        return sel.loc[sel["geometry"].contains(gmt.Point(x, y))]

    def soilcode_at(self, x: int | float, y: int | float) -> str:
        """
        Select the soilunit code for a given point (x, y).

        Parameters
        ----------
        x, y : int | float
            Coordinates of the point to select the soilunit code for.

        Returns
        -------
        str
            Soilunit code for the given point (x, y).

        """
        selection = self._contains(x, y)
        return selection["soilunit_code"].item()

    def soilprofile_at(self, x: int | float, y: int | float) -> pd.DataFrame:
        """
        Select the soilprofile for a given point (x, y).

        Parameters
        ----------
        x, y : int | float
            Coordinates of the point to select the soilprofile for.

        Returns
        -------
        pd.DataFrame
            Soilprofile for the given point (x, y).

        """
        sel = self._contains(x, y)

        profile = self.soilprofiles.loc[
            self.soilprofiles["normalsoilprofile_id"]
            == sel["normalsoilprofile_id"].item()
        ]
        return profile.copy()  # Do not return a view

    def load_soilprofile(self, parcel: Parcel) -> pd.DataFrame:
        """
        Load the soil profile for a given parcel.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            `Parcel` for which the soil profile is loaded.

        Returns
        -------
        pd.DataFrame
            Pandas `DataFrame` with soil profile information.

        """
        if parcel.soilcode is None:
            profile = self.soilprofile_at(parcel.x, parcel.y)
        else:
            profile = self.soilprofiles.loc[
                self.soilprofiles["soilunit_code"] == parcel.soilcode
            ]
        profile = profile.copy()
        profile.loc[:, "geology"] = utils.lithology_to_geology(
            profile["lithology"].values
        )
        return profile


@dataclass(repr=False, slots=True)
class WeatherData:
    """
    Container for weather data.

    Parameters
    ----------
    stations : gpd.GeoDataFrame
        GeoDataFrame with KNMI weather station locations.
    measurements : pd.DataFrame
        DataFrame with measurement data from the KNMI. The data must at least contain
        temperature ("TG") data. Precipitation ("RR") and evapotranspiration ("EVAP")
        are optional and are only needed when the `precip_evap_method="precip_evap"` in
        :class:`~parcel2d_modflow.mf.Modflow`.
    regions : gpd.GeoDataFrame
        GeoDataFrame with weather regions.
    correction_params : :class:`~parcel2d_modflow.constants.ParameterCorrectionCurve`
        NamedTuple containing the correction parameters for temperature data.
    kappa : :class:`~parcel2d_modflow.constants.BestKappa`
        NamedTuple containing best kappa parameters for the soil temperature module.
    """

    stations: gpd.GeoDataFrame
    measurements: pd.DataFrame
    regions: gpd.GeoDataFrame
    correction_params: ParameterCorrectionCurve
    kappa: BestKappa

    def __repr__(self):
        stations = type(self.stations)
        measurements = type(self.measurements)
        regions = type(self.regions)
        correction_params = type(self.correction_params)
        kappa = type(self.kappa)
        return (
            f"{self.__class__.__name__}({stations=}, {measurements=}, {regions=}, "
            f"{correction_params=}, {kappa=})"
        )

    def calc_corrected_temperature(self) -> None:
        """
        Calculate the corrected air temperature based on the correction parameters. This
        adds a new column "corrected_air_temp" to the temperature DataFrame.

        """
        dayofyear = self.measurements.index.dayofyear
        a, b, c, d = self.correction_params
        correction = a * np.sin(dayofyear * b - c) + d
        corrected = self.measurements["TG"] + correction
        self.measurements["corrected_air_temp"] = corrected

    def get_corrected_air_temperature(
        self,
        parcel: Parcel,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        spinup: int = 60,
    ) -> pd.Series:
        """
        Select the corrected air temperature for a given parcel and time period.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the temperature data is loaded.
        start_date : pd.Timestamp
            Start date (day) of the time period.
        end_date : pd.Timestamp
            End date (day) of the time period.
        spinup : int, optional
            "Spinup" period in days for which to select temperature data. The selects the
            number of days before the start date. The default is 60 days.

        Returns
        -------
        pd.Series
            Pandas Series with the corrected air temperature values with datetime index
            for the given time period including spinup days.

        """
        if "corrected_air_temp" not in self.measurements.columns:
            self.calc_corrected_temperature()

        spinup = pd.Timedelta(days=spinup)

        temperature = self.measurements[
            self.measurements["STN"] == parcel.nearest_weather_station
        ]
        return temperature.loc[(start_date - spinup) : end_date, "corrected_air_temp"]

    def get_weather_region(self, parcel: Parcel) -> str:
        """
        Select the name of the weather region a given parcel is located in.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the weather region is loaded.

        Returns
        -------
        str
            Name of the weather region.

        """
        return self.regions.loc[
            self.regions["geometry"].contains(gmt.Point(parcel.x, parcel.y)),
            "weather_rg",
        ].item()

    def _load_modflow_series(
        self, column: str, parcel: Parcel, date_range: pd.DatetimeIndex, spinup: int
    ) -> components.ModflowInputSeries:
        data = self.measurements.loc[
            self.measurements["STN"] == parcel.nearest_weather_station, column
        ]
        start_date = date_range[0]
        try:
            precipitation_series = data.loc[date_range]
            data_start = (
                data.loc[slice(start_date - pd.Timedelta(days=spinup), start_date)]
                .mean()
                .item()
            )
        except KeyError as e:
            raise MissingDataError(
                f"Weather data is missing '{column}' data for the required modelling period."
            ) from e

        return components.ModflowInputSeries(data_start, precipitation_series.values)

    def load_precipitation(
        self,
        parcel: Parcel,
        date_range: pd.DatetimeIndex,
        spinup: int = 60,
    ) -> components.ModflowInputSeries:
        """
        Load the precipitation data for a given parcel and time period.

        Parameters
        ----------
        parcel : Parcel
            Parcel for which the precipitation data is loaded.
        date_range: pd.DatetimeIndex
            Date range to load the data for. Will raise an error if no daily data is
            present for the entire date range.
        spinup : int, optional
            "Spinup" period in days for which to select precipitation data. This selects
            the number of days before the start date. The default is 60 days.

        Returns
        -------
        components.ModflowInputSeries
            Pandas Series containing the precipitation data for the given parcel.

        """
        return self._load_modflow_series("RH", parcel, date_range, spinup)

    def load_evapotranspiration(
        self,
        parcel: Parcel,
        date_range: pd.DatetimeIndex,
        spinup: int = 60,
    ) -> components.ModflowInputSeries:
        """
        Load the evapotranspiration data for a given parcel and time period.

        Parameters
        ----------
        parcel : :class:`~somers.base.Parcel`
            Parcel for which the evapotranspiration data is loaded.
        date_range: pd.DatetimeIndex
            Date range to load the data for. Will raise an error if no daily data is
            present for the entire date range.
        spinup : int, optional
            "Spinup" period in days for which to select evapotranspiration data. This selects
            the number of days before the start date. The default is 60 days.

        Returns
        -------
        components.ModflowInputSeries
            Evapotranspiration data for the given parcel.

        """
        return self._load_modflow_series("EV24", parcel, date_range, spinup)

    def measurements_to_csv(self, path: str | Path, **kwargs) -> None:
        """
        Save the measurements data to a csv file similar to ones downloaded from the KNMI
        but without the header.

        Parameters
        ----------
        path : str | Path
            Path to save the csv file to.
        **kwargs
            Additional keyword arguments to pass to `pd.DataFrame.to_csv()`.

        """
        measurements = self.measurements.copy()

        # We need to convert units back to the original units and stored integer formats
        if "TG" in measurements.columns:
            to_degree_celsius = 10
            measurements["TG"] = (measurements["TG"] * to_degree_celsius).astype(int)

        to_m_per_day = 1e4
        if "RH" in measurements.columns:
            measurements["RH"] *= to_m_per_day
        if "EV24" in measurements.columns:
            measurements["EV24"] *= to_m_per_day

        measurements.to_csv(path, **kwargs)


# TODO: Figure out how to deal with Presets. So far, has not been used in any Somers work
# but in the calibration of the Modflow model. Now, it only allows for loading a single
# time series of aquifer flux, recharge, ditch stage and pssi stage data for a given
# parcel. Also, these are in separate DataFrames. This is not very flexible and should be
# improved in the future. Preferably, input data from Presets should be the same as
# LhmData (Maybe rename this class then).
@dataclass(repr=False)
class Presets:
    resistance: int | float = None
    ditch_stage: pd.DataFrame = None
    pssi_stage: pd.DataFrame = None
    ditch_frequency: str = "7D"
    ssi_frequency: str = "D"

    def __post_init__(self):
        errors = []
        if self.ditch_stage is not None and self.ditch_stage.dims != ("name", "time"):
            errors.append(
                InvalidPresetDataError(
                    f"{self.__class__.__name__}.ditch_stage must be a DataArray with dims "
                    f"('name', 'time'). Now, it has dims {self.ditch_stage.dims}."
                )
            )

        if self.pssi_stage is not None and self.pssi_stage.dims != ("name", "time"):
            errors.append(
                InvalidPresetDataError(
                    f"{self.__class__.__name__}.pssi_stage must be a DataArray with dims "
                    f"('name', 'time'). Now, it has dims {self.pssi_stage.dims}."
                )
            )
        if errors:
            raise InvalidPresetDataError(errors)

    def __repr__(self):
        resistance = self.resistance
        ditch_stage = None if self.ditch_stage is None else type(self.ditch_stage)
        pssi_stage = None if self.pssi_stage is None else type(self.pssi_stage)
        return (
            f"{self.__class__.__name__}({resistance=}, {ditch_stage=}, {pssi_stage=})"
        )

    def load_ditches(
        self, parcel: Parcel, settings: ModelSettings
    ) -> components.Ditches:
        """
        Load a time series of ditch stage data for the Modflow model for a required modelling
        period. This is used to set the ditch component in the Modflow model.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.base.Parcel`
            Parcel for which the preset ditch stage data is loaded.
        settings : :class:`~parcel2d_modflow.base.ModelSettings`
            General settings for the model run containing the date range to load the
            ditch stage data for.

        Returns
        -------
        :class:`~parcel2d_modflow.components.Ditches`
            Ditch component for the Modflow model.

        Raises
        ------
        MissingDataError
            If the ditch stage data does not contain daily data for the required modelling
            period.

        """
        try:
            ditch_stage = self.ditch_stage.sel(
                name=parcel.name, time=settings.date_range
            )
        except KeyError as e:
            raise MissingDataError(
                f"{self.__class__.__name__}.ditch_stage does not have daily data for "
                f"parcel: {parcel.name} in the required modelling period between "
                f"{settings.start_date=} and {settings.end_date=}."
            ) from e

        water_depth = np.max(
            [
                ditch_stage.values.min()
                - (parcel.surface_level - settings.ditch_depth),
                settings.min_water_depth,
            ]
        )
        ditch_bottom = ditch_stage.values.min() - water_depth
        ditch_stage = ditch_stage.resample(time=self.ditch_frequency).mean()
        return components.Ditches(
            ditch_bottom,
            settings.ditch_resistance,
            ditch_stage.values,
            ditch_stage.time.values,
        )

    def load_ssi_measure(
        self, parcel: Parcel, settings: ModelSettings, measure: str
    ) -> components.SsiMeasure:
        """
        Load a time series of SSI or PSSI stage data for the Modflow model for a required
        modelling period and given attributes of a parcel. This is used to set the ssi
        component in a `Modflow` model.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.base.Parcel`
            Parcel for which the preset SSI or PSSI stage data is loaded.
        settings : :class:`~parcel2d_modflow.base.ModelSettings`
            General settings for the model run containing the date range to load the
            SSI or PSSI stage data for.
        measure : str
            Name of the measure to load. Can be either "ssi" or "pssi".

        Returns
        -------
        :class:`~parcel2d_modflow.components.SsiMeasure`
            SSI or PSSI measure component for the Modflow model.

        Raises
        ------
        MissingDataError
            If the SSI or PSSI stage data does not contain daily data for the required
            modelling period.

        """
        try:
            if measure == "ssi":
                drain_stage = self.ditch_stage.sel(
                    name=parcel.name, time=settings.date_range
                )
            elif measure == "pssi":
                drain_stage = self.pssi_stage.sel(
                    name=parcel.name, time=settings.date_range
                )
        except KeyError:
            raise MissingDataError(
                f"{self.__class__.__name__} does not have daily data for SSI/PSSI for "
                f"parcel: {parcel.name} in the required modelling period between "
                f"{settings.start_date=} and {settings.end_date=}."
            )

        drain_stage = drain_stage.resample(time=self.ssi_frequency).mean()
        drain_depth = np.min(
            [
                parcel.surface_level - parcel.drain_depth,
                np.min(drain_stage) - settings.min_drain_depth,
            ]
        )
        return components.SsiMeasure(
            drain_depth,
            parcel.drain_distance,
            drain_stage.values,
            drain_stage.time.values,
        )
