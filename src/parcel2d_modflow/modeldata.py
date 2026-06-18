from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from parcel2d_modflow import components

if TYPE_CHECKING:
    import xarray as xr

    from parcel2d_modflow.base import ModelSettings


class LhmData:
    pass


@dataclass(repr=False)
class Presets:
    resistance: int | float = None
    recharge: pd.DataFrame = None
    aquifer_flux: pd.DataFrame = None
    ditch_stage: pd.DataFrame = None
    pssi_stage: pd.DataFrame = None
    soilcode: str = None
    carbon_profile: xr.DataArray = None
    ditch_frequency: str = "7D"
    ssi_frequency: str = "D"

    def __post_init__(self):
        """
        TODO: Add validation for the presets where values to meet several criteria.
        """
        pass

    def __repr__(self):
        resistance = self.resistance
        recharge = self.recharge if self.recharge is None else type(self.recharge)
        aquifer_flux = None if self.aquifer_flux is None else type(self.aquifer_flux)
        ditch_stage = None if self.ditch_stage is None else type(self.ditch_stage)
        pssi_stage = None if self.pssi_stage is None else type(self.pssi_stage)
        soilcode = self.soilcode
        carbon_profile = (
            None if self.carbon_profile is None else type(self.carbon_profile)
        )
        return (
            f"{self.__class__.__name__}("
            f"{resistance=}, {recharge=},{aquifer_flux=}, {ditch_stage=}, "
            f"{pssi_stage=}, {soilcode=}, {carbon_profile=})"
        )

    def load_aquifer_flux(self, settings: ModelSettings) -> None:
        """
        Load a daily time series of aquifer flux data for the Modflow model for a required
        modelling period. This is used to set the aquifer component in the Modflow model.

        Parameters
        ----------
        settings : :class:`~somers.base.ModelSettings`
            General settings for the model run containing the date range to load the
            aquifer flux data for.

        Returns
        -------
        :class:`~somers.components.Aquifer`
            Aquifer component for Modflow model containing the start aquifer flux for the
            time period and the aquifer flux through time.

        Raises
        ------
        KeyError
            If the aquifer flux data does not contain daily data for the required modelling
            period.

        """
        try:
            series = self.aquifer_flux.loc[settings.date_range].values.flatten()
        except KeyError:
            raise KeyError(
                f"{self.__class__.__name__}.aquifer_flux does not have daily data for "
                f"the required modelling period between {settings.start_date=} and "
                f"{settings.end_date=}."
            )
        start = np.mean(series[:30])
        return components.Aquifer(start, series)

    def load_ditches(self, settings: ModelSettings, surface_level: int | float) -> None:
        """
        Load a time series of ditch stage data for the Modflow model for a required modelling
        period. This is used to set the ditch component in the Modflow model.

        Parameters
        ----------
        settings : :class:`~somers.base.ModelSettings`
            General settings for the model run containing the date range to load the
            ditch stage data for.
        surface_level : int | float
            Surface level of a parcel (m +NAP) the ditch stage data is loaded for.

        Returns
        -------
        :class:`~somers.components.Ditches`
            Ditch component for the Modflow model.

        Raises
        ------
        KeyError
            If the ditch stage data does not contain daily data for the required modelling
            period.

        """
        try:
            ditch_stage = self.ditch_stage.loc[settings.date_range]
        except KeyError:
            raise KeyError(
                f"{self.__class__.__name__}.ditch_stage does not have daily data for "
                f"the required modelling period between {settings.start_date=} and "
                f"{settings.end_date=}."
            )

        water_depth = np.max(
            [
                ditch_stage.values.min() - (surface_level - settings.ditch_depth),
                settings.min_water_depth,
            ]
        )
        ditch_bottom = ditch_stage.values.min() - water_depth
        ditch_stage = ditch_stage.resample(self.ditch_frequency).mean()
        return components.Ditches(
            ditch_bottom,
            settings.ditch_resistance,
            ditch_stage.values.flatten(),
            ditch_stage.index,
        )

    def load_recharge(self, settings: ModelSettings) -> None:
        """
        Load a daily time series of recharge data for the Modflow model for a required
        modelling period. This is used to set the recharge component in the `Modflow` model.

        Parameters
        ----------
        settings : :class:`~somers.base.ModelSettings`
            General settings for the model run containing the date range to load the recharge
            data for.

        Returns
        -------
        :class:`~somers.components.Recharge`
            Recharge component for Modflow model containing the start recharge for the
            time period and the recharge through time.

        Raises
        ------
        KeyError
            If the recharge data does not contain daily data for the required modelling
            period.

        """
        try:
            series = self.recharge.loc[settings.date_range].values.flatten()
        except KeyError:
            raise KeyError(
                f"{self.__class__.__name__}.recharge does not have daily data for "
                f"the required modelling period between {settings.start_date=} and "
                f"{settings.end_date=}."
            )
        start = np.mean(series[:30])
        return components.Recharge(start, series)

    def load_ssi_measure(
        self,
        measure: str,
        date_range: pd.DatetimeIndex,
        drain_depth: int | float,
        drain_distance: int | float,
        surface_level: int | float,
        min_drain_depth: int | float = 0.2,
    ):
        """
        Load a time series of SSI or PSSI stage data for the Modflow model for a required
        modelling period and given attributes of a parcel. This is used to set the ssi
        component in a `Modflow` model.

        Parameters
        ----------
        measure : str
            Name of the measure to load. Can be either "ssi" or "pssi".
        date_range : pd.DatetimeIndex
            Date range of the modelling period to load the SSI or PSSI data for.
        drain_depth : int | float
            Depth of a drain in meters below surface level. The default is None.
        drain_distance : int | float
            Distance between drains in meters. The default is None.
        surface_level : int | float
            Surface level of a parcel (m +NAP).
        min_drain_depth : int | float, optional
            Minimum drainage depth for SSI or PSSI measure in meters. The default is 0.2.

        Returns
        -------
        :class:`~somers.components.SsiMeasure`
            SSI or PSSI measure component for the Modflow model.

        Raises
        ------
        KeyError
            If the SSI or PSSI stage data does not contain daily data for the required
            modelling period.

        """
        try:
            if measure == "ssi":
                drain_stage = self.ditch_stage.loc[date_range]
            elif measure == "pssi":
                drain_stage = self.pssi_stage.loc[date_range]
        except KeyError:
            raise KeyError(
                f"{self.__class__.__name__} does not have daily data for SSI/PSSI in the "
                f"required modelling period between {date_range[0]} and {date_range[-1]}. "
                "Running a `somers.pp2d.Modflow` module with SSI or PSSI measure requires "
                "daily data for the entire modelling period."
            )
        drain_stage = drain_stage.resample(self.ssi_frequency).mean()
        drain_depth = np.min(
            [surface_level - drain_depth, np.min(drain_stage) - min_drain_depth]
        )
        return components.SsiMeasure(
            drain_depth, drain_distance, drain_stage.values.flatten(), drain_stage.index
        )
