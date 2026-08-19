from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from parcel2d_modflow import components
from parcel2d_modflow.base import AbstractModule, Parcel
from parcel2d_modflow.config import ModelSettings
from parcel2d_modflow.exceptions import (
    InvalidInputError,
    MissingColumnError,
    ValidationError,
)
from parcel2d_modflow.mf._model import ModflowModel
from parcel2d_modflow.mf.evt_profiles import calc_evt_profile
from parcel2d_modflow.modeldata import GroundwaterData, Presets, WeatherData
from parcel2d_modflow.utils import strip_column_units


class Runs(NamedTuple):
    """
    NamedTuple containing the indexes of run numbers which are successful and failed during
    "simple" and "complex" runs with the Modflow model.
    """

    success_simple: list[int]
    failure_simple: list[int]
    success_complex: list[int]
    failure_complex: list[int]


class Modflow(AbstractModule):
    """
    Groundwater Module to run a 2D Modflow model for the groundwater component Parcels.

    Parameters
    ----------
    parameters : pd.DataFrame
        DataFrame with stochastic parameters to run the Modflow model with.
    modflow_executable : str | Path
        Path to the Modflow executable (e.g. "mf6.exe"). This is needed to run the Modflow
        model.
    aquifer_method : str
        Method to run the Modflow model with. Currently only 'flux' is implemented.
    measure : str, optional
        Measure to run the Modflow model with. Can be 'ref', 'ssi' or 'pssi'. The default
        is 'ref'.
    gw_recharge_method : str
        Method to run the Modflow model with for how groundwater recharge is determined,
        can be 'precip_evap' or 'recharge'.
        - 'precip_evap': The Modflow model uses precipitation and evapotranspiration data
        to determine groundwater recharge.
        - 'recharge': The Modflow model uses recharge data to determine groundwater recharge.
    evt_method : str, optional
        Method to calculate the evapotranspiration profile for the Modflow model when the
        `gw_recharge_method` is 'precip_evap', otherwise this will be ignored. Can be
        'woerkom', 'combi' or 'boon'. The default is 'woerkom'.
    modflow_kwargs : dict[str, Any], optional
        Optional keyword arguments for `flopy.mf6.MFSimulation` constructor for the Modflow
        model. The default is None, which passes an empty dict.
    """

    _module_type = "groundwater"

    def __init__(
        self,
        *,
        parameters: pd.DataFrame,
        modflow_executable: str | Path,
        aquifer_method: Literal["flux"] = "flux",
        measure: Literal["ref", "ssi", "pssi"] = "ref",
        gw_recharge_method: Literal["precip_evap", "recharge"] = "recharge",
        evt_method: Literal["woerkom", "combi", "boon"] = "woerkom",
        modflow_kwargs: dict[str, Any] = None,
    ):
        self._validate_init_args(
            measure, parameters, aquifer_method, gw_recharge_method
        )

        self.parameters = strip_column_units(parameters)
        self.aquifer_method = aquifer_method
        self.gw_recharge_method = gw_recharge_method
        self.executable = Path(modflow_executable)
        self.measure = measure
        self.evt_method = evt_method
        self.modflow_kwargs = modflow_kwargs or {}

        # Attributes set after initialization of the Modflow module for a given Parcel
        self._discretization = None
        self._recharge = None
        self._evapotranspiration = None
        self._precipitation = None
        self._aquifer = None
        self._ditches = None
        self._trenches = None
        self._ssi = None

        # Attributes set after running the Modflow model for a given Parcel
        self._success_and_failures = None

    def __repr__(self):
        aquifer_method = self.aquifer_method
        recharge_method = self.gw_recharge_method
        measure = self.measure
        return f"{self.__class__.__name__}({aquifer_method=}, {recharge_method=}, {measure=})"

    @staticmethod
    def _validate_init_args(
        measure: str,
        parameters: pd.DataFrame,
        aquifer_method: str,
        gw_recharge_method: str,
    ) -> None:
        """
        Validate the initialization arguments for the Modflow module.

        Raises
        ------
        ValidationError
            If the measure is not a valid measure, if entry drain resistance is required
            for the measure but not provided in the parameters, or if the aquifer method
            is invalid.

        """
        errors = []
        if measure not in {"ref", "ssi", "pssi"}:
            errors.append(
                InvalidInputError(
                    f"Measure '{measure}' is not a valid measure. Valid measures are: "
                    "{'ref', 'ssi', 'pssi'}"
                )
            )

        if (measure != "ref") and not any(
            parameters.columns.str.contains("entry_drain_resistance")
        ):
            errors.append(
                MissingColumnError(
                    f"Entry drain resistance is required for the measure: {measure}. "
                    "Please add column 'entry_drain_resistance (d)' to the parameter file."
                )
            )

        if aquifer_method != "flux":
            errors.append(
                InvalidInputError(
                    f"Aquifer method '{aquifer_method}' is not implemented for the Modflow module. "
                    "Currently only 'flux' method is implemented."
                )
            )

        if gw_recharge_method not in {"precip_evap", "recharge"}:
            errors.append(
                InvalidInputError(
                    f"Recharge method '{gw_recharge_method}' is not valid. Valid methods "
                    "are: {'precip_evap', 'recharge'}"
                )
            )
        if errors:
            raise ValidationError(errors)

    @property
    def discretization(self) -> components.SubsurfaceStructure:
        """
        :class:`~parcel2d_modflow.components.SubsurfaceStructure` input for the Modflow model.
        Available after initialization of the `Module` for a given :class:`~parcel2d_modflow.Parcel`.

        """
        return self._discretization

    @property
    def recharge(self) -> components.ModflowInputSeries:
        """
        :class:`~parcel2d_modflow.components.ModflowInputSeries` input for the Modflow model.
        Available after initialization of the `Module` for a given :class:`~parcel2d_modflow.Parcel`.

        """
        return self._recharge

    @property
    def precipitation(self) -> components.ModflowInputSeries:
        """
        :class:`~parcel2d_modflow.components.ModflowInputSeries` input for the Modflow model.
        Available after initialization of the `Module` for a given :class:`~parcel2d_modflow.Parcel`.

        """
        return self._precipitation

    @property
    def evapotranspiration(self) -> components.ModflowInputSeries:
        """
        :class:`~parcel2d_modflow.components.ModflowInputSeries` input for the Modflow model.
        Available after initialization of the `Module` for a given :class:`~parcel2d_modflow.Parcel`.

        """
        return self._evapotranspiration

    @property
    def aquifer(self) -> components.ModflowInputSeries:
        """
        :class:`~parcel2d_modflow.components.ModflowInputSeries` input for the Modflow model.
        Available after initialization of the `Module` for a given :class:`~parcel2d_modflow.Parcel`.

        """
        return self._aquifer

    @property
    def ditches(self) -> components.Ditches:
        """
        :class:`~parcel2d_modflow.components.Ditches` input for the Modflow model. Available
        after initialization of the `Module` for a given :class:`~parcel2d_modflow.Parcel`.

        """
        return self._ditches

    @property
    def trenches(self) -> components.Trenches:
        """
        :class:`~parcel2d_modflow.components.Trenches` input for the Modflow model. Available
        after initialization of the `Module` for a given :class:`~parcel2d_modflow.Parcel` and
        trenches are incorporated in the modelling.

        """
        return self._trenches

    @property
    def ssi(self) -> components.SsiMeasure:
        """
        :class:`~parcel2d_modflow.components.SsiMeasure` input for the Modflow model. Available
        after initialization of the `Module` for a given :class:`~parcel2d_modflow.Parcel` with
        'ssi' or 'pssi' as a measure.

        """
        return self._ssi

    @property
    def success_and_failures(self) -> Runs:
        """
        NamedTuple containing the run numbers which are successful and failed during
        "simple" and "complex" runs with the Modflow model.

        """
        if self._success_and_failures is None:
            raise ValueError("ModflowModel has not been run yet.")
        return self._success_and_failures

    def initialize(
        self,
        parcel: Parcel,
        settings: ModelSettings,
        *,
        lhm: GroundwaterData,
        weather: WeatherData = None,
        presets: Presets = None,
    ) -> None:
        """
        Initialize every component for a Modflow groundwater model for a parcel.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            `Parcel` for which the Modflow model is initialized.
        settings : :class:`~parcel2d_modflow.ModelSettings`
            General settings for the model run.
        lhm : :class:`~parcel2d_modflow.modeldata.GroundwaterData`
            LHM data container to select the relevant hydrological information for the
            parcel from. See :class:`~parcel2d_modflow.modeldata.GroundwaterData` docstring
            for more information.
        weather : :class:`~parcel2d_modflow.modeldata.WeatherData`, optional
            Weather data container to select the relevant weather information for the parcel
            from. See :class:`~parcel2d_modflow.modeldata.WeatherData` docstring for more
            information.
        presets : :class:`~parcel2d_modflow.modeldata.Presets`
            Presets data container with bounding conditions for the Modflow model.

        """
        if presets is None:
            presets = Presets()

        self._discretize_parcel(
            parcel, lhm, settings.dz_resistance_layer, presets.resistance
        )

        if self.gw_recharge_method == "precip_evap":
            self._load_precip_evap_data(parcel, weather, settings)
        elif self.gw_recharge_method == "recharge":
            self._load_recharge(parcel, lhm, settings, presets)

        self._load_aquifer(parcel, lhm, settings, presets)
        self._load_ditches(parcel, settings, presets)

        if settings.add_trenches:
            self._trenches = parcel.load_trenches(settings.trench_resistance)

        if self.measure == "ssi" or self.measure == "pssi":
            self._load_ssi(parcel, settings, presets)

    def run(self, parcel: Parcel, settings: ModelSettings) -> xr.DataArray:
        """
        Run the module for a given parcel and settings. This runs all stochastic
        combinations of the given Modflow parameters and results in a phreatic head for the
        parcel for each run over the given time period.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            `Parcel` for which the `Modflow` is run.
        settings : :class:`~parcel2d_modflow.ModelSettings`
            General settings for the model run.

        Returns
        -------
        phreatic_head : xr.DataArray
            Phreatic head for the parcel for each run over the given time period. The
            phreatic head is a 3D array with dimensions (runs, time, x).

        """
        if settings.dimension == "1D":
            raise NotImplementedError("1D Modflow model not implemented")

        phreatic_head = np.full(
            (
                len(self.parameters),
                len(settings.date_range),
                len(parcel.discretization.xcol),
            ),
            np.nan,
        )  # Pre-allocate array for phreatic head output of all runs with dimensions (runs, time, x) and fill with NaN values

        complexity = "SIMPLE"
        logger.debug(
            "Run Parcel ID: {name}, Soilcode: {soilcode}, complexity: {complexity}",
            name=parcel.name,
            soilcode=parcel.soilcode,
            complexity=complexity,
        )
        model = self.create_modflow_model(parcel, settings, complexity)
        phreatic_head, success_simple, failure_simple = self.run_modflow_model(
            model, self.parameters, phreatic_head, settings.start_date
        )

        if failure_simple:
            complexity = "COMPLEX"
            logger.debug(
                "Run Parcel ID: {name}, Soilcode: {soilcode}, complexity: {complexity}",
                name=parcel.name,
                soilcode=parcel.soilcode,
                complexity=complexity,
            )
            model = self.create_modflow_model(parcel, settings, complexity)
            phreatic_head, success_complex, failure_complex = self.run_modflow_model(
                model,
                self.parameters.loc[failure_simple],
                phreatic_head,
                settings.start_date,
            )
        else:
            success_complex = []
            failure_complex = []

        self._success_and_failures = Runs(
            success_simple, failure_simple, success_complex, failure_complex
        )
        phreatic_head = xr.DataArray(
            data=phreatic_head,
            coords={
                "runs": self.parameters.runnr,
                "time": settings.date_range,
                "x": parcel.discretization.xcol,
            },
            dims=("runs", "time", "x"),
        )
        return phreatic_head.isel(runs=success_simple + success_complex)

    def reset(self) -> None:
        """
        Reset the `Modflow` module to its initial state.

        """
        self._discretization = None
        self._recharge = None
        self._aquifer = None
        self._ditches = None
        self._trenches = None
        self._ssi = None
        self._success_and_failures = None

    def _discretize_parcel(
        self,
        parcel: Parcel,
        lhm: GroundwaterData,
        dz_resistance_layer: float = 0.5,
        preset_resistance: int | float = None,
    ) -> None:
        """
        Create discretization for the Modflow model of the soil profile and confining layer
        for a parcel.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            `Parcel` for which the Modflow model is discretized.
        lhm : :class:`~parcel2d_modflow.modeldata.GroundwaterData`
            LHM data container to select the confining layer information for the parcel.
        dz_resistance_layer : int | float, optional
            Thickness of the resistance layer in meters used in the Modflow groundwater
            model at the top of the aquifer. The default is 0.5.
        preset_resistance : int | float, optional
            Preset resistance (days) to be used for the resistance layer. The default is
            None.

        """
        soil = parcel.soilprofile
        thickness_holocene = np.sum(soil[soil["geology"] == 1]["thickness"])

        confining, thin_confining = lhm.load_confining_layer(
            parcel, thickness_holocene, dz_resistance_layer
        )

        thickness = np.concatenate((soil["thickness"].values, confining.thickness))
        lithology = np.concatenate((soil["lithology"].values, confining.lithology))
        geology = np.concatenate((soil["geology"].values, confining.geology))

        if thin_confining:
            ncells = np.sum(soil["geology"] == 2)
            k_soil = np.repeat(confining.kvalues[0], ncells)
            kvalues = np.concatenate((k_soil, [confining.kvalues[1]]))
        else:
            kvalues = confining.kvalues
            if preset_resistance is not None:
                kvalues[0] = dz_resistance_layer / preset_resistance

        self._discretization = components.SubsurfaceStructure(
            thickness, lithology, geology, kvalues
        )

    def _load_recharge(
        self,
        parcel: Parcel,
        lhm: GroundwaterData,
        settings: ModelSettings,
        presets: Presets,
    ) -> None:
        """
        Load LHM recharge data for the Modflow model for a given parcel and time period.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the recharge data is loaded at xy-location.
        lhm : :class:`~parcel2d_modflow.modeldata.GroundwaterData`
            LHM data container to select the recharge information for the parcel.
        settings : :class:`~parcel2d_modflow.ModelSettings`
            General settings for the model run.
        presets : :class:`~parcel2d_modflow.modeldata.Presets`
            Optional `Presets` containing an optional daily time series of recharge
            data for the ModflowModel in m/d. The default is None.

        """
        if presets.recharge is not None:
            self._recharge = presets.load_recharge(settings)
        else:
            self._recharge = lhm.load_recharge(parcel, settings)

    def _load_precip_evap_data(
        self, parcel: Parcel, weather: WeatherData, settings: ModelSettings
    ) -> None:
        """
        Load precipitation and evapotranspiration data for determining the groundwater
        recharge in the model for a given parcel and time period.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the recharge data is loaded at xy-location.
        weather : :class:`~parcel2d_modflow.modeldata.WeatherData`
            Weather data container with precipitation and evapotranspiration information
            for the parcel.
        settings : :class:`~parcel2d_modflow.ModelSettings`
            General settings for the model run.

        """
        self._precipitation = weather.load_precipitation(parcel, settings.date_range)
        self._evapotranspiration = weather.load_evapotranspiration(
            parcel, settings.date_range
        )

    def _load_aquifer(
        self,
        parcel: Parcel,
        lhm: GroundwaterData,
        settings: ModelSettings,
        presets: Presets,
    ) -> None:
        """
        Load LHM flux data for the Modflow model for a given parcel and time period.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the flux data is loaded at xy-location.
        lhm : :class:`~parcel2d_modflow.modeldata.GroundwaterData`
            LHM data container to select the flux information for the parcel.
        settings : :class:`~parcel2d_modflow.ModelSettings`
            General settings for the model run.
        presets : :class:`~parcel2d_modflow.modeldata.Presets`
            Optional `Presets` containing an optional daily time series of flux data
            for the ModflowModel in m/d. The default is None.

        Raises
        ------
        NotImplementedError
            Raises an error for aquifer methods that have not been implemented yet.

        """
        if self.aquifer_method == "flux":
            if presets.aquifer_flux is not None:
                self._aquifer = presets.load_aquifer_flux(settings)
            else:
                self._aquifer = lhm.load_aquifer_flux(parcel, settings)
        else:
            raise NotImplementedError(
                "Only 'flux' method for aquifer in a Modflow model is implemented for now."
            )

    def _load_ditches(
        self, parcel: Parcel, settings: ModelSettings, presets: Presets
    ) -> None:
        """
        Load ditch input for the Modflow model for a given parcel and time period.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the ditch input data is loaded.
        settings : :class:`~parcel2d_modflow.ModelSettings`
            General settings for the model run.
        presets : :class:`~parcel2d_modflow.modeldata.Presets`
            Optional `Presets` containing an optional daily time series of ditch
            water levels for the ModflowModel in m +NAP. The default is None.

        """
        if presets.ditch_stage is not None:
            self._ditches = presets.load_ditches(settings, parcel.surface_level)
        else:
            self._ditches = parcel.load_ditches(
                settings.date_range,
                settings.winter_period,
                settings.ditch_depth,
                settings.ditch_resistance,
                settings.min_water_depth,
            )

    def _load_ssi(self, parcel: Parcel, settings: ModelSettings, presets: Presets):
        """
        Load ssi or pssi measure for the Modflow model for a given parcel and time period.

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the ssi or pssi measure input data is loaded.
        settings : :class:`~parcel2d_modflow.ModelSettings`
            General settings for the model run.
        presets : :class:`~parcel2d_modflow.modeldata.Presets`
            Optional `Presets` containing an optional daily time series of ditch
            water levels (i.e. `Presets.ditch_stage`) if the measure is "ssi" or pssi stage
            levels (i.e. `Presets.pssi_stage) if the measure is "pssi". Both must be in m
            +NAP. The default is None.

        """
        if presets.ditch_stage is not None or presets.pssi_stage is not None:
            self._ssi = presets.load_ssi_measure(
                self.measure,
                settings.date_range,
                parcel.drain_depth,
                parcel.drain_distance,
                parcel.surface_level,
                settings.min_drain_depth,
            )
        else:
            self._ssi = parcel.load_ssi_measure(
                settings.date_range, settings.winter_period, settings.min_drain_depth
            )

    def create_modflow_model(
        self, parcel: Parcel, settings: ModelSettings, complexity: str
    ) -> ModflowModel:
        """
        Create a `ModflowModel` to run the stochastic groundwater conditions for a given
        parcel and model settings using Modflow 6 modelling components from Flopy (USGS).

        Parameters
        ----------
        parcel : :class:`~parcel2d_modflow.Parcel`
            Parcel for which the Modflow model is created.
        settings : :class:`~parcel2d_modflow.ModelSettings`
            NamedTuple containing general settings for the SOMERS model run.
        complexity : str
            Complexity of the model. Can be 'simple' or 'complex'.

        Returns
        -------
        :class:`~parcel2d_modflow.mf._model.ModflowModel`
            `ModflowModel` instance containing Modflow 6 modelling components to run the
            groundwater model.

        """
        model = ModflowModel(parcel, settings, self.discretization.thickness)
        model.setup_flopy_simulation(complexity, self.executable, **self.modflow_kwargs)

        if self.gw_recharge_method == "precip_evap":
            model.set_precipitation(self.precipitation)
            evt_profile = calc_evt_profile(self.evt_method)
            model.set_evapotranspiration(self.evapotranspiration, evt_profile)
        elif self.gw_recharge_method == "recharge":
            model.set_recharge(self.recharge)

        model.set_surface_drainage()
        model.set_ditch_boundary(self.ditches)

        if self.trenches is not None:  # pragma: no cover
            model.set_trenches(self.trenches)

        if self.aquifer_method == "flux":
            model.set_aquifer_flux(self.aquifer, settings.date_range)

        return model

    def run_modflow_model(
        self,
        model,
        parameters: pd.DataFrame,
        result: np.ndarray,
        start_date: pd.Timestamp,
    ) -> tuple[list, list]:
        """
        Run an initialized `ModflowModel` with given stochastic parameters. This runs all
        given parameter combinations and writes the modelled phreatic heads to CSV files.
        The function returns the corresponding run numbers of successful and failed runs:
        which parameter combinations can be modelled succesfully and which not.

        Parameters
        ----------
        model : :class:`~parcel2d_modflow.mf._model.ModflowModel`
            Initialized `ModflowModel` instance containing Modflow 6 modelling components
            to run the groundwater model.
        parameters : pd.DataFrame
            Parameter combinations for each run to run the Modflow model with.
        result : np.ndarray
            Pre-allocated array to store the phreatic head output of all runs with dimensions
            (runs, time, x) and filled with NaN values.
        start_date : pd.Timestamp
            Start date of the model run to select the corresponding phreatic head output
            from the model results.

        Returns
        -------
        tuple[np.ndarray, list, list]
            Tuple containing the updated result array, list of successful run numbers, and
            list of failed run numbers.

        """
        sy_sand = 0.25

        failure_runs = []
        success_runs = []

        mask_holocene = self.discretization.geology == 1
        for params in parameters.itertuples():
            runnr = params.Index
            logger.debug("Run parameters: {params}", params=params)

            kh_top = np.repeat(params.kh, sum(mask_holocene))
            kh = np.concatenate((kh_top, self.discretization.kvalues))
            kh_over_kv = np.where(mask_holocene, 5, 1)

            model.set_k(kh=kh, kh_over_kv=kh_over_kv)

            specific_yield = np.select(
                [
                    self.discretization.lithology == 1,
                    self.discretization.lithology == 2,
                    self.discretization.lithology == 3,
                ],
                [params.sy_peat, params.sy_clay, params.sy_clay],
                default=sy_sand,
            )
            model.set_specific_yield(specific_yield=specific_yield, high=1.0, low=0.5)

            if self.ssi is not None:
                entry_drain_resistance = params.entry_drain_resistance
                model.set_ssi_boundary(entry_drain_resistance, self.ssi)

            try:
                model.write()
                model.run()
                ph_run = model.get_phreatic_head()
            except ValueError:
                logger.debug("Run failed with parameters: {params}", params=params)
                failure_runs.append(runnr)
            else:
                result[runnr, :, :] = ph_run.sel(time=slice(start_date, None)).values
                success_runs.append(runnr)

        return result, success_runs, failure_runs
