from typing import NamedTuple

import flopy
import numpy as np
import pandas as pd
import xarray as xr

from parcel2d_modflow.mf.budget_file import read_cbc

FloatArray = np.ndarray

# Define constants
SURFACE_DRAINAGE_CONDUCTANCE = 100
KH_DRAIN = 5  # m/d
SAVE_FLOWS = False


class ModflowModel:
    """
    Two dimensional MODFLOW6 groundwater flow model
    """

    def __init__(
        self,
        parcel: NamedTuple,
        settings: NamedTuple,
        thickness: np.ndarray,
    ):
        self.name = parcel.name
        self.save_flows = settings.save_flopy
        temp_dir_name = parcel.name + "_" + parcel.soilcode
        self.output_dir_runs = settings.workdir / f"{temp_dir_name}/runs"
        self.working_dir = settings.workdir / f"{temp_dir_name}/modelfiles"
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.start = settings.start_date - pd.Timedelta(days=1)
        self.end = settings.end_date + pd.Timedelta(days=1)
        self.time = settings.date_range.insert(0, self.start)  # Extra day for warmup
        date_range = settings.date_range.insert(len(settings.date_range), self.end)
        self.duration = (date_range - self.time).days.astype(float)

        self.parcel_width = parcel.width
        self.surface = np.round(
            parcel.surface_level, 2
        )  # Rounding to prevent cell bottom mismatch in model.riv file

        layer_bot = self.surface - thickness.cumsum()
        layer_top = layer_bot + thickness
        layer_z = 0.5 * (layer_bot + layer_top)  # center of each layer
        soillayers = np.repeat(
            settings.soil_layer_thickness, parcel.discretization.nlayers
        )
        layers_below_soil = thickness[
            layer_z <= self.surface - settings.soilprofile_thickness
        ]

        self.dz = np.concatenate((soillayers, layers_below_soil))
        self.nlayers = len(self.dz)
        self.bottom = self.surface - self.dz.cumsum()
        self.top = self.bottom + self.dz
        self.z = 0.5 * (self.top + self.bottom)
        self.x = parcel.discretization.xcol
        self.ncol = len(self.x)
        self.dx = settings.dx
        self.dy = 1.0
        self.vertical_index = np.digitize(self.z, layer_bot, right=True)

    def setup_flopy_simulation(self, complexity: str, executable: str = None, **kwargs):
        """
        Setup all the Flopy simulation objects for the Modflow model.

        Parameters
        ----------
        complexity : str
            Complexity of the simulation. Can be 'simple' or 'complex'.
        executable : str, optional
            Path-like string to the Modflow executable. If not provided, the model cannot
            run.

        """
        if "verbosity_level" not in kwargs:
            kwargs["verbosity_level"] = 0

        self.sim = flopy.mf6.MFSimulation(
            sim_name=self.name,
            sim_ws=str(self.working_dir),
            exe_name=executable,
            **kwargs,
        )
        self.tdis = flopy.mf6.ModflowTdis(
            self.sim,
            start_date_time=self.start.strftime("%Y-%m-%d"),
            nper=self.duration.size,
            perioddata=[(perlen, 1, 1) for perlen in self.duration],
        )

        self.solver = flopy.mf6.ModflowIms(
            self.sim,
            outer_dvclose=1.0e-4,
            inner_dvclose=1.0e-5,
            # outer_maximum=50,
            # inner_maximum=200,
            rcloserecord=1.0e-6,
            complexity=complexity,
            no_ptcrecord=["FIRST"],
            linear_acceleration="BICGSTAB",  # since newton gives assymetric matrix
            relaxation_factor=0.98,
        )

        # Nota bene: this is where we set Newton-Rhapson formulation
        self.gwf = flopy.mf6.ModflowGwf(self.sim, newtonoptions=["NEWTON"])
        self.dis = flopy.mf6.ModflowGwfdis(
            self.gwf,
            nrow=1,
            delr=self.dx,
            ncol=self.ncol,
            delc=1,
            nlay=self.nlayers,
            top=self.top[0],
            botm=self.bottom,
        )

        self.ic = flopy.mf6.ModflowGwfic(self.gwf, strt=self.surface)
        self.oc = flopy.mf6.ModflowGwfoc(
            self.gwf,
            head_filerecord=f"{self.name}.hds",
            budget_filerecord=f"{self.name}.cbb",
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )

        # Initialize all other variables (some optional) as None
        self.kh = None
        self.kh_over_kv = None
        self.recharge = None
        self.ditch_stage = None
        self.aquifer_chd = None
        self.aquifer_wel = None
        self.riv = None
        self.riv_drn = None
        self.trn = None
        self.ssi = None
        self.wel = None
        self.npf = None
        self.rch = None
        self.sto = None

        # Output
        self.head = None
        self.budgets = None

    def remove_package(self, package):
        if package is not None:
            self.gwf.remove_package(package)

    def set_k(self, kh: FloatArray, kh_over_kv: FloatArray):
        """
        Set horizontal conductivity value, kh;
        Set vertical anistropy, defined as kh/kv,
        (Should generally be larger than 1).

        Parameters
        ----------
        kh: np.ndarray of floats
            horizontal conductivity (m/d)
        kh_over_kv: np.ndarray of floats:
            anistropy factor, kh / kv
        """
        if (kh_over_kv < 1).any():
            raise UserWarning("kh_over_kv contains values < 1")

        self.kh = np.tile(kh[self.vertical_index, None, None], (1, 1, self.ncol))
        self.kh_over_kv = np.tile(
            kh_over_kv[self.vertical_index, None, None], (1, 1, self.ncol)
        )

        self.kv = self.kh / self.kh_over_kv

        self.remove_package(self.npf)
        self.npf = flopy.mf6.ModflowGwfnpf(
            self.gwf,
            save_saturation=True,
            icelltype=1,
            k=self.kh,
            k22=self.kh,
            k33=self.kv,
            save_flows=self.save_flows,
        )

    def set_specific_yield(self, specific_yield: FloatArray, high: float, low: float):
        """
        Specific yield is linearly interpolated from a low value at the top of
        the model, to a higher value at 1.2 m, to simulate reduced specific yield
        in the upper layer of the soil. To not scale specific yield, set low=high=1.0

        Parameters
        ----------
        specific_yield: np.ndarray of floats
            "Unscaled" specific yield of soil layers.
        high: float
            Should be between 0.0 and 1.0. Should generally be 1.0.
        low: float
            Should be between 0.0 and 1.0.
        """
        if (low > 1.0) or (high > 1.0):
            raise ValueError("'low' and 'high' must be smaller than, or equal to 1.0")
        if low > high:
            raise ValueError("'low' must be smaller than 'high'")

        sy_modifier = np.interp(self.surface - self.z, xp=[0.0, 1.2], fp=[low, high])
        self.specific_yield = specific_yield[self.vertical_index] * sy_modifier
        self.remove_package(self.sto)
        self.sto = flopy.mf6.ModflowGwfsto(
            self.gwf,
            iconvert=1,
            sy=self.specific_yield,
            ss=1.0e-5,  # elastic storage
            steady_state={0: True},
            transient={1: True},
        )

    def set_recharge(self, recharge: NamedTuple):
        """
        Parameters
        ----------
        recharge: np.ndarray of floats
            Groundwater recharge (m/d).
            Size should be equal to the number of transient stress periods;
            i.e. it is assumed every stress period has its own recharge value.
        """
        if recharge.series.size != self.duration.size - 1:
            raise ValueError(
                "Recharge size does not match time discretization. "
                f"Expected {self.duration.size - 1}, got {recharge.series.size}"
            )

        self.recharge = np.insert(recharge.series, 0, recharge.start).astype(np.float64)

        self.remove_package(self.rch)
        self.rch = flopy.mf6.ModflowGwfrcha(
            self.gwf, recharge=recharge.start, save_flows=self.save_flows
        )
        # Do not remove lines below. Needed if `write_recharge_file` is removed.
        # stress_period_data = {i: v for i, v in enumerate(self.recharge)}
        # self.rch = flopy.mf6.ModflowGwfrcha(
        #     self.gwf, recharge=stress_period_data[0], save_flows=self.save_flows
        # )

    def write_recharge_file(self):
        header = (
            "# File generated by Flopy version 3.3.4 on 12/15/2022 at 18:33:45.\n"
            "BEGIN options\n"
            "  READASARRAYS\n"
            "END options\n\n"
        )

        format_rch = (
            "BEGIN period  {period}\n"
            "  recharge\n"
            "    CONSTANT  {value}\n"
            "END period  {period}\n\n"
        )

        text = header
        for period, value in enumerate(self.recharge):
            text += format_rch.format(period=period + 1, value=value)

        with open(self.working_dir.joinpath("model.rcha"), "w") as f:
            f.write(text)

    def set_aquifer_head(self, aquifer_input: NamedTuple, time: pd.DatetimeIndex):
        """
        To set simply give the aquifer input NamedTuple,
        including a one sized array,
        with a one sized time array to match.

        Parameters in aquifer_input
        ----------
        aquifer_series: np.ndarray of floats
            head to set on the lower boundary.
            May vary in time, does not vary in space.
        aquifer_start: float
            Starting head for calculation of steady state starting head
        time: np.ndarray of datetimes (string, np.datetime64, pd.Timestamp)
            Starting times at which the head is active.
            Will default to forward filling in time.
        """
        aquifer_head = aquifer_input.series
        aquifer_start = aquifer_input.start
        self.aquifer_wel = None

        periods = np.searchsorted(self.time, time)
        self.periods_chd = np.insert(periods, 0, 0)

        # Compute mean head for first steady-state period
        self.aquifer_head = np.insert(aquifer_head, 0, aquifer_start)

        indices = [(self.nlayers - 1, 0, i) for i in range(self.ncol)]
        stress_period_data = {}
        for i, h in zip(self.periods_chd, self.aquifer_head):
            stress_period_data[i] = [(i, h) for i in indices]

        self.remove_package(self.aquifer_chd)
        self.aquifer_chd = flopy.mf6.ModflowGwfchd(
            self.gwf,
            stress_period_data=stress_period_data[0],
            save_flows=self.save_flows,
            filename="model.aquifer_chd",
            pname="aquifer_chd",
        )

    def write_chd_file(self):
        indices = [(self.nlayers, 1, i + 1) for i in range(self.ncol)]

        header = (
            "# File generated by Flopy version 3.3.4 on 12/15/2022 at 18:33:45.\n"
            "BEGIN options\n"
            "END options\n\n"
            "BEGIN dimensions\n"
            "  MAXBOUND  {maxbound}\n"
            "END dimensions\n\n"
        )
        line_format = "  {lay} {row} {col}      {head}\n"
        period_format_begin = "BEGIN period  {period}\n"
        period_format_end = "END period  {period}\n\n"

        text = header.format(maxbound=self.ncol)
        for period, head in zip(self.periods_chd, self.aquifer_head):
            period += 1
            text += period_format_begin.format(period=period)
            for index in indices:
                lay, row, col = index
                text += line_format.format(lay=lay, row=row, col=col, head=float(head))
            text += period_format_end.format(period=period)

        with open(self.working_dir.joinpath("model.aquifer_chd"), "w") as f:
            f.write(text)

    def set_aquifer_flux(self, aquifer: NamedTuple, time: pd.DatetimeIndex):
        """
        To set a flux, simply give the aquifer input NamedTuple,
        including a one sized array,
        with a one sized time array to match.
        aquifer_flux will be multiplied by dx to generate the flux.

        Parameters in aquifer_input
        ----------
        aquifer: :class:`~parcel2d_modflow.components.Aquifer`
            Aquifer NamedTuple object containing the aquifer flux series and start value.
        time: np.ndarray of datetimes (string, np.datetime64, pd.Timestamp)
            Starting times at which the head is active.
            Will default to forward filling in time.
        """  # TODO: docstring is incorrect.
        self.aquifer_chd = None

        # Compute mean flux for first steady-state period
        self.aquifer_flux = np.insert(aquifer.series, 0, aquifer.start)

        cellsize = self.dx * self.dy
        self.aquifer_flux = self.aquifer_flux * cellsize

        periods = np.searchsorted(self.time, time)
        self.periods_wel = np.insert(periods, 0, 0)

        ix = self.nlayers - 1
        indices = [(ix - 1, 0, i) for i in range(self.ncol)]
        stress_period_data = {}
        for i, h in zip(self.periods_wel, self.aquifer_flux):
            stress_period_data[i] = [(i, h) for i in indices]

        self.remove_package(self.aquifer_wel)
        self.aquifer_wel = flopy.mf6.ModflowGwfwel(
            self.gwf,
            stress_period_data=stress_period_data[0],
            save_flows=self.save_flows,
            filename="model.aquifer_wel",
            pname="aquifer_wel",
        )

    def write_wel_file(self):
        indices = [(self.nlayers, 1, i + 1) for i in range(self.ncol)]

        header = (
            "# File generated by Flopy version 3.3.4 on 12/15/2022 at 18:33:45.\n"
            "BEGIN options\n"
            "END options\n\n"
            "BEGIN dimensions\n"
            "  MAXBOUND  {maxbound}\n"
            "END dimensions\n\n"
        )
        line_format = "  {lay} {row} {col}      {head}\n"
        period_format_begin = "BEGIN period  {period}\n"
        period_format_end = "END period  {period}\n\n"

        text = header.format(maxbound=self.ncol)
        for period, head in zip(self.periods_wel, self.aquifer_flux):
            period += 1
            text += period_format_begin.format(period=period)
            for index in indices:
                lay, row, col = index
                text += line_format.format(lay=lay, row=row, col=col, head=float(head))
            text += period_format_end.format(period=period)

        with open(self.working_dir.joinpath("model.aquifer_wel"), "w") as f:
            f.write(text)

    def set_trenches(self, trench_input: NamedTuple):
        """
        Set trenches with a fixed depth on given locations in the parcel.

        Parameters
        ----------
        trench_depth: float
            depth of trenches (m nap)
        trench_locations: np.ndarray
            columns of trenches ()
        """

        trench_depth = trench_input.depth
        trench_locations = trench_input.locations
        trench_resistance = trench_input.resistance

        conductance = self.dz * self.dy / trench_resistance
        level = self.bottom + (self.dz * 0.5)

        trench_columns = np.round(np.array(trench_locations) / self.dx).astype(int)
        trench_layer = int(
            (self.top.size - 1) - np.searchsorted(self.top[::-1], trench_depth)
        )

        indices = []
        for col in trench_columns:
            for lay in range(0, trench_layer + 1):
                indices.append((lay, 0, col))

        stress_period_data = {
            0: [(index, level[index[0]], conductance[index[0]]) for index in indices]
        }

        self.remove_package(self.trn)
        self.trn = flopy.mf6.ModflowGwfdrn(
            self.gwf,
            stress_period_data=stress_period_data,
            save_flows=self.save_flows,
            filename="model.trn",
            pname="trn",
        )

    def set_ditch_boundary(self, ditches: NamedTuple):
        """
        Set ditchs on both sides (left/right) of the parcel.

        Parameters
        ----------
        ditches : :class:`~parcel2d_modflow.components.Ditches`
        """
        ditch_stage = ditches.stage

        level = self.bottom + (self.dz * 0.5)
        # add wet part (riv-package)
        conductance = self.dz * self.dy / ditches.resistance
        time_indices = self.time.searchsorted(ditches.dates)

        # add steady-state step
        time_indices = np.insert(time_indices, 0, 0)
        ditch_stage = np.insert(ditch_stage, 0, ditch_stage[0])

        ix = self.ncol - 1
        stress_period_data = {}
        for time_idx, stage in zip(time_indices, ditch_stage):
            layers = np.flatnonzero((self.bottom < stage) & (self.top > ditches.bottom))

            indices = [(lyr, 0, 0) for lyr in layers] + [(lyr, 0, ix) for lyr in layers]

            stress_period_data[time_idx] = [
                (index, stage, conductance[index[0]], self.bottom[index[0]])
                for index in indices
            ]

        self.remove_package(self.riv)
        self.riv = flopy.mf6.ModflowGwfriv(
            self.gwf,
            stress_period_data=stress_period_data,
            save_flows=self.save_flows,
        )

        stress_period_data = {}
        for time_idx, stage in zip(time_indices, ditch_stage):
            layers = np.flatnonzero(self.bottom > stage)
            if len(layers) == 0:
                layers = np.flatnonzero(self.top > stage)
            indices = [(lyr, 0, 0) for lyr in layers] + [(lyr, 0, ix) for lyr in layers]

            stress_period_data[time_idx] = [
                (index, level[index[0]], conductance[index[0]]) for index in indices
            ]

        self.remove_package(self.riv_drn)
        self.riv_drn = flopy.mf6.ModflowGwfdrn(
            self.gwf,
            stress_period_data=stress_period_data,
            save_flows=self.save_flows,
            filename="model.riv_drn",
            pname="riv_drn",
        )

    def set_ssi_boundary(self, entry_drain_resistance: float, ssi_input: NamedTuple):
        """
        Set drains on given depth and interval in the parcel.

        Parameters
        ----------
        drain_distance: float
            Distance (m) between drains
        drain_depth: float
            depth of drains (m NAP)
        entry_drain_resistance: float
            entry resistance of drains (d)
        drainstage: float
            stage in drain, onderwaterdrainage drainstage == ditch stage,
            drukdrainage drains != ditch stage
        """
        drain_distance = ssi_input.drain_distance
        drain_depth = ssi_input.drain_depth
        drain_stage = ssi_input.drain_stage

        if drain_stage.size != ssi_input.time.size:
            raise ValueError("Drainstage does not equal number of dates. ")

        circumference = np.pi * 0.06  # in m
        conductance = circumference * self.dy / entry_drain_resistance

        time_indices = self.time.searchsorted(ssi_input.time)
        ndrains = (np.round(self.parcel_width / drain_distance) + 1).astype(int)
        drain_cols = np.round(np.linspace(0, (self.ncol - 1), ndrains)).astype(int)[
            1:-1
        ]
        drain_layer = int(
            (self.top.size - 1) - np.searchsorted(self.top[::-1], drain_depth)
        )

        indices = [(drain_layer, 0, int(col)) for col in drain_cols]

        # add steady-state step
        time_indices = np.insert(time_indices, 0, 0)
        drain_stage = np.insert(drain_stage, 0, drain_stage[0])

        stress_period_data = {}
        for per, stage in zip(time_indices, drain_stage):
            stress_period_data[per] = [
                (index, stage, conductance, drain_depth) for index in indices
            ]

        self.remove_package(self.ssi)
        self.ssi = flopy.mf6.ModflowGwfriv(
            self.gwf,
            stress_period_data=stress_period_data,
            save_flows=self.save_flows,
            filename="model.ssi",
            pname="ssi",
        )

    def set_surface_drainage(self):
        """
        To set a surface drain to prevent groundwater levels above the land
        surface

        Parameters
        ----------
        elev: np.ndarray of floats
            the elevation of the drain. (in this case maaiveld)
        time: np.ndarray of datetimes (string, np.datetime64, pd.Timestamp)
            Starting time at which the drain is active.
            Will default to forward filling in time.
        conductance: float
            the hydraulic conductance of the interface between the aquifer and
            the drain
        """
        stress_period_data = {
            0: [
                ((0, 0, i), self.surface, SURFACE_DRAINAGE_CONDUCTANCE)
                for i in range(self.ncol)
            ]
        }
        self.drn = flopy.mf6.ModflowGwfdrn(
            self.gwf,
            stress_period_data=stress_period_data,
            save_flows=self.save_flows,
        )

    def write(self):
        self.sim.write_simulation()
        self.write_recharge_file()
        if self.aquifer_chd is None:
            self.write_wel_file()
        elif self.aquifer_wel is None:
            self.write_chd_file()

    def read_head(self):
        with flopy.utils.HeadFile(self.working_dir / f"{self.name}.hds") as f:
            data = f.get_alldata()

        self.head = xr.DataArray(
            data=data,
            coords={"time": self.time, "z": self.z, "y": [0.5], "x": self.x},
            dims=["time", "z", "y", "x"],
            name="head",
        )

    def read_budgets(self):
        d = read_cbc(
            self.working_dir / f"{self.name}.cbb",
            self.working_dir / "model.dis.grb",
        )
        self.budgets = {}
        for k, v in d.items():
            v = v.assign_coords(time=self.time)
            self.budgets[k] = v

    def run(self):
        self.sim.run_simulation(silent=True)
        self.read_head()
        if self.save_flows:
            self.read_budgets()

    def plot_model(self):
        pcs = flopy.plot.PlotCrossSection(self.gwf, line={"row": 0})
        pcs.plot_grid()
        pcs.plot_bc(name="ditches", package=self.riv, color="blue", kper=0)
        pcs.plot_bc(
            name="ditch drains", package=self.riv_drn, color="darkgreen", kper=0
        )
        # pcs.plot_bc(name="wvp", package=self.aquifer_chd, color="red", kper=0)
        pcs.plot_bc(name="wvp", package=self.aquifer_wel, color="red", kper=0)
        pcs.plot_bc(name="ssi", package=self.ssi, color="red", kper=0)
        pcs.plot_bc(name="trn", package=self.trn, color="purple", kper=0)

    def get_phreatic_head(self):
        """
        Store results of the top layer in a CSV.
        """

        pressure_head = self.head - self.head.z  # NOTE: Maybe use self.bottom
        indices = pressure_head.where(pressure_head > 0).argmin("z")
        phreatic_head = self.head.isel(z=indices).drop_vars("z").squeeze()
        return phreatic_head

    def store_aquifer_head(self, runnr):
        """
        Store results of the seepage layer in a CSV.
        """
        self.output_dir_runs.mkdir(parents=True, exist_ok=True)
        wvp_table = (
            self.head.isel(z=-1, y=0, drop=True)
            .to_dataframe()
            .reset_index()
            .pivot(index="time", columns="x", values="head")
        )
        wvp_table.to_csv(
            self.output_dir_runs.joinpath(f"{runnr}_output_aquifer_head.csv")
        )

    def store_model_inputs_ref(
        self, runnr, thickness, kh, kh_over_kv, specific_yield, lithology, soilcode
    ):
        """
        Store model parameter inputs in a CSV.
        """
        self.output_dir_runs.mkdir(parents=True, exist_ok=True)
        model_inputs = pd.DataFrame(
            data=np.column_stack((kh, kh_over_kv, specific_yield, lithology, soilcode)),
            columns=("kh", "kh_over_kv", "specific_yield", "lithology", "soilcode"),
            index=thickness,
        )

        model_inputs.to_csv(self.output_dir_runs.joinpath(f"{runnr}_input.csv"))

    def store_model_inputs_maatregel(
        self,
        runnr,
        thickness,
        kh,
        kh_over_kv,
        specific_yield,
        dr_res,
        lithology,
        soilcode,
    ):
        """
        Store model parameter inputs in a CSV.
        """
        self.output_dir_runs.mkdir(parents=True, exist_ok=True)
        model_inputs = pd.DataFrame(
            data=np.column_stack(
                (kh, kh_over_kv, specific_yield, dr_res, lithology, soilcode)
            ),
            columns=(
                "kh",
                "kh_over_kv",
                "specific_yield",
                "drain resistance",
                "lithology",
                "soilcode",
            ),
            index=thickness,
        )

        model_inputs.to_csv(self.output_dir_runs.joinpath(f"{runnr}_input.csv"))

    def store_budgets(self, runnr):
        self.output_dir_runs.mkdir(parents=True, exist_ok=True)
        selection = set(self.budgets.keys()).difference(
            ("right-face-flow", "front-face-flow", "npf")
        )

        budget_table = xr.Dataset()
        for package in selection:
            budget_table[package] = (
                self.budgets[package]
                .rename({"layer": "z"})
                .sum(["y"])
                .assign_coords(time=self.time, z=self.z, x=self.x)
            )
        budget_table.to_netcdf(self.output_dir_runs.joinpath(f"{runnr}_budgets.nc"))  #

    def plot_streamlines(self, ax, time, levels=20):
        """
        Make a contour plot of heads (blue) and streamlines (red)

        Parameters
        ----------
        ax: matplotlib.Axes
        time: str, np.datetime, pd.Timestamp
        levels: int
            Number of contour levels to draw. Defaults to 20.
        """
        head = self.head.sel(time=time).isel(y=0, drop=True)
        flow = self.budgets["right-face-flow"].sel(time=time).isel(y=0, drop=True)
        Z, X = np.meshgrid(self.z, self.x, indexing="ij")  # noqa: N806
        streamfunction = flow.copy(data=np.cumsum(flow.values[::-1], axis=0)[::-1])
        ax.contour(
            X + 0.5 * self.dx,
            Z,
            streamfunction.values,
            levels=levels,
            linestyles="solid",
            colors="red",
        )
        ax.contour(X, Z, head.values, levels=levels, linestyles="solid", colors="blue")

    def budget_overview(self) -> pd.DataFrame:
        selection = set(self.budgets.keys()).difference(
            ("right-face-flow", "front-face-flow", "lower-face-flow", "npf")
        )
        df = pd.DataFrame()
        for package in selection:
            df[package] = self.budgets[package].sum(["y", "x", "layer"])
        return df
