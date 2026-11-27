# %%
from pathlib import Path

import geopandas as gpd
import hydroeval as he
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from parcel2d_modflow.io.postgis import initialize_database, read_table_from_database
from parcel2d_modflow.io.read import read_modflow_parameters

#%%
def calculate_weighted_performance(performance_wells: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weighted run performance based on all wells in a transect.

    Weights are based on the number of valid runs per well.

    Parameters
    ----------
    performance_wells : pd.DataFrame
        DataFrame containing per-well performance metrics with a MultiIndex
        (piez_name, runs) and columns ["r", "alpha", "beta", "rmse", "count"].

    Returns
    -------
    pd.DataFrame
        Weighted performance metrics per run with index "runs" and columns
        ["r", "alpha", "beta", "kge", "rmse", "count"].
    """
    weighted_performance = pd.DataFrame()

    weighted_performance["r"] = performance_wells.groupby(["runs"]).apply(
        lambda x: np.average(x.r, weights=x["count"])
    )
    weighted_performance["alpha"] = performance_wells.groupby(["runs"]).apply(
        lambda x: np.average(x.alpha, weights=x["count"])
    )
    weighted_performance["beta"] = performance_wells.groupby(["runs"]).apply(
        lambda x: np.average(x.beta, weights=x["count"])
    )
    weighted_performance["kge"] = weighted_performance.apply(
        lambda x: (
            1 - np.sqrt((x.alpha - 1) ** 2 + (x.beta - 1) ** 2 + (x.r - 1) ** 2)
        ),
        axis=1,
    )
    weighted_performance["rmse"] = performance_wells.groupby(["runs"]).apply(
        lambda x: np.average(x.rmse, weights=x["count"])
    )
    weighted_performance["count"] = performance_wells.groupby(["runs"]).apply(
        lambda x: np.mean(x["count"])
    )

    return weighted_performance


def evaluate_simulation(simulation: xr.DataArray, evaluation: xr.DataArray) -> pd.DataFrame:
    """
    Evaluate simulation performance against observed data using KGE and RMSE.

    Parameters
    ----------
    simulation : xr.DataArray
        Simulated groundwater head with dimensions (time, runs).
    evaluation : xr.DataArray
        Observed groundwater head with dimension (time).

    Returns
    -------
    pd.DataFrame
        DataFrame containing per-run performance metrics with index "runs" and columns
        ["kge", "r", "alpha", "beta", "rmse", "count"].
    """
    kge, r, alpha, beta = he.evaluator(
        he.kge,
        simulations=simulation.values,
        evaluation=evaluation.values,
        axis=0,
    )
    rmse = he.evaluator(
        he.rmse,
        simulations=simulation.values,
        evaluation=evaluation.values,
        axis=0,
    )

    count = len(kge)

    return pd.DataFrame(
        data={
            "kge": kge,
            "r": r,
            "alpha": alpha,
            "beta": beta,
            "rmse": rmse,
            "count": count,
        },
        index=simulation.runs,
    )


def plot_piezometer_simulation_envelope(
    sim_data: xr.DataArray,
    piez_data: xr.DataArray,
    performance_well: pd.DataFrame,
    title: str,
    output_path: Path,
):
    """
    Plot observed head, best KGE simulation, and top-30 simulation envelope for a piezometer.

    Parameters
    ----------
    sim_data : xr.DataArray
        Simulated groundwater head with dimensions (time, runs).
    piez_data : xr.DataArray
        Observed groundwater head with dimension (time).
    performance_well : pd.DataFrame
        DataFrame containing per-run performance metrics with index "runs" and columns
        ["kge", "r", "alpha", "beta", "rmse", "count"].
    title : str
        Title for the plot.
    output_path : Path
        Path to save the generated plot.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    sim_plot = sim_data.copy()

    best_30 = performance_well.nlargest(30, "kge").index.get_level_values("runs")

    lower = sim_plot.sel(runs=best_30).min("runs")
    upper = sim_plot.sel(runs=best_30).max("runs")

    performance_index = performance_well.index.get_level_values("runs")
    performance_well = performance_well.copy()
    performance_well.index = performance_index

    best_run = performance_well["kge"].idxmax()
    if best_run in sim_plot.runs:
        best_series = sim_plot.sel(runs=best_run)
    else:
        best_series = sim_plot.sel(runs=str(best_run))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        sim_plot.time,
        lower,
        upper,
        color="tab:orange",
        alpha=0.2,
        label="Simulation best 30 input parameters",
    )
    ax.plot(sim_plot.time, piez_data, color="black", linewidth=1.5, label="Observed")
    ax.plot(
        sim_plot.time,
        best_series,
        color="tab:blue",
        linewidth=1.5,
        label=f"Best simulation (KGE={performance_well.loc[best_run, 'kge']:.2f})",
    )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Head [m NAP]")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _as_well_list(value) -> list[str]:
    """
    Convert parcel well_id field to a clean list of well identifiers.

    Parameters
    ----------
    value : Any
        Value representing well identifiers, which can be a single identifier,
        a list, tuple, numpy array, or pandas Series.

    Returns
    -------
    list[str]
        List of well identifiers as strings, excluding any missing values.
    """
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        wells = list(value)
    elif pd.isna(value):
        wells = []
    else:
        wells = [value]

    return [str(well) for well in wells if pd.notna(well)]


def _read_piezometer_metadata(dotenv_path: Path) -> pd.DataFrame:
    """
    Query the reference piezometer metadata used to locate each well in a
    parcel's simulated profile (via its ``distance_to_ditch_m``).

    Parameters
    ----------
    dotenv_path : Path
        Path to the .env file for the database connection.

    Returns
    -------
    pd.DataFrame
        Piezometer metadata, including "well_id" and "distance_to_ditch_m".

    """
    connection = initialize_database(dotenv_path)
    try:
        return read_table_from_database(
            engine=connection,
            schema_name="metadata_gecontroleerd_2024",
            table_name="kalibratie",
            user_query=(
                "WHERE selection = 'yes' "
                "AND parcel_type = 'ref' "
                "AND transect IS NOT NULL"
            ),
        )
    finally:
        connection.dispose()


def _score_well(
    sim_da: xr.DataArray,
    observations: xr.DataArray,
    piez_meta_df: pd.DataFrame,
    parcel_name: str,
    piez_name: str,
    fig_folder: Path,
    plotting: bool,
) -> pd.DataFrame | None:
    """
    Score one piezometer within a parcel's simulation.

    Returns None if the well has no metadata, or no timesteps where both the
    simulated and observed head are present for every run.

    Parameters
    ----------
    sim_da : xr.DataArray
        Simulated groundwater head for the parcel, with dimensions
        (runs, time, x).
    observations : xr.DataArray
        Observed groundwater head for all wells, with dimensions
        (well_id, time).
    piez_meta_df : pd.DataFrame
        Piezometer metadata, including "well_id" and "distance_to_ditch_m".
    parcel_name : str
        Name of the parcel the well belongs to, used for plot titles/paths
        and log messages.
    piez_name : str
        Identifier of the piezometer to score.
    fig_folder : Path
        Directory to save the per-well plot to, if `plotting` is True.
    plotting : bool
        Whether to generate a simulation envelope plot for this well.

    Returns
    -------
    pd.DataFrame or None
        Per-run performance metrics with a MultiIndex (piez_name, runs), or
        None if the well could not be scored.

    """
    piez_meta = piez_meta_df[piez_meta_df.well_id == piez_name]
    if piez_meta.empty:
        return None

    piez_at_sim_loc = observations.sel(well_id=piez_name)
    sim_at_piez_loc = sim_da.sel(
        x=piez_meta.iloc[0].distance_to_ditch_m, method="nearest"
    ).compute()

    # Keep only timesteps where every run's simulated value and the
    # observation are present, since the resulting `count` is later used to
    # weight this well against the parcel's other wells.
    both_times = sim_at_piez_loc.notnull().all("runs") & piez_at_sim_loc.notnull()
    sim_data = sim_at_piez_loc.where(both_times, drop=True)
    piez_data = piez_at_sim_loc.where(both_times, drop=True)

    if sim_data.sizes.get("time", 0) == 0 or sim_data.sizes.get("runs", 0) == 0:
        print(f"No overlapping data for piezometer {piez_name} in parcel {parcel_name}")
        return None

    sim_data = sim_data.transpose("time", "runs")

    performance_well = evaluate_simulation(sim_data, piez_data).sort_index()
    performance_well.index = pd.MultiIndex.from_product(
        [[piez_name], performance_well.index], names=["piez_name", "runs"]
    )

    if plotting:
        plot_piezometer_simulation_envelope(
            sim_data=sim_data,
            piez_data=piez_data,
            performance_well=performance_well.copy(),
            title=f"parcel: {parcel_name}, piez: {piez_name}",
            output_path=(
                fig_folder / f"{parcel_name}_{piez_name}_simulation_envelope.png"
            ),
        )

    return performance_well


def _score_parcel(
    parcel,
    method_dir: Path,
    observations: xr.DataArray,
    piez_meta_df: pd.DataFrame,
    fig_folder: Path,
    plotting: bool,
) -> pd.DataFrame | None:
    """
    Score every piezometer of one parcel.

    Returns None if the parcel's simulation file is missing, or none of its
    wells could be scored (see `_score_well`).

    Parameters
    ----------
    parcel : NamedTuple
        One row of the parcel GeoDataFrame, as produced by `itertuples`.
    method_dir : Path
        Directory containing the method-specific simulation results.
    observations : xr.DataArray
        Observed groundwater head for all wells, with dimensions
        (well_id, time).
    piez_meta_df : pd.DataFrame
        Piezometer metadata, including "well_id" and "distance_to_ditch_m".
    fig_folder : Path
        Directory to save per-well plots to, if `plotting` is True.
    plotting : bool
        Whether to generate simulation envelope plots for this parcel's
        wells.

    Returns
    -------
    pd.DataFrame or None
        Combined per-well, per-run performance metrics for the parcel, or
        None if the parcel could not be scored.

    """
    parcel_name = f"{parcel.name}_{parcel.soilcode}"
    sim_data_fn = method_dir / parcel_name / f"phreatic_head_{parcel_name}.nc"
    if not sim_data_fn.exists():
        return None

    sim_da = xr.open_dataarray(sim_data_fn).drop_duplicates("runs")

    performance_transect = pd.DataFrame()
    for piez_name in _as_well_list(parcel.well_id):
        performance_well = _score_well(
            sim_da, observations, piez_meta_df, parcel_name, piez_name,
            fig_folder, plotting,
        )
        if performance_well is not None:
            performance_transect = pd.concat([performance_transect, performance_well])

    return performance_transect if not performance_transect.empty else None


def _write_transect_results(
    parcel_name: str, performance_transect: pd.DataFrame, transect_dir: Path
) -> pd.DataFrame:
    """
    Write a parcel's per-well and weighted performance to CSV.

    Parameters
    ----------
    parcel_name : str
        Name of the parcel, used in the output file names.
    performance_transect : pd.DataFrame
        Combined per-well, per-run performance metrics for the parcel, as
        returned by `_score_parcel`.
    transect_dir : Path
        Directory to write the parcel's CSV files to.

    Returns
    -------
    pd.DataFrame
        The parcel's weighted performance, tagged with "parcel_name", for
        later aggregation across all parcels.

    """
    performance_transect.to_csv(
        transect_dir / f"{parcel_name}_all_wells_performance.csv"
    )

    weighted_performance = calculate_weighted_performance(performance_transect)
    weighted_performance.to_csv(
        transect_dir / f"{parcel_name}_weighted_performance.csv"
    )

    weighted_with_parcel = weighted_performance.copy()
    weighted_with_parcel["parcel_name"] = parcel_name
    return weighted_with_parcel.reset_index()


def _select_best_parameters(
    global_weighted_performance: pd.DataFrame,
    modflow_params: pd.DataFrame,
    n_best: int = 30,
) -> pd.DataFrame:
    """
    Select the Modflow parameters of the `n_best` highest-KGE runs.

    Parameters
    ----------
    global_weighted_performance : pd.DataFrame
        Weighted performance metrics per run, as returned by
        `calculate_weighted_performance`.
    modflow_params : pd.DataFrame
        Modflow parameters for all runs, including "runnr".
    n_best : int, optional
        Number of best-performing runs to select, by default 30.

    Returns
    -------
    pd.DataFrame
        Modflow parameters of the `n_best` highest-KGE runs, indexed by
        "runnr".

    """
    # `sort_values` uses a stable sort so that, for equal KGE scores at the
    # n_best boundary, the run with the lowest run number is kept.
    best_parameters_all = global_weighted_performance.sort_index().sort_values(
        by=["kge"], ascending=False, kind="stable"
    )
    best_parameter_selection = best_parameters_all.iloc[:n_best]

    best_modflow_parameters = modflow_params[
        modflow_params.runnr.isin(best_parameter_selection.index.astype(int))
    ]
    return best_modflow_parameters.set_index("runnr")


def analyze_calibration(
    method_dir: Path,
    parcel_gdf_fn: Path,
    obs_nc_fn: Path,
    modflow_params_path: Path,
    dotenv_path: Path,
    plotting: bool = True,
) -> None:
    """
    Analyze calibration runs one method, optionally producing plots, using files and PostGIS.

    Scores every parcel's wells (see `_score_parcel`), writes per-parcel
    results (see `_write_transect_results`), then combines all parcels into
    global performance files and a best-30-parameters selection (see
    `_select_best_parameters`).

    Parameters
    ----------
    method_dir : Path
        Directory containing the method-specific simulation results.
    parcel_gdf_fn : Path
        Path to the parcel GeoDataFrame file (Parquet format).
    obs_nc_fn : Path
        Path to the observed data NetCDF file.
    modflow_params_path : Path
        Path to the Modflow parameters file.
    dotenv_path : Path
        Path to the .env file for database connection.
    plotting : bool, optional
        Whether to generate plots, by default True.

    Returns
    -------
    None

    """
    parcel_gdf = gpd.read_parquet(parcel_gdf_fn)

    kalibratie_dir = method_dir / "kalibratie"
    kalibratie_dir.mkdir(exist_ok=True, parents=True)

    fig_folder = kalibratie_dir / "plots"
    if plotting:
        fig_folder.mkdir(parents=True, exist_ok=True)

    transect_dir = kalibratie_dir / "transect_performance"
    transect_dir.mkdir(parents=True, exist_ok=True)

    modflow_params = read_modflow_parameters(modflow_params_path)
    observations = xr.open_dataarray(obs_nc_fn)
    piez_meta_df = _read_piezometer_metadata(dotenv_path)

    performance_global = pd.DataFrame()
    all_transect_weighted = []

    for parcel in tqdm(parcel_gdf.itertuples(index=False), total=len(parcel_gdf)):
        performance_transect = _score_parcel(
            parcel, method_dir, observations, piez_meta_df, fig_folder, plotting
        )
        if performance_transect is None:
            continue

        parcel_name = f"{parcel.name}_{parcel.soilcode}"
        all_transect_weighted.append(
            _write_transect_results(parcel_name, performance_transect, transect_dir)
        )
        # Match the original workflow: add full transect performance to global pool.
        performance_global = pd.concat([performance_global, performance_transect])

    if performance_global.empty:
        raise ValueError(f"No well performance data found in {method_dir}")

    performance_global.to_csv(kalibratie_dir / "all_piez_performance.csv")

    global_weighted_performance = calculate_weighted_performance(performance_global)
    global_weighted_performance.to_csv(
        kalibratie_dir / "global_weighted_performance.csv"
    )

    best_modflow_parameters = _select_best_parameters(
        global_weighted_performance, modflow_params
    )
    best_modflow_parameters.to_csv(kalibratie_dir / "best_kali_parameters.csv")

    if all_transect_weighted:
        pd.concat(all_transect_weighted, ignore_index=True).to_csv(
            transect_dir / "all_transects_weighted_performance.csv", index=False
        )

# %%
