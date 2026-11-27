from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize, TwoSlopeNorm

_METRICS = ["kge", "rmse", "r", "alpha", "beta"]


def _read_piez_performance_csv(
    csv_path: Path, method: str, label: str
) -> pd.DataFrame:
    """
    Read and process piezometer performance CSV for a specific calibration method.
    
    Parameters
    ----------
    csv_path : Path
        Path to the piezometer performance CSV file.
    method : str
        Calibration method key or identifier.
    label : str
        Human-readable label for the calibration method.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the processed piezometer performance with added method information.
    """
    perf = pd.read_csv(csv_path)
    perf = perf.rename(columns={perf.columns[0]: "piez_name", perf.columns[1]: "runs"})
    perf["method_key"] = method
    perf["method_label"] = label
    return perf


def _read_parameter_tables(parameter_files: list[Path]) -> list[pd.DataFrame]:
    """Read per-method best-30 parameter tables for the comparison plots."""
    return [pd.read_csv(path, index_col=0) for path in parameter_files]


def plot_parameter_histograms(
    tables: list[pd.DataFrame], labels: list[str], output_dir: Path
) -> None:
    """
    Plot per-parameter histograms for the reference and calibration methods.

    Parameters
    ----------
    tables : list[pd.DataFrame]
        Best-30 parameter tables per method, as returned by
        `_read_parameter_tables`.
    labels : list[str]
        Human-readable labels corresponding to each table.
    output_dir : Path
        Directory to save the generated plot to.

    Returns
    -------
    None
    """
    columns = tables[0].columns.tolist()
    limits = {
        col: (
            float(min(table[col].min() for table in tables)),
            float(max(table[col].max() for table in tables)),
        )
        for col in columns
    }
    bins = {col: np.linspace(*limits[col], 16) for col in columns}

    fig, axes = plt.subplots(
        len(tables),
        len(columns),
        sharey=True,
        sharex="col",
        figsize=(15, 3 * len(tables)),
        squeeze=False,
    )
    for row, (table, label) in enumerate(zip(tables, labels)):
        for col_idx, col in enumerate(columns):
            ax = axes[row, col_idx]
            ax.hist(
                table[col].dropna(),
                bins=bins[col],
                color="tab:blue",
                alpha=0.35,
                edgecolor="tab:blue",
                linewidth=0.8,
            )
            ax.set_xlim(*limits[col])
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(col)
            if row == len(tables) - 1:
                ax.set_xlabel(col)
            if col_idx == 0:
                ax.set_ylabel(label)
    fig.suptitle("Best 30 parameters per calibration method vs original best 30", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "best_parameters_comparison.png", dpi=150)
    plt.close(fig)


def plot_parameter_summary(
    tables: list[pd.DataFrame], labels: list[str], output_dir: Path
) -> None:
    """
    Plot per-parameter distribution summaries (violin + scatter) for the
    reference and calibration methods.

    Parameters
    ----------
    tables : list[pd.DataFrame]
        Best-30 parameter tables per method, as returned by
        `_read_parameter_tables`.
    labels : list[str]
        Human-readable labels corresponding to each table.
    output_dir : Path
        Directory to save the generated plot to.

    Returns
    -------
    None
    """
    columns = tables[0].columns.tolist()
    fig, axes = plt.subplots(
        1, len(columns), figsize=(15, 4.8), sharey=False, constrained_layout=True
    )
    rng = np.random.default_rng(42)
    xpos = np.arange(len(labels))
    for ax, col in zip(np.ravel(axes), columns):
        series = [table[col].dropna().to_numpy(dtype=float) for table in tables]
        violin = ax.violinplot(
            series, positions=xpos, widths=0.8, showmeans=False,
            showmedians=False, showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor("tab:blue")
            body.set_edgecolor("tab:blue")
            body.set_alpha(0.22)
        for x, values in zip(xpos, series):
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(
                x + jitter, values, s=14, color="tab:blue", alpha=0.55, linewidths=0,
            )
            ax.hlines(np.nanmedian(values), x - 0.22, x + 0.22, color="tab:blue", linewidth=1.6)
        ax.set_title(f"{col} distribution (n=30)")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(alpha=0.25, axis="y")
    fig.savefig(output_dir / "best_parameters_summary.png", dpi=150)
    plt.close(fig)


def plot_performance_heatmap(
    perf_best: pd.DataFrame,
    method_order: list[str],
    piez_order: list[str],
    output_dir: Path,
) -> None:
    """
    Plot per-well performance heatmaps for different calibration methods.

    Parameters
    ----------
    perf_best : pd.DataFrame
        Best performance metrics per piezometer and method.
    method_order : list[str]
        Method identifiers, in the order they should appear in the plot.
    piez_order : list[str]
        Piezometer identifiers, in the order they should appear in the plot.
    output_dir : Path
        Directory to save the generated plot to.

    Returns
    -------
    None
    """
    metrics = ["kge", "rmse", "r", "alpha", "beta"]
    fig, axes = plt.subplots(
        nrows=len(_METRICS), ncols=1,
        figsize=(3 * len(method_order), 0.6 * len(piez_order)),
        constrained_layout=True,
    )
    rmse_max = float(
        np.nan_to_num(
            np.nanpercentile(perf_best["rmse"].to_numpy(dtype=float), 95),
            nan=1.0, posinf=1.0, neginf=1.0,
        )
    )
    norms = {
        "kge": TwoSlopeNorm(vmin=-1.0, vcenter=-0.41, vmax=1.0),
        "rmse": Normalize(vmin=0.0, vmax=rmse_max + 1e-9),
        "r": TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
        "alpha": TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0),
        "beta": TwoSlopeNorm(vmin=1.0, vcenter=1.5, vmax=2.0),
    }
    for ax, metric in zip(np.ravel(axes), _METRICS):
        grid = (
            perf_best.pivot_table(
                index="piez_name", columns="method_label", values=metric, aggfunc="mean"
            ).reindex(index=piez_order, columns=method_order)
        )
        cmap = "RdYlGn_r" if metric == "rmse" else "RdYlGn"
        image = ax.imshow(grid.values.astype(float), aspect="auto", cmap=cmap, norm=norms[metric])
        ax.set_title(f"{metric.upper()} per piezometer and method (best run by KGE)")
        ax.set_ylabel("Piezometer")
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels(grid.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels(grid.index)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label=metric)
    fig.savefig(output_dir / "compare_heatmap.png", dpi=180)
    plt.close(fig)


def plot_performance_boxplot(
    perf_best: pd.DataFrame, method_order: list[str], output_dir: Path
) -> None:
    """
    Plot per-metric performance distributions across piezometers, for
    different calibration methods.

    Parameters
    ----------
    perf_best : pd.DataFrame
        Best performance metrics per piezometer and method.
    method_order : list[str]
        Method identifiers, in the order they should appear in the plot.
    output_dir : Path
        Directory to save the generated plot to.

    Returns
    -------
    None
    """
    metrics = ["kge", "rmse", "r", "alpha", "beta"]
    fig, axes = plt.subplots(
        1, len(_METRICS), figsize=(4.2 * len(_METRICS), 5), constrained_layout=True
    )
    for ax, metric in zip(np.ravel(axes), _METRICS):
        series = [
            perf_best.loc[perf_best["method_label"] == method, metric].dropna().values
            for method in method_order
        ]
        ax.boxplot(series, tick_labels=method_order, patch_artist=True, showfliers=False)
        ax.set_title(f"{metric.upper()} distribution across piezometers")
        ax.set_xlabel("Method")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.savefig(output_dir / "compare_boxplot.png", dpi=180)
    plt.close(fig)


class _MethodSimulation(NamedTuple):
    """Weekly-resampled simulation result for one method at one well."""

    weekly_mean: xr.DataArray
    best_run: str
    performance: pd.DataFrame
    data: xr.DataArray


def _method_at_well(df: pd.DataFrame, label: str, piez_name) -> pd.DataFrame:
    """Select one method's rows for one well from a performance table."""
    return df[
        (df["method_label"] == label) & (df["piez_name"].astype(str) == str(piez_name))
    ]


def plot_well_comparison(
    perf_all: pd.DataFrame,
    perf_best: pd.DataFrame,
    method_order: list[str],
    piez_order: list[str],
    simulation_files: dict[str, Path],
    output_dir: Path,
) -> None:
    """Plot weekly observed vs. best-simulated heads per well, across methods."""
    best_label = perf_best.groupby("method_label")["kge"].median().idxmax()
    for piez_name in piez_order:
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=False)
        fig.subplots_adjust(right=0.78)
        simulated_by_method: dict[str, _MethodSimulation] = {}
        observed_weekly = None
        kge_lines = []
        for label in method_order:
            method_perf = _method_at_well(perf_best, label, piez_name)
            if method_perf.empty:
                continue
            method = method_perf["method_key"].iloc[0]
            best_run = str(method_perf["runs"].iloc[0])
            with xr.open_dataarray(simulation_files[method]) as data:
                da = data.sel(well_id=piez_name).load()
            da = da.assign_coords(runs=da.runs.astype(str))
            observed_weekly = da.sel(runs="obs").resample(time="W").mean()
            weekly_mean = da.sel(runs=best_run).resample(time="W").mean()
            simulated_by_method[label] = _MethodSimulation(
                weekly_mean, best_run, method_perf, da
            )

        sim_presence = xr.concat(
            [sim.weekly_mean.notnull() for sim in simulated_by_method.values()],
            dim="method",
        ).any("method")
        observed = observed_weekly.where(sim_presence)
        ax.plot(observed.time.values, observed.values, color="black", linewidth=1.8, label="Observed")
        for label, sim in simulated_by_method.items():
            if label == best_label:
                top_runs = (
                    _method_at_well(perf_all, label, piez_name)
                    .sort_values("kge", ascending=False).head(30)["runs"]
                    .astype(str).tolist()
                )
                top = sim.data.sel(runs=top_runs).resample(time="W").mean()
                lower, upper = top.min("runs"), top.max("runs")
                ax.fill_between(
                    lower.time.values, lower.values, upper.values,
                    color="tab:orange", alpha=0.22,
                    label="Top-30 min-max envelope (weekly)",
                )
            ax.plot(
                sim.weekly_mean.time.values, sim.weekly_mean.values, linewidth=1.2,
                label=f"{label} (run {sim.best_run}, weekly mean)",
            )
            kge_lines.append(f"{label}: {sim.performance['kge'].iloc[0]:.3f}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Head [m NAP]")
        ax.grid(alpha=0.25)
        ax.legend(loc=2, bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9)
        ax.text(
            1.02, 0.02, "Best-run KGE\n" + "\n".join(kge_lines),
            transform=ax.transAxes, va="bottom", ha="left",
        )
        fig.suptitle(
            f"Piezometer {piez_name}: observed vs best simulation per method | "
            f"Best median-KGE method: {best_label}", fontsize=13,
        )
        fig.savefig(output_dir / f"{piez_name}_best_sim_vs_obs.png", dpi=180)
        plt.close(fig)


def plot_calibration_comparison(
    method_dirs: dict[str, Path],
    simulation_files: dict[str, Path],
    labels: dict[str, str],
    original_parameters: Path,
    output_dir: Path,
) -> None:
    """
    Generate parameter, performance, and per-well comparison plots for different calibration methods.

    Parameters
    ----------
    method_dirs : dict[str, Path]
        Dictionary mapping method identifiers to their corresponding directories containing simulation results.
    simulation_files : dict[str, Path]
        Dictionary mapping method identifiers to their corresponding simulation NetCDF files.
    labels : dict[str, str]
        Dictionary mapping method identifiers to human-readable labels.
    original_parameters : Path
        Path to the CSV file containing the original best parameters.
    output_dir : Path
        Directory to save the generated plots.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(method_dirs)
    parameter_files = [original_parameters] + [
        method_dirs[method] / "kalibratie" / "best_kali_parameters.csv"
        for method in methods
    ]
    parameter_tables = _read_parameter_tables(parameter_files)
    parameter_labels = ["Original best 30"] + [labels[method] for method in methods]
    plot_parameter_histograms(parameter_tables, parameter_labels, output_dir)
    plot_parameter_summary(parameter_tables, parameter_labels, output_dir)

    perf_all = pd.concat(
        [
            _read_piez_performance_csv(
                method_dirs[method] / "kalibratie" / "all_piez_performance.csv",
                method, labels[method],
            )
            for method in methods
        ],
        ignore_index=True,
    )
    best_idx = perf_all.groupby(["method_label", "piez_name"])["kge"].idxmax()
    perf_best = perf_all.loc[best_idx].copy()
    method_order = (
        perf_best[["method_label", "method_key"]].drop_duplicates()
        .sort_values("method_key")["method_label"].tolist()
    )
    piez_order = (
        perf_best.pivot_table(
            index="piez_name", columns="method_label", values="kge", aggfunc="mean"
        ).mean(axis=1).sort_values(ascending=False).index.tolist()
    )
    plot_performance_heatmap(perf_best, method_order, piez_order, output_dir)
    plot_performance_boxplot(perf_best, method_order, output_dir)
    plot_well_comparison(
        perf_all, perf_best, method_order, piez_order, simulation_files, output_dir
    )
