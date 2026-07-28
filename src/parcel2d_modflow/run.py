from __future__ import annotations

import itertools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

from parcel2d_modflow.aggregation import calculate_lg3
from parcel2d_modflow.base import Parcel
from parcel2d_modflow.io.read import read_data_from_config
from parcel2d_modflow.io.write import write_batch
from parcel2d_modflow.logging import init_logger
from parcel2d_modflow.mf import Modflow
from parcel2d_modflow.validation import validate_parcels

if TYPE_CHECKING:
    import geopandas as gpd

    from parcel2d_modflow.config import Config, ModelSettings
    from parcel2d_modflow.modeldata import GroundwaterData, ModelData, Soilmap


_WORKER_DATA: ModelData | None = None
_WORKER_SETTINGS: ModelSettings | None = None
_WORKER_MODFLOW_KWARGS: dict[str, Any] | None = None


def _init_worker(data: ModelData, config: Config):
    """
    Initializer function for worker processes in multiprocessing pool. This sets the global
    variables for the worker processes, which allows us to avoid passing large data objects
    (e.g. soilmap) as arguments to the worker function, which would require pickling and
    can lead to significant overhead.

    """
    global _WORKER_DATA, _WORKER_SETTINGS, _WORKER_MODFLOW_KWARGS
    _WORKER_DATA = data
    _WORKER_SETTINGS = config.settings
    _WORKER_MODFLOW_KWARGS = config.modflow_settings.model_dump(exclude="parameters")
    _WORKER_MODFLOW_KWARGS["parameters"] = data.parameters


def run_config(config: Config, *, write_batches: bool = False):
    logger.info(
        f"Run model: multiprocessing={config.run_settings.multiprocessing}, "
        f"log_level={config.run_settings.log_level}"
    )

    if config.run_settings.multiprocessing:
        results = _run_parallel(config, write_batches=write_batches)
    else:
        results = _run_linear(config)

    return results


def _create_batches(size: int, parcels: gpd.GeoDataFrame):
    """
    Helper to create batches from indices of a GeoDataFrame for multiprocessing runs.

    """
    for batch in itertools.batched(parcels.index, size):
        yield list(batch)


def _run_parallel(config: Config, write_batches: bool):
    if write_batches:
        config.output.directory.mkdir(parents=True, exist_ok=True)

    data = read_data_from_config(config)

    num_processes = int(
        min(len(data.parcels), os.cpu_count()) * config.run_settings.multiprocess_scale
    )

    results: list[pd.DataFrame] = []

    logger.info(
        f"Starting runs for {len(data.parcels)} parcels with {num_processes} parallel processes"
    )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=num_processes,
        mp_context=context,
        initializer=_init_worker,
        initargs=(data, config),
    ) as exc:
        tasks = [
            exc.submit(_run_batch, b, config.run_settings.log_level)
            for b in _create_batches(config.run_settings.batch_size, data.parcels)
        ]
        for ii, task in enumerate(as_completed(tasks), start=1):
            try:
                processed_batch = task.result()
            except Exception:
                logger.exception("Error processing batch")
            else:
                results.append(processed_batch)

                if write_batches:
                    write_batch(processed_batch, ii, config.output)

    results = pd.concat(results)

    return results.sort_index()


def _run_batch(indices: list[int], log_level: str):
    if _WORKER_DATA is None or _WORKER_SETTINGS is None:
        raise RuntimeError(
            "Worker data and settings have not been initialized. Cannot run batch."
        )

    logger.info(f"[PID {os.getpid()}] Processing indices: {indices[0]}...{indices[-1]}")
    parcels = _WORKER_DATA.parcels.loc[indices]

    return run_parcels(
        parcels=parcels,
        settings=_WORKER_SETTINGS,
        gw_data=_WORKER_DATA.groundwater,
        soilmap=_WORKER_DATA.soilmap,
        modflow_kwargs=_WORKER_MODFLOW_KWARGS,
    )


def _run_linear(config: Config):
    data = read_data_from_config(config)

    logger.info(f"Starting run for {len(data.parcels)} parcels")

    modflow_kwargs = config.modflow_settings.model_dump(exclude="parameters")
    modflow_kwargs["parameters"] = data.parameters

    return run_parcels(
        parcels=data.parcels,
        settings=config.settings,
        gw_data=data.groundwater,
        soilmap=data.soilmap,
        modflow_kwargs=modflow_kwargs,
    )


@validate_parcels
def _prepare_parcels(
    parcels: gpd.GeoDataFrame, settings: ModelSettings, soilmap: Soilmap
):
    parcel_attributes = parcels.columns
    for p in parcels.itertuples(index=False):
        temp_dir_name = f"{p.name}_{p.soilcode}"
        Path(settings.workdir / temp_dir_name).mkdir(exist_ok=True)
        parcel = Parcel(**dict(zip(parcel_attributes, p)))

        if parcel.soilcode is None:
            parcel.soilcode = soilmap.soilcode_at(parcel.x, parcel.y)

        parcel.discretize_soildepth(settings)
        parcel.soilprofile = soilmap.load_soilprofile(parcel)
        yield parcel


def run_parcels(
    parcels: gpd.GeoDataFrame,
    settings: ModelSettings,
    gw_data: GroundwaterData,
    soilmap: Soilmap,
    modflow_kwargs: dict[str, Any],
):
    module = Modflow(**modflow_kwargs)

    years = settings.date_range.year.unique()

    model_results = []
    prepared_parcels = _prepare_parcels(parcels, settings, soilmap)
    for parcel in prepared_parcels:
        module.initialize(parcel, settings, gw_data)

        try:
            logger.debug(f"Run Parcel ID: {parcel.name}, Soilcode: {parcel.soilcode}")
            ph = module.run(parcel, settings)
        except Exception:
            logger.exception(
                f"Error processing Parcel ID: {parcel.name}, Soilcode: {parcel.soilcode}"
            )
            model_results.append(np.full((len(years), np.nan)))
        else:
            if settings.save_phreatic_head:
                name_soilcode = f"{parcel.name}_{parcel.soilcode}"
                ph.to_netcdf(
                    settings.workdir
                    / f"{name_soilcode}/phreatic_head_{name_soilcode}.nc"
                )

            lg3 = calculate_lg3(ph)
            model_results.append(lg3.mean(dim="runs").values)
        finally:
            module.reset()

    model_results = pd.DataFrame(model_results, columns=years, index=parcels.index)

    return pd.concat([parcels[["name", "soilcode"]], model_results], axis=1)
