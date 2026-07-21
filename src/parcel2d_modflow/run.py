import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any

from parcel2d_modflow.base import Parcel
from parcel2d_modflow.mf import Modflow
from parcel2d_modflow.validation import validate_parcels

if TYPE_CHECKING:
    import geopandas as gpd

    from parcel2d_modflow.config import Config, ModelSettings
    from parcel2d_modflow.modeldata import GroundwaterData, Soilmap


def run_config(config: Config):
    pass


def _create_batches(size: int, parcels: gpd.GeoDataFrame):
    """
    Helper to create batches from indices of a GeoDataFrame for multiprocessing runs.

    """
    for batch in itertools.batched(parcels.index, size):
        yield list(batch)


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
    modflow_kwargs: dict[str, Any] = None,
    **kwargs,
):
    parcels = _prepare_parcels(parcels, settings, soilmap)

    modflow_kwargs = modflow_kwargs or {}
    module = Modflow(**modflow_kwargs)
    for parcel in parcels:
        module.initialize(parcel, settings, gw_data)
        module.run(parcel, settings)
        module.reset()

        # Aggregate to LG3 results

    return
