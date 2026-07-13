from pathlib import Path
from typing import TYPE_CHECKING

from parcel2d_modflow.base import Parcel
from parcel2d_modflow.validation import validate_parcels

if TYPE_CHECKING:
    import geopandas as gpd

    from parcel2d_modflow.base import ModelSettings
    from parcel2d_modflow.modeldata import Soilmap


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


def run_parcels(parcels):
    pass
