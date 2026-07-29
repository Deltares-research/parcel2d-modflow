from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd

    from parcel2d_modflow.config import OutputSettings


def write_batch(batch: gpd.GeoDataFrame, n: int, output: OutputSettings) -> None:
    """
    Write a processed batch from a multiprocessing run to a separate file.

    Parameters
    ----------
    batch : gpd.GeoDataFrame
        The GeoDataFrame containing the processed batch.
    n : int
        The batch number, used for naming the output file.
    output : OutputSettings
        The output settings specifying the directory, prefix, and format for the output
        file.

    """
    outfile = output.directory / f"{output.prefix}_{n:05d}.{output.format}"
    batch.to_parquet(outfile)
