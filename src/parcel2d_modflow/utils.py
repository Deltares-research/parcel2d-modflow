import tempfile
from enum import IntEnum
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


class Lithology(IntEnum):
    organic = 1
    clay = 2
    loam = 3
    sand = 4


def geopandas_read(file: str | Path, **kwargs) -> gpd.GeoDataFrame:
    file = Path(file)
    if file.suffix in {".shp", ".gpkg"}:
        return gpd.read_file(file, **kwargs)
    elif file.suffix in {".parquet", ".geoparquet"}:
        return gpd.read_parquet(file, **kwargs)
    else:
        raise ValueError(f"File type {file.suffix} is not supported by geopandas.")


def pandas_read(file: str | Path, **kwargs):
    file = Path(file)
    if file.suffix in {".csv", ".tsv", "*.txt"}:
        return pd.read_csv(file, **kwargs)
    elif file.suffix in {".parquet"}:
        return pd.read_parquet(file, **kwargs)
    else:
        raise ValueError(f"File type {file.suffix} is not supported by pandas.")


def strip_column_units(parameters: pd.DataFrame) -> pd.DataFrame:
    """
    Strip units from the column names of the parameters DataFrame.

    Parameters
    ----------
    parameters : pd.DataFrame
        DataFrame with stochastic parameters to run the Modflow model with.

    Returns
    -------
    pd.DataFrame
        DataFrame with stripped column names.

    """
    parameters.columns = [c.split(" ")[0] for c in parameters.columns]
    return parameters


def create_workdir() -> Path:
    """Create a temporary working directory for monitoring runs."""
    workdir = Path(tempfile.gettempdir()) / "somers_monitoring"
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def determine_lithology_from(soiltable: pd.DataFrame) -> np.ndarray:
    """
    Rowwise determination of lithological classes ("organic", "clay", "loam", "sand")
    based on the table of typical soilprofiles from the BRO Bodemkaart.

    Parameters
    ----------
    soiltable : pd.DataFrame
        Pandas DataFrame of the typical soilprofiles in the BRO Bodemkaart.

    Returns
    -------
    lithology : np.ndarray
        Numpy array of the lithology classes.

    """
    soiltable["sand"] = 100 - soiltable["loamcontent"]

    ## get indices of lithologies
    organic = _is_organic(soiltable)
    sand = _is_sand(soiltable)
    clay = _is_clay(soiltable)

    lithology = np.full(len(soiltable), Lithology.loam)
    lithology[sand] = Lithology.sand
    lithology[clay & ~organic] = Lithology.clay
    lithology[organic] = Lithology.organic
    return lithology


def _is_organic(soiltable: pd.DataFrame):
    """
    Helper function for 'determine_lithology_from' to return a Boolean Series
    of indices where the soiltable maps to organic

    Parameters
    ----------
    soiltable : pd.DataFrame
        See doc 'determine_lithology_from'.

    """
    is_organic = soiltable["organicmattercontent"] > 0.25
    weathered_organic = soiltable["peattype"].str.contains("verweerd", na=False)
    return is_organic & ~weathered_organic


def _is_sand(soiltable: pd.DataFrame):
    """
    Helper function for 'determine_lithology_from' to return a Boolean Series
    of indices where the soiltable maps to sand.

    Parameters
    ----------
    soiltable : pd.DataFrame
        See doc 'determine_lithology_from'.

    """
    return (soiltable["sand"] >= 65) & (soiltable["lutitecontent"] < 35)


def _is_clay(soiltable: pd.DataFrame):
    """
    Helper function for 'determine_lithology_from' to return a Boolean Series
    of indices where the soiltable maps to clay.

    Parameters
    ----------
    soiltable : pd.DataFrame
        See doc 'determine_lithology_from'.

    """
    return soiltable["lutitecontent"] > 50


def lithology_to_geology(lithology: np.ndarray) -> np.ndarray:
    """
    Convert an array of lithology to Holocene and older geology classes based on value and
    order of voxels. When lithology is sand, is not on top and the remaining lithologies
    below are sand it assumed to be not Holocene.

    Parameters
    ----------
    lithology : np.ndarray
        Numpy array of lithology classes.

    Returns
    -------
    geology : np.ndarray
        Numpy array of geology classes (1: Holocene, 2: older).

    """
    geology = np.full_like(lithology, 1, dtype=int)

    if (lithology == Lithology.sand).any():
        # Always assume top lithology is Holocene so start iteration from 1
        for ii in range(1, len(lithology)):
            if (lithology[ii:] == Lithology.sand).all():
                geology[ii:] = 2
                break

    return geology
