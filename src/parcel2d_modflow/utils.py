import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


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
