from typing import NamedTuple, Union

import numpy as np
import pandas as pd


class SoilDiscretization(NamedTuple):
    nlayers: int
    zmid: np.ndarray
    zbot: np.ndarray
    xcol: np.ndarray


class SubsurfaceStructure(NamedTuple):
    """
    Attributes
    ----------
    thickness: np.ndarray
        vertical cell dimensions (m)
    lithology: np.ndarray
        soil class (1 & 2 = peat, 3 = clay or veraard veen, 4-7 = sand)
        assigned according to thickess distribution
    geology: np.ndarray
        whether cell is above or below the 1st SDL (resistance layer) of LHM input (1 = above, 2 = resitance layer or below)
        assigned according to thickess distribution
    kvalue: np.ndarray
        array with horizontal conductivity
        assigned according to thickess distribution
    """

    thickness: np.ndarray
    lithology: np.ndarray
    geology: np.ndarray
    kvalues: np.ndarray


class Recharge(NamedTuple):
    """
    start: float
        Recharge for steady state run
    series: np.ndarray
        Recharge on a daily basis (d)
    """

    start: float
    series: np.ndarray


class Aquifer(NamedTuple):
    """
    start: float
        aquifer start value for steady state run
    series: np.ndarray
        hydraulic head in aquifer 1
    """

    start: float
    series: np.ndarray


class Ditches(NamedTuple):
    """
    bottom: float
        ditch bottom (m nap)
    resistance: float
        ditchbed resistance (days)
    ditch_stage: np.ndarray
        ditchstage at time steps
    time: Union[str, np.datetime64, pd.Timestamp]
        corresponding time steps
    """

    bottom: float
    resistance: float
    stage: np.ndarray
    dates: Union[str, np.datetime64, pd.Timestamp]


class SsiMeasure(NamedTuple):
    """
    drain_depth: float
        depth of drains (m nap)
    drain_distance: float
        distance between drains (m)
    drain_stage: np.ndarray
        drain stage at time steps (m nap)
    time: Union[str, np.datetime64, pd.Timestamp]
        corresponding timesteps
    """

    drain_depth: float
    drain_distance: float
    drain_stage: np.ndarray
    time: Union[str, np.datetime64, pd.Timestamp]


class Trenches(NamedTuple):
    """
    depth: float
        Depth of trenches in meter +NAP.
    locations: np.ndarray
        Locations of trenches in meter distance along width of a 2D model section.
    resistance: float
        Resistance of the trench in days.
    """

    depth: float
    locations: np.ndarray
    resistance: float
