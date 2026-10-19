#%%
from functools import cache
from typing import NamedTuple

import numpy as np


class EVTProfile(NamedTuple):
    """
    EVTProfile represents the evapotranspiration profile.

    Parameters
    ----------
    evt_ext_depth : float
        The external depth of the evapotranspiration profile.
    rel_segment_bottom : np.ndarray
        The relative bottom positions of the segments.
    evt_fraction_values : np.ndarray
        The evapotranspiration fraction values for each segment.
    """

    evt_ext_depth: float
    rel_segment_bottom: np.ndarray
    evt_fraction_values: np.ndarray


def evt_woerkom():
    start = 0.4
    end = 3.1
    p = 0.2

    evt_fraction_values = [1, 0.9, 0.7, 0.1]  # [1.8, 1.43, 1.18, 0.9, 0.8, 0.7, 0.1]

    rel_segment_rate = np.array(evt_fraction_values)
    segment_bottom = start + (end - start) * (1 - rel_segment_rate ** (1 / p))
    rel_segment_rate = np.r_[[1], rel_segment_rate]
    segment_bottom = np.r_[[0], segment_bottom]
    return rel_segment_rate, segment_bottom


def evt_combi():
    rel_segment_rate = np.array([1.8, 1.43, 1.18, 0.9, 0.8, 0.7, 0.1])
    gw_stand = np.array([0.1, 0.2, 0.4, 0.7, 1.0, 2.5, 3.1])
    return rel_segment_rate, gw_stand


def evt_boon():
    gw_stand = np.array([0.01, 0.1, 0.2, 0.4, 0.8, 1.2, 3.0])
    #email correspondentie met mevr. Boon met onderwerp 'RE: Verdampings-fit' van 28-09-2026
    fraction = 0.8355 + 0.4689 * np.exp(-gw_stand*100 / 12.9768)
    return fraction, gw_stand

EVT_METHODS = {
    "woerkom": evt_woerkom,
    "combi": evt_combi,
    "boon": evt_boon,
}

@cache
def calc_evt_profile(method_name: str) -> EVTProfile:
    evt_method = EVT_METHODS.get(method_name, evt_woerkom)
    rel_segment_rate, abs_segment_bottom = evt_method()

    evt_ext_depth = np.round(abs_segment_bottom.max(), 2)
    rel_segment_bottom = np.round(abs_segment_bottom / evt_ext_depth, 2)
    rel_segment_bottom = np.minimum(rel_segment_bottom, 0.999999)
    rel_segment_bottom = np.maximum(rel_segment_bottom, 0.000001)

    return EVTProfile(
        evt_ext_depth=evt_ext_depth,
        rel_segment_bottom=rel_segment_bottom,
        evt_fraction_values=rel_segment_rate,
    )

# %%
