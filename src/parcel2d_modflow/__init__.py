from parcel2d_modflow._io.read import (
    read_bro_soilmap,
    read_lhm_data,
    read_modflow_parameters,
)
from parcel2d_modflow.base import ModelSettings, Parcel
from parcel2d_modflow.mf.module import Modflow
from parcel2d_modflow.modeldata import LhmData, Presets, Soilmap

__version__ = "0.1.0"

__doc__ = """
This package contains the 2D groundwater flow model for organic parcels used in the
somers-peatparcel2d-aap package, implemented using the MODFLOW framework. The model can
be used independently of the somers-peatparcel2d-aap package, but is designed to be used
in conjunction with it. As such, it contains a number of classes and functions that are
related to the somers-peatparcel2d-aap package and documentation often refers to SOMERS.
"""

__all__ = [
    "ModelSettings",
    "Parcel",
    "Modflow",
    "LhmData",
    "Soilmap",
    "Presets",
    "read_bro_soilmap",
    "read_lhm_data",
    "read_modflow_parameters",
]
