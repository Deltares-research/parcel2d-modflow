from parcel2d_modflow.base import Parcel
from parcel2d_modflow.config import ModelSettings
from parcel2d_modflow.io.read import (
    read_bro_soilmap,
    read_config,
    read_groundwater_data,
    read_modflow_parameters,
    read_parcels,
)
from parcel2d_modflow.logging import init_logger
from parcel2d_modflow.mf import Modflow
from parcel2d_modflow.modeldata import GroundwaterData, Presets, Soilmap
from parcel2d_modflow.run import run_calibration, run_config, run_parcels

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
    "GroundwaterData",
    "Soilmap",
    "Presets",
    "init_logger",
    "read_bro_soilmap",
    "read_groundwater_data",
    "read_modflow_parameters",
    "read_config",
    "read_parcels",
    "run_config",
    "run_calibration",
    "run_parcels",
]
