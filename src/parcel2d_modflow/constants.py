from typing import NamedTuple


class ParameterCorrectionCurve(NamedTuple):
    """
    Attributes
    ----------
    a: float
        parameter a
    b: float
        parameter b
    c: float
        parameter c
    d: float
        parameter d
    """

    a: float = 0.5818894418751023
    b: float = 0.01721420632103996
    c: float = 2.3031766226512267
    d: float = 0.6407629491996472


class BestKappa(NamedTuple):
    rmse: float = 0.01813931095477387
    r: float = 0.015646822914572863
    nse: float = 0.018554725628140704
