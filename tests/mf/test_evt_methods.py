import pytest
from numpy.testing import assert_array_almost_equal

from parcel2d_modflow.mf.evt_profiles import EVTProfile, calc_evt_profile


@pytest.mark.parametrize(
    "method, expected_result",
    [
        (
            "woerkom",
            EVTProfile(
                evt_ext_depth=3.1,
                rel_segment_bottom=[0.0, 0.13, 0.49, 0.85, 1.0],
                evt_fraction_values=[1.0, 1.0, 0.9, 0.7, 0.1],
            ),
        ),
        (
            "combi",
            EVTProfile(
                evt_ext_depth=3.1,
                rel_segment_bottom=[0.03, 0.06, 0.13, 0.23, 0.32, 0.81, 1.0],
                evt_fraction_values=[1.8, 1.43, 1.18, 0.9, 0.8, 0.7, 0.1],
            ),
        ),
        (
            "boon",
            EVTProfile(evt_ext_depth=3.0, 
                       rel_segment_bottom=[1.00000e-06, 3.00000e-02, 7.00000e-02, 1.30000e-01, 2.70000e-01, 4.00000e-01, 9.99999e-01], 
                       evt_fraction_values=[1.26962344, 1.0524753 , 0.93590154, 0.85699812, 0.83648565, 0.83554519, 0.8355]
                       ),
        ),
    ],
    ids=["woerkom", "combi", "boon"],
)
def test_calc_evt_profile(method, expected_result):
    result = calc_evt_profile(method)
    assert isinstance(result, EVTProfile)
    assert_array_almost_equal(result.evt_ext_depth, expected_result.evt_ext_depth)
    assert_array_almost_equal(
        result.rel_segment_bottom, expected_result.rel_segment_bottom
    )
    assert_array_almost_equal(
        result.evt_fraction_values, expected_result.evt_fraction_values
    )
