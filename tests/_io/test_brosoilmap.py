import sqlite3

import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from parcel2d_modflow._io.soilmap import BroSoilmap


class TestBroSoilmap:
    @pytest.mark.unittest
    def test_create_soilmap_with_units(self, simple_bro_soilmap):
        expected_profiles = [1050, 1010, 1010, 1130]
        expected_soilcodes = ["hVc", "hVb", "hVb", "pVb"]
        expected_area_ids = [
            "V2023-1.soilarea.0000022581",
            "V2023-1.soilarea.0000022124",
            "V2023-1.soilarea.0000022112",
            "V2023-1.soilarea.0000034574",
        ]

        bro_soilmap = BroSoilmap.from_geopackage(simple_bro_soilmap)
        soilmap = bro_soilmap.create_soilmap_with_units()

        assert isinstance(soilmap, gpd.GeoDataFrame)
        assert soilmap.shape == (4, 4)
        assert_array_equal(soilmap["normalsoilprofile_id"], expected_profiles)
        assert_array_equal(soilmap["soilunit_code"], expected_soilcodes)
        assert_array_equal(soilmap["maparea_id"], expected_area_ids)

    @pytest.mark.unittest
    def test_create_soilprofile_table(self, simple_bro_soilmap):
        expected_columns = [
            "normalsoilprofile_id",
            "lowervalue",
            "uppervalue",
            "organicmattercontent",
            "peattype",
            "loamcontent",
            "lutitecontent",
            "siltcontent",
            "cnratio",
        ]

        bro_soilmap = BroSoilmap.from_geopackage(simple_bro_soilmap)
        table = bro_soilmap.create_soilprofile_table()

        assert isinstance(table, pd.DataFrame)
        assert table.shape == (13, 9)
        assert_array_equal(table.columns, expected_columns)
