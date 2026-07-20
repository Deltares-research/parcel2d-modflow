import sqlite3

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from parcel2d_modflow._io.geopackage import Geopackage


class TestGeopackage:
    @pytest.mark.unittest
    def test_get_connection(self, simple_bro_soilmap):
        gp = Geopackage(simple_bro_soilmap)
        gp.get_connection()
        assert isinstance(gp.connection, sqlite3.Connection)

    @pytest.mark.unittest
    def test_layers(self, simple_bro_soilmap):
        gp = Geopackage(simple_bro_soilmap)
        layers = gp.layers()
        assert isinstance(layers, pd.DataFrame)
        assert_array_equal(
            layers["name"],
            [
                "soilarea",
                "soilarea_soilunit",
                "soilarea_normalsoilprofile",
                "soilhorizon",
            ],
        )
        desired = pd.Series(["Polygon", pd.NA, pd.NA, pd.NA], name="geometry_type")
        assert_series_equal(layers["geometry_type"], desired)

    @pytest.mark.unittest
    def test_context_manager(self, simple_bro_soilmap):
        gp = Geopackage(simple_bro_soilmap)
        assert gp.connection is None
        with gp:
            assert isinstance(gp.connection, sqlite3.Connection)
        assert gp.connection is None

    @pytest.mark.unittest
    def test_get_cursor(self, simple_bro_soilmap):
        with Geopackage(simple_bro_soilmap) as gp:
            cursor = gp._get_cursor()
            assert isinstance(cursor, sqlite3.Cursor)

    @pytest.mark.unittest
    def test_get_column_names(self, simple_bro_soilmap):
        with Geopackage(simple_bro_soilmap) as gp:
            columns = gp.get_column_names("soilarea_soilunit")
            assert_array_equal(
                columns,
                ["fid", "maparea_id", "soilunit_code", "soilunit_sequencenumber"],
            )

    @pytest.mark.unittest
    def test_read_table(self, simple_bro_soilmap):
        test_table = "soilarea_soilunit"
        with Geopackage(simple_bro_soilmap) as gp:
            table = gp.read_table(test_table)
            assert isinstance(table, pd.DataFrame)
            assert_array_equal(
                table.columns,
                ["fid", "maparea_id", "soilunit_code", "soilunit_sequencenumber"],
            )

            table = gp.table_head(test_table)
            assert len(table) == 4

    @pytest.mark.unittest
    def test_query(self, simple_bro_soilmap):
        with Geopackage(simple_bro_soilmap) as gp:
            query = "SELECT * FROM soilarea_soilunit"
            table = gp.query(query)
        assert isinstance(table, pd.DataFrame)
        assert_array_equal(
            table.columns,
            ["fid", "maparea_id", "soilunit_code", "soilunit_sequencenumber"],
        )

        with Geopackage(simple_bro_soilmap) as gp:
            query = "SELECT * FROM soilarea_soilunit WHERE fid = 1"
            table = gp.query(query, outcolumns=["A", "B", "C"])
        assert_array_equal(table.columns, ["A", "B", "C"])
