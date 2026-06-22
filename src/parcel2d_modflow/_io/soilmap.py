from enum import StrEnum
from pathlib import Path

import geopandas as gpd
import pandas as pd

from parcel2d_modflow._io.geopackage import Geopackage


class SoilmapLayers(StrEnum):
    SOILAREA = "soilarea"
    AREAOFPEDOLOGICALINTEREST = "areaofpedologicalinterest"
    NGA_PROPERTIES = "nga_properties"
    SOILMAP = "soilmap"
    NORMALSOILPROFILES = "normalsoilprofiles"
    NORMALSOILPROFILES_LANDUSE = "normalsoilprofiles_landuse"
    SOILHORIZON = "soilhorizon"
    SOILHORIZON_FRACTIONPARTICLESIZE = "soilhorizon_fractionparticlesize"
    SOILLAYER = "soillayer"
    SOIL_UNITS = "soil_units"
    SOILCHARACTERISTICS_BOTTOMLAYER = "soilcharacteristics_bottomlayer"
    SOILCHARACTERISTICS_TOPLAYER = "soilcharacteristics_toplayer"
    SOILAREA_NORMALSOILPROFILE = "soilarea_normalsoilprofile"
    SOILAREA_SOILUNIT = "soilarea_soilunit"
    SOILAREA_SOILUNIT_SOILCHARACTERISTICSTOPLAYER = (
        "soilarea_soilunit_soilcharacteristicstoplayer"  # noqa: E501
    )
    SOILAREA_SOILUNIT_SOILCHARACTERISTICSBOTTOMLAYER = (
        "soilarea_soilunit_soilcharacteristicsbottomlayer"  # noqa: E501
    )


class BroSoilmap:
    """
    Class to handle the Bro Soilmap file for data selections and facilitate making
    combinations between the different data tables in the BRO CPT geopackage.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the spatial locations of the CPT data. The "find key" per
        "bro_id" to each related tables in the Geopackage as index (index name: "fid").
    db : :class:`~geost.io.Geopackage`
        Geost Geopackage instance to handle the database connections and queries.
    """

    def __init__(self, gdf: gpd.GeoDataFrame, db: Geopackage):
        self.gdf = gdf
        self.db = db

    @classmethod
    def from_geopackage(cls, file: str | Path, **gpd_kwargs):
        """
        Create a BroSoilmap instance from a Geopackage file.

        Parameters
        ----------
        file : str | Path
            Path to the Geopackage file.
        gpd_kwargs
            Keyword arguments to pass to the GeoDataFrame constructor.

        Returns
        -------
        BroSoilmap
            Instance of the BroSoilmap class.

        """
        if "fid_as_index" not in gpd_kwargs:  # Needs to retain index for db selections
            gpd_kwargs["fid_as_index"] = True

        if "layer" in gpd_kwargs:
            raise ValueError("Layer cannot be passed as a Geopandas keyword argument.")

        if "columns" in gpd_kwargs:
            raise ValueError(
                "Columns cannot be passed as a Geopandas keyword argument."
            )

        columns_gdf = ["maparea_id", "geometry"]
        gdf = gpd.read_file(
            file, layer=SoilmapLayers.SOILAREA, columns=columns_gdf, **gpd_kwargs
        )
        db = Geopackage(file)

        return cls(gdf, db)

    def create_soilmap_with_units(self) -> gpd.GeoDataFrame:
        """
        Combine soil unit codes with the GeoDataFrame containing the soilmap polygons.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the soilmap and soil units.

        """
        sn = SoilmapLayers.SOILAREA_NORMALSOILPROFILE
        su = SoilmapLayers.SOILAREA_SOILUNIT

        query = f"""
            SELECT
                SN.maparea_id, SN.normalsoilprofile_id, SU.soilunit_code
            FROM {sn} SN
            JOIN {su} SU ON SN.maparea_id = SU.maparea_id
        """

        with self.db:
            units = self.db.query(query)

        soilmap = self.gdf.merge(units, on="maparea_id", how="left")

        return soilmap

    def create_soilprofile_table(self) -> pd.DataFrame:
        """
        Select the soil profile information from the BRO soilmap geopackage.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the soil profile information.

        """
        sh = SoilmapLayers.SOILHORIZON

        query = f"""
            SELECT
                SH.normalsoilprofile_id,
                SH.lowervalue,
                SH.uppervalue,
                SH.organicmattercontent,
                SH.peattype,
                SH.loamcontent,
                SH.lutitecontent,
                SH.siltcontent,
                SH.cnratio
            FROM {sh} SH
        """

        with self.db:
            table = self.db.query(query)

        return table
