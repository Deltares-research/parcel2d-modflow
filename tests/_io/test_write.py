import pytest

from parcel2d_modflow.config import OutputSettings
from parcel2d_modflow.io import write


@pytest.mark.unittest
def test_write_batch(parcels, tmp_path):
    output_settings = OutputSettings(
        directory=tmp_path,
        prefix="batch",
        format="parquet",
    )
    batch_number = 23
    write.write_batch(parcels, batch_number, output_settings)
    expected_file = output_settings.directory / r"batch_00023.parquet"
    assert expected_file.exists()
