import importlib
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

pytest.importorskip("hydroeval")

analysis = importlib.import_module(
    "parcel2d_modflow.postprocessing.calibration.analyze_calibration"
)


@pytest.fixture
def calibration_inputs(tmp_path, monkeypatch):
    method_dir = tmp_path / "method"
    parcel_dir = method_dir / "A_peat"
    parcel_dir.mkdir(parents=True)
    (parcel_dir / "phreatic_head_A_peat.nc").touch()
    time = pd.date_range("2022-01-01", periods=5)
    observations = xr.DataArray(
        [[1.0, 2.0, 3.0, 4.0, 5.0]],
        dims=("well_id", "time"),
        coords={"well_id": ["well1"], "time": time},
    )
    simulations = xr.DataArray(
        [[[1.0], [2.0], [3.0], [4.0], [5.0]],
         [[1.2], [2.2], [3.2], [4.2], [5.2]]],
        dims=("runs", "time", "x"),
        coords={"runs": [1, 2], "time": time, "x": [0.5]},
    )
    engine = Mock()
    monkeypatch.setattr(analysis, "initialize_database", lambda path: engine)
    metadata = pd.DataFrame({"well_id": ["well1"], "distance_to_ditch_m": [0.5]})
    monkeypatch.setattr(
        analysis, "read_table_from_database", lambda **kwargs: metadata
    )
    monkeypatch.setattr(
        analysis.gpd, "read_parquet",
        lambda path: pd.DataFrame(
            {"name": ["A"], "soilcode": ["peat"], "well_id": [["well1"]]}
        ),
    )
    monkeypatch.setattr(
        analysis, "read_modflow_parameters",
        lambda path: pd.DataFrame({"runnr": [1, 2], "kh": [1.0, 2.0]}),
    )
    monkeypatch.setattr(
        analysis.xr, "open_dataarray",
        lambda path: observations if path.name == "observations.nc" else simulations,
    )
    plot = Mock()
    monkeypatch.setattr(analysis, "plot_piezometer_simulation_envelope", plot)

    return (
        method_dir, observations, simulations, metadata, engine, plot,
        (
            method_dir, tmp_path / "parcels.parquet", tmp_path / "observations.nc",
            tmp_path / "parameters.csv", tmp_path / ".env",
        ),
    )


@pytest.mark.unittest
@pytest.mark.parametrize("plotting", [True, False])
def test_analyze_calibration_writes_results(calibration_inputs, plotting):
    method_dir, _, _, _, engine, plot, paths = calibration_inputs

    analysis.analyze_calibration(*paths, plotting=plotting)

    output = method_dir / "kalibratie"
    selected = pd.read_csv(output / "best_kali_parameters.csv")
    assert selected["runnr"].tolist() == [1, 2]
    assert (output / "all_piez_performance.csv").exists()
    assert (output / "global_weighted_performance.csv").exists()
    engine.dispose.assert_called_once()
    assert plot.called == plotting
    assert (output / "plots").exists() == plotting


@pytest.mark.unittest
def test_analyze_calibration_excludes_missing_timesteps(calibration_inputs):
    method_dir, observations, simulations, _, _, _, paths = calibration_inputs
    observations.values[0, 0] = np.nan
    simulations.values[1, 1, 0] = np.nan

    analysis.analyze_calibration(*paths, plotting=False)

    result = pd.read_csv(
        method_dir / "kalibratie" / "all_piez_performance.csv"
    ).set_index("runs")
    assert_array_equal(result.index, [1, 2])
    assert result.loc[1, "kge"] == pytest.approx(1.0)
    assert result.loc[1, "rmse"] == pytest.approx(0.0)
    assert result.loc[2, "rmse"] == pytest.approx(0.2)


@pytest.mark.unittest
@pytest.mark.parametrize("missing", ["simulation", "metadata", "observations"])
def test_analyze_calibration_no_usable_data(calibration_inputs, missing):
    method_dir, observations, _, metadata, engine, _, paths = calibration_inputs
    if missing == "simulation":
        (method_dir / "A_peat" / "phreatic_head_A_peat.nc").unlink()
    elif missing == "metadata":
        metadata.drop(metadata.index, inplace=True)
    else:
        observations.values[:] = np.nan

    with pytest.raises(ValueError, match="No well performance data found"):
        analysis.analyze_calibration(*paths, plotting=False)

    engine.dispose.assert_called_once()
    assert not (method_dir / "kalibratie" / "best_kali_parameters.csv").exists()


@pytest.mark.unittest
def test_analyze_calibration_disposes_connection_on_query_error(
    calibration_inputs, monkeypatch
):
    _, _, _, _, engine, _, paths = calibration_inputs

    def fail_query(**kwargs):
        raise RuntimeError("database query failed")

    monkeypatch.setattr(analysis, "read_table_from_database", fail_query)
    with pytest.raises(RuntimeError, match="database query failed"):
        analysis.analyze_calibration(*paths, plotting=False)

    engine.dispose.assert_called_once()


@pytest.mark.unittest
def test_analyze_calibration_combines_multiple_wells_per_parcel(
    calibration_inputs, monkeypatch
):
    """
    Verify, without depending on hydroeval's KGE/RMSE formulas, that
    `analyze_calibration` selects each well's own x position and valid
    timesteps before scoring it, and weights multiple wells of the same
    parcel using `calculate_weighted_performance`.

    """
    method_dir, _, _, _, _, _, paths = calibration_inputs

    time = pd.date_range("2022-01-01", periods=5)
    x_well1, x_well2 = 0.5, 1.5

    # well2 has two missing observations, so only 3 of 5 timesteps are valid,
    # unlike well1's 5 of 5.
    obs_well1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    obs_well2 = np.array([10.0, 20.0, np.nan, np.nan, 50.0])
    observations = xr.DataArray(
        np.stack([obs_well1, obs_well2]),
        dims=("well_id", "time"),
        coords={"well_id": ["well1", "well2"], "time": time},
    )

    sim_well1 = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 2.0, 3.0, 4.0, 7.0]]
    )  # (runs, time)
    sim_well2 = np.array(
        [[10.0, 20.0, 99.0, 99.0, 45.0], [15.0, 25.0, 99.0, 99.0, 55.0]]
    )
    simulations = xr.DataArray(
        np.stack([sim_well1, sim_well2], axis=-1),
        dims=("runs", "time", "x"),
        coords={"runs": [1, 2], "time": time, "x": [x_well1, x_well2]},
    )

    monkeypatch.setattr(
        analysis.xr, "open_dataarray",
        lambda path: observations if path.name == "observations.nc" else simulations,
    )
    monkeypatch.setattr(
        analysis, "read_table_from_database",
        lambda **kwargs: pd.DataFrame(
            {"well_id": ["well1", "well2"], "distance_to_ditch_m": [x_well1, x_well2]}
        ),
    )
    monkeypatch.setattr(
        analysis.gpd, "read_parquet",
        lambda path: pd.DataFrame(
            {"name": ["A"], "soilcode": ["peat"], "well_id": [["well1", "well2"]]}
        ),
    )

    # Replace `evaluate_simulation` with a deterministic stub so this test
    # only exercises `analyze_calibration`'s own well-selection, timestep
    # masking, and weighting code, never hydroeval's KGE/RMSE formulas.
    calls = []
    fake_performance = {
        "well1": {"r": 0.9, "alpha": 1.0, "beta": 1.0, "rmse": 0.1},
        "well2": {"r": 0.5, "alpha": 1.4, "beta": 0.8, "rmse": 0.5},
    }

    def fake_evaluate_simulation(simulation, evaluation):
        piez_name = "well1" if not calls else "well2"
        calls.append(
            {
                "piez_name": piez_name,
                "sim_values": simulation.values.copy(),
                "obs_values": evaluation.values.copy(),
            }
        )
        values = fake_performance[piez_name]
        n_runs = simulation.sizes["runs"]
        n_time = simulation.sizes["time"]
        return pd.DataFrame(
            {
                "r": [values["r"]] * n_runs,
                "alpha": [values["alpha"]] * n_runs,
                "beta": [values["beta"]] * n_runs,
                "rmse": [values["rmse"]] * n_runs,
                "count": [n_time] * n_runs,
            },
            index=simulation.runs,
        )

    monkeypatch.setattr(analysis, "evaluate_simulation", fake_evaluate_simulation)

    analysis.analyze_calibration(*paths, plotting=False)

    # Confirm `analyze_calibration` selected each well's own x position and
    # dropped only that well's invalid (NaN) timesteps before scoring it.
    assert len(calls) == 2
    assert_array_almost_equal(calls[0]["obs_values"], obs_well1)
    assert_array_almost_equal(calls[0]["sim_values"], sim_well1.T)
    assert_array_almost_equal(calls[1]["obs_values"], [10.0, 20.0, 50.0])
    assert_array_almost_equal(calls[1]["sim_values"], sim_well2[:, [0, 1, 4]].T)

    # Confirm the per-well results were combined with the correct weights
    # (5 valid timesteps for well1, 3 for well2) using the project's own
    # `calculate_weighted_performance` formula.
    r_w = (0.9 * 5 + 0.5 * 3) / 8
    alpha_w = (1.0 * 5 + 1.4 * 3) / 8
    beta_w = (1.0 * 5 + 0.8 * 3) / 8
    rmse_w = (0.1 * 5 + 0.5 * 3) / 8
    kge_w = 1 - np.sqrt((alpha_w - 1) ** 2 + (beta_w - 1) ** 2 + (r_w - 1) ** 2)

    result = pd.read_csv(
        method_dir / "kalibratie" / "transect_performance" / "A_peat_weighted_performance.csv",
        index_col="runs",
    )
    assert_array_almost_equal(result["r"], [r_w, r_w])
    assert_array_almost_equal(result["alpha"], [alpha_w, alpha_w])
    assert_array_almost_equal(result["beta"], [beta_w, beta_w])
    assert_array_almost_equal(result["rmse"], [rmse_w, rmse_w])
    assert_array_almost_equal(result["kge"], [kge_w, kge_w])
    assert_array_almost_equal(result["count"], [4.0, 4.0])
