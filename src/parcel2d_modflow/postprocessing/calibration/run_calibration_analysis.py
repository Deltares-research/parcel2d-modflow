"""Configure and run calibration analysis and comparison plots."""

from importlib import import_module
from pathlib import Path


def main(plotting: bool = True) -> None:
    """Run analysis, optionally producing calibration and comparison plots."""
    # Set these paths to the files and directories for this analysis.
    data_dir = Path("EDIT_ME")
    methods_dir = data_dir / "TestKaliData" / "server_evt"
    parcels = data_dir / "TestKaliData" / "calibration_parcels_form_db.geoparquet"
    observations = data_dir / "TestKaliData" / "piez_obs.nc"
    parameters = data_dir / "kali_modflow_params.csv"
    original_parameters = data_dir / "somers_2_1_parameter_selection.csv"
    dotenv_path = data_dir / ".env"
    output_dir = methods_dir / "figures"
    labels = {
        "recharge_final_conv": "Recharge method (original) \n with new parameter range",
        "boon_final_conv": 'Boon"s measurements',
    }
    for package in ("hydroeval", "matplotlib") if plotting else ("hydroeval",):
        try:
            import_module(package)
        except ModuleNotFoundError as exc:
            if exc.name != package:
                raise
            raise ModuleNotFoundError(
                f"{package} is required for calibration analysis"
                + (" when plotting=True" if package == "matplotlib" else "")
                + ". Install it before running this script."
            ) from exc

    analyze_calibration = import_module(
        "parcel2d_modflow.postprocessing.calibration.analyze_calibration"
    ).analyze_calibration
    if plotting:
        plot_calibration_comparison = import_module(
            "parcel2d_modflow.postprocessing.calibration.plots_compare_calimethods"
        ).plot_calibration_comparison

    method_dirs = {method: methods_dir / method for method in labels}
    simulation_files = {
        method: path / "kalibratie" / "sim_obs_piez_data.nc"
        for method, path in method_dirs.items()
    }
    for method_dir in method_dirs.values():
        analyze_calibration(
            method_dir, parcels, observations, parameters, dotenv_path,
            plotting=plotting,
        )

    if plotting:
        plot_calibration_comparison(
            method_dirs, simulation_files, labels, original_parameters, output_dir
        )


if __name__ == "__main__":
    main(plotting=True)
