import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


RUNNER = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "parcel2d_modflow"
    / "postprocessing"
    / "calibration"
    / "run_calibration_analysis.py"
)


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("calibration_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_runner_does_not_check_dependencies(monkeypatch):
    def unexpected_import(name):
        raise AssertionError(f"Unexpected dependency check: {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    spec = importlib.util.spec_from_file_location("calibration_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@pytest.mark.parametrize("plotting", [True, False])
def test_runner_checks_and_imports_only_needed_dependencies(runner, monkeypatch, plotting):
    analyze = Mock()
    plot = Mock()
    imported = []

    def fake_import(name):
        imported.append(name)
        if name.endswith(".analyze_calibration"):
            return SimpleNamespace(analyze_calibration=analyze)
        if name.endswith(".plots_compare_calimethods"):
            return SimpleNamespace(plot_calibration_comparison=plot)
        return SimpleNamespace()

    monkeypatch.setattr(runner, "import_module", fake_import)
    runner.main(plotting=plotting)

    assert imported[0] == "hydroeval"
    assert ("matplotlib" in imported) == plotting
    assert any(name.endswith(".plots_compare_calimethods") for name in imported) == plotting
    assert analyze.call_count == 2
    assert all(call.kwargs["plotting"] == plotting for call in analyze.call_args_list)
    assert plot.called == plotting


@pytest.mark.parametrize("package", ["hydroeval", "matplotlib"])
def test_runner_reports_missing_dependency_before_analysis(runner, monkeypatch, package):
    def fake_import(name):
        if name == package:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        assert name == "hydroeval"
        return SimpleNamespace()

    monkeypatch.setattr(runner, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError, match=package):
        runner.main()


def test_runner_does_not_mask_missing_transitive_dependency(runner, monkeypatch):
    def fake_import(name):
        raise ModuleNotFoundError("missing nested dependency", name="nested_dependency")

    monkeypatch.setattr(runner, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError, match="missing nested dependency"):
        runner.main()
