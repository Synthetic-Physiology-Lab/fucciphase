from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fucciphase.main_cli import _import_napari_with_qt
from fucciphase.utils import simulate_single_track

EXAMPLE_DIR = REPO_ROOT / "examples" / "cli_quickstart"
SENSOR_DICT = {
    "phase_percentages": [33.3, 33.3, 33.4],
    "center": [20.0, 55.0, 70.0, 95.0],
    "sigma": [5.0, 5.0, 10.0, 1.0],
}


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = str(REPO_ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    return subprocess.run(
        [sys.executable, "-m", "fucciphase", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )


def _assert_matches_expected(actual_path: Path, expected_path: Path) -> None:
    actual_df = pd.read_csv(actual_path)
    expected_df = pd.read_csv(expected_path)

    assert list(actual_df.columns) == list(expected_df.columns)
    for column in expected_df.columns:
        if pd.api.types.is_numeric_dtype(expected_df[column]):
            np.testing.assert_allclose(
                actual_df[column].to_numpy(),
                expected_df[column].to_numpy(),
                rtol=1e-7,
                atol=1e-7,
            )
        else:
            assert actual_df[column].equals(expected_df[column]), column


def test_cli_help_succeeds_without_napari(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "-h")
    assert "TrackMate XML or tabular CSV/XLSX file" in result.stdout


def test_cli_processes_canonical_csv_example(tmp_path: Path) -> None:
    _run_cli(
        tmp_path,
        str(EXAMPLE_DIR / "tiny_tracks.csv"),
        "-ref",
        str(EXAMPLE_DIR / "tiny_reference.csv"),
        "--sensor_file",
        str(EXAMPLE_DIR / "tiny_sensor.json"),
        "-dt",
        "0.48",
        "-m",
        "MEAN_INTENSITY_CH4",
        "-c",
        "MEAN_INTENSITY_CH3",
    )

    output_path = tmp_path / "outputs" / "tiny_tracks_processed.csv"
    assert output_path.exists()
    _assert_matches_expected(
        output_path,
        EXAMPLE_DIR / "tiny_tracks_expected_output.csv",
    )


def test_cli_processes_canonical_xlsx_example(tmp_path: Path) -> None:
    _run_cli(
        tmp_path,
        str(EXAMPLE_DIR / "tiny_tracks.xlsx"),
        "-ref",
        str(EXAMPLE_DIR / "tiny_reference.csv"),
        "--sensor_file",
        str(EXAMPLE_DIR / "tiny_sensor.json"),
        "-dt",
        "0.48",
        "-m",
        "MEAN_INTENSITY_CH4",
        "-c",
        "MEAN_INTENSITY_CH3",
    )

    output_path = tmp_path / "outputs" / "tiny_tracks_processed.csv"
    assert output_path.exists()
    _assert_matches_expected(
        output_path,
        EXAMPLE_DIR / "tiny_tracks_expected_output.csv",
    )


def test_cli_processes_trackmate_xml(tmp_path: Path, trackmate_xml: Path) -> None:
    reference_path = tmp_path / "reference.csv"
    sensor_path = tmp_path / "sensor.json"

    track = simulate_single_track(track_id=7)
    reference_df = pd.DataFrame(
        {
            "percentage": track["PERCENTAGE"],
            "time": track["FRAME"] * 0.48,
            "cyan": track["MEAN_INTENSITY_CH3"],
            "magenta": track["MEAN_INTENSITY_CH4"],
        }
    )
    reference_df.to_csv(reference_path, index=False)
    sensor_path.write_text(json.dumps(SENSOR_DICT), encoding="utf-8")

    _run_cli(
        tmp_path,
        str(trackmate_xml),
        "-ref",
        str(reference_path),
        "--sensor_file",
        str(sensor_path),
        "-dt",
        "0.48",
        "-m",
        "MEAN_INTENSITY_CH2",
        "-c",
        "MEAN_INTENSITY_CH1",
        "--generate_unique_tracks",
    )

    xml_stem = Path(trackmate_xml).stem
    assert (tmp_path / "outputs" / f"{xml_stem}_processed.csv").exists()
    assert (tmp_path / "outputs" / f"{xml_stem}_processed.xml").exists()


def test_pyproject_includes_small_example_assets_for_wheel() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.hatch.build.targets.wheel.force-include]" in pyproject_text
    assert "examples/cli_quickstart" in pyproject_text
    assert "fucciphase/data/cli_quickstart" in pyproject_text
    assert 'napari  = ["napari", "pyqt6"' in pyproject_text


def test_napari_import_requires_qt_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "napari", SimpleNamespace())
    monkeypatch.setattr(
        "fucciphase.main_cli.importlib.util.find_spec",
        lambda name: None if name in {"PyQt6", "PyQt5", "PySide6"} else object(),
    )

    with pytest.raises(ImportError, match="No Qt bindings were found for napari"):
        _import_napari_with_qt()
