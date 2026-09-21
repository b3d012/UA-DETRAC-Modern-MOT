from pathlib import Path


def test_expected_project_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
        "configs",
        "data",
        "docs",
        "experiments",
        "src/datasets",
        "src/detectors",
        "src/trackers",
        "src/evaluation",
        "src/analysis",
        "outputs/detections",
        "outputs/tracks",
        "outputs/metrics",
        "outputs/figures",
        "scripts",
        "tests",
    ]
    for relative in expected:
        assert (root / relative).exists(), relative
