# smart_parking/datasets/utils.py
"""Common helpers for resolving dataset paths and aliases."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ALIASES = {
    "pklot": PROJECT_ROOT / "datasets" / "ammarnassanalhajali",
    "cnr": PROJECT_ROOT / "datasets" / "CNR",
}


def resolve_dataset_path(dataset: str | None) -> Path | None:
    """Resolve dataset alias or path to an absolute Path."""
    if not dataset:
        return None

    alias = dataset.lower()
    if alias in DATASET_ALIASES:
        candidate = DATASET_ALIASES[alias]
        if candidate.is_dir():
            return candidate
        msg = f"Dataset alias '{dataset}' maps to '{candidate}' but it does not exist."
        raise FileNotFoundError(msg)

    path = Path(dataset)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if path.is_dir():
        return path

    raise FileNotFoundError(f"Dataset path '{path}' does not exist.")

