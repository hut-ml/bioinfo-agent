"""Load the raw counts and sample metadata."""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("BIOINFO_DATA_DIR", PROJECT_ROOT / "candidate_data"))


@dataclass
class Dataset:
    counts: pd.DataFrame  # genes x samples
    metadata: pd.DataFrame  # indexed by sample_id, column: condition

    @property
    def n_genes(self) -> int:
        return self.counts.shape[0]

    @property
    def n_samples(self) -> int:
        return self.counts.shape[1]


def load_dataset(directory: Path | str | None = None) -> Dataset:
    directory = Path(directory) if directory else data_dir()
    counts = pd.read_csv(directory / "counts.csv", index_col="gene")
    metadata = pd.read_csv(directory / "metadata.csv", index_col="sample_id")

    missing_meta = sorted(set(counts.columns) - set(metadata.index))
    missing_counts = sorted(set(metadata.index) - set(counts.columns))
    if missing_meta or missing_counts:
        raise ValueError(
            "Sample IDs in counts.csv and metadata.csv do not match: "
            f"{len(missing_meta)} only in counts {missing_meta[:5]}, "
            f"{len(missing_counts)} only in metadata {missing_counts[:5]}"
        )

    metadata = metadata.loc[counts.columns].copy()
    return Dataset(counts=counts, metadata=metadata)


def cohort_table(metadata: pd.DataFrame) -> pd.Series:
    """Number of samples per condition."""
    return metadata["condition"].value_counts().sort_index()


def add_lane(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return metadata with a `lane` column (L1 / L2) taken from the sample ID suffix."""
    lane = metadata.index.to_series().str.extract(r"_(L\d+)$", expand=False)
    if lane.isna().any():
        bad = list(lane.index[lane.isna()][:5])
        raise ValueError(f"Sample IDs without a lane suffix such as _L1: {bad}")
    return metadata.assign(lane=lane.to_numpy())
