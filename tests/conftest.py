import numpy as np
import pandas as pd
import pytest

from bioinfo_agent.analysis import run_de, save_results
from bioinfo_agent.data import load_dataset

N_GENES = 150
UP = [f"UP{i}" for i in range(8)]  # 4x higher in treated, in both lanes
DOWN = [f"DOWN{i}" for i in range(4)]  # 4x lower in treated
LANE = [
    f"LANE{i}" for i in range(8)
]  # 6x higher in L2 only: DE without lane, not with it
GROUPS = {
    "c": ("control", "L1", 8),
    "t1": ("treated", "L1", 6),
    "t2": ("treated", "L2", 6),
}


def _write_synthetic(directory):
    """Counts with planted effects, laid out like the real data (control-L1, treated-L1, treated-L2)."""
    rng = np.random.default_rng(0)
    genes = UP + DOWN + LANE + [f"G{i}" for i in range(N_GENES - 20)]
    base = rng.uniform(100, 1000, N_GENES)
    samples, conditions, columns = [], [], []
    for prefix, (condition, lane, n) in GROUPS.items():
        for i in range(n):
            mean = base.copy()
            if condition == "treated":
                mean[:8] *= 4
                mean[8:12] /= 4
            if lane == "L2":
                mean[12:20] *= 6
            columns.append(rng.negative_binomial(10, 10 / (10 + mean)))
            samples.append(f"{prefix}{i}_{lane}")
            conditions.append(condition)
    counts = pd.DataFrame(
        np.array(columns).T, index=pd.Index(genes, name="gene"), columns=samples
    )
    counts.to_csv(directory / "counts.csv")
    pd.DataFrame({"sample_id": samples, "condition": conditions}).to_csv(
        directory / "metadata.csv", index=False
    )


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic_data")
    _write_synthetic(directory)
    return directory


@pytest.fixture(scope="session")
def synthetic_run(synthetic_data, tmp_path_factory):
    """(table, summary, results directory) of the full pipeline on the synthetic data."""
    table, summary, pca, pca_treated = run_de(load_dataset(synthetic_data))
    results = save_results(
        table, summary, pca, pca_treated, tmp_path_factory.mktemp("synthetic_results")
    )
    return table, summary, results
