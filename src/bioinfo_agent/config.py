"""Thresholds, file names and paths shared by the analysis and everything that reads its results.

Kept free of heavy imports (pydeseq2, scipy) so that reading results stays fast.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PADJ_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 1.0
MIN_TOTAL_COUNT = 10

DESIGN = "~ condition"
DESIGN_LANE_ADJUSTED = "~ lane + condition"

N_PCS = 5
N_VARIABLE_GENES = 2000

RESULTS_FILE = "de_results.csv"
SUMMARY_FILE = "summary.json"
PCA_FILE = "pca.csv"
PCA_TREATED_FILE = "pca_treated.csv"


def data_dir() -> Path:
    return Path(os.environ.get("BIOINFO_DATA_DIR", PROJECT_ROOT / "candidate_data"))


def results_dir() -> Path:
    return Path(os.environ.get("BIOINFO_RESULTS_DIR", PROJECT_ROOT / "results"))
