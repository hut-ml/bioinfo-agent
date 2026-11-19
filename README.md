# bioinfo-agent

A small agent that answers questions about a bulk RNA-seq differential expression (DE) analysis
(control vs treated). The analysis is precomputed, so every number the agent reports comes from the results.

## Install

Requires [uv](https://docs.astral.sh/uv/), which also installs Python 3.12 if needed.

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
```

## Run

```bash
uv run bioinfo-agent info     # cohort summary: genes, samples, samples per condition
uv run bioinfo-agent genes    # analysis overview and top DE genes (--direction up|down, -n 20, --sort-by log2fc)
uv run bioinfo-agent gene STAT1 CIITA   # look up genes (case-insensitive, suggests close names)
uv run bioinfo-agent precompute         # re-run the DE analysis (about 20 s), rewrites results/
uv run pytest                 # tests
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs lint, tests and the `info` and `gene` commands on macOS and Windows.

The data directory defaults to `candidate_data/`; set `BIOINFO_DATA_DIR` to use another one.
Results are read from `results/`; set `BIOINFO_RESULTS_DIR` to use another directory.

## Data notes

- `counts.csv`: 15,831 genes (rows) x 90 samples (columns), raw integer counts.
- `metadata.csv`: `sample_id`, `condition` (control / treated).
- Groups: 40 control, 50 treated.
- The sample ID suffix is the sequencing lane. Every `_L2` sample is treated, so the groups are
  control-L1 (40), treated-L1 (25) and treated-L2 (25): lane is partly confounded with condition.

## Analysis

The analysis is precomputed with [PyDESeq2](https://pydeseq2.readthedocs.io/) and committed in
`results/de_results.csv` (one row per gene) and `results/summary.json` (design, thresholds, counts, size
factors). Questions are answered from these files only.

- **Filter:** genes with fewer than 10 reads in total are dropped (none were: all 15,831 genes are tested).
- **Model:** `~ condition`, treated vs control. A positive log2FC means higher in treated.
- **Fold changes:** shrunken (`lfc_shrink`), which stabilises low-count genes. The unshrunken value is
  kept as `log2fc_unshrunk`.
- **Significant:** adjusted p-value (Benjamini-Hochberg) `padj < 0.05` and `|log2FC| >= 1`.
  Result: 222 up, 100 down.
- Where `padj` underflows to 0, genes are ranked by `|stat|`.

### Example genes

| Gene | Verdict |
| --- | --- |
| CIITA | Up in treated: log2FC +1.34, padj 1.6e-76 |
| SPATS2L | Up in treated: log2FC +1.14, padj 2.2e-14 |
| IDH3B | Not DE: log2FC -0.06, padj 1 |
| STAT1 | Not DE: log2FC -0.03, padj 1 |

### Observations and future directions

- **Lane:** the sample ID suffix is the sequencing lane, and every `_L2` sample is treated (control-L1 40,
  treated-L1 25, treated-L2 25), so lane is partly confounded with condition. It is not in the model.
  Next step: compare against `~ lane + condition` (treatment effect estimated from L1 only) and flag genes
  whose call changes. A quick check suggests SPATS2L and IDH3B are among them.
- Sex-marker genes are not in the data, so sex balance across groups cannot be checked.
