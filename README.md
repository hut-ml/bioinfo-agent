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
uv run pytest                 # tests
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs lint, tests and the `info` command on macOS and Windows.

The data directory defaults to `candidate_data/`; set `BIOINFO_DATA_DIR` to use another one.

## Data notes

- `counts.csv`: 15,831 genes (rows) x 90 samples (columns), raw integer counts.
- `metadata.csv`: `sample_id`, `condition` (control / treated).
- Groups: 40 control, 50 treated.

