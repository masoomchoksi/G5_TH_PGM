# CSE 516 — DECI / Causal Structure Learning Experiments

This repository organizes the code and frozen datasets supplied for experiments E1, E2, E4–E5, and E7. E7 includes both the graph-agnostic missing-data baseline and the graph-aware imputation extension.

## Repository structure

```text
Code/
├── README.md
├── .gitignore
├── requirements/
│   ├── deci-py310.txt
│   └── analysis.txt
├── data/
│   ├── E1/       # E1 sample-size datasets
│   ├── E2/       # E2 graph-size datasets
│   ├── E45/      # CSuite datasets for E4–E5
│   └── E7/       # frozen missing-data datasets + SHA-256 manifest
├── src/
│   ├── e1/       # E1 training, baselines, metrics, aggregation
│   ├── e2/       # E2 training, scoring, aggregation
│   ├── e4_e5/    # CSuite validation, training, aggregation
│   └── e7/       # E7 baseline + graph-aware extension
├── docs/
│   └── validation/e4_e5/  # supplied validation output snapshots
└── results/     # created when experiments run; excluded from Git
```

The source folders intentionally keep experiment-specific copies of similarly named modules (for example, `metrics.py` and `train_deci.py`). E1 and E2/E7 do not use identical versions, so they should not be consolidated or overwritten.

## Setup

Run commands from the repository root (`Code/`), unless a specific experiment guide says otherwise.

### DECI environment

Causica 0.4.6 is used by the DECI training code and requires Python 3.10 for this setup.

```bash
python -m venv .venv-deci
# Windows PowerShell: .venv-deci\Scripts\Activate.ps1
# macOS / Linux:
source .venv-deci/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/deci-py310.txt
```

### Analysis / baseline environment

Use this for the E1 classical baselines and post-processing scripts that need SciPy, Matplotlib, or tabulate:

```bash
python -m venv .venv-analysis
# Windows PowerShell: .venv-analysis\Scripts\Activate.ps1
# macOS / Linux:
source .venv-analysis/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/analysis.txt
```

Keep the two environments separate when reproducing the experiments. Hardware-specific PyTorch/CUDA installation may need to follow the appropriate PyTorch instructions for your machine.

## Experiment guides

| Experiment | Description | Guide |
|---|---|---|
| E1 | Sample-size sensitivity; Table 4 | [`src/e1/README.md`](src/e1/README.md) |
| E2 | Graph-size and cyclic-posterior analysis; Table 7 | [`src/e2/README.md`](src/e2/README.md) |
| E4–E5 | CSuite replication check; Table 5 | [`src/e4_e5/README.md`](src/e4_e5/README.md) |
| E7 baseline | Missing-data DECI with graph-agnostic imputation; Table 6 / P5 | [`src/e7/README.md`](src/e7/README.md) |
| E7 graph-aware | Markov-blanket imputation extension | [`src/e7/README_graphaware.md`](src/e7/README_graphaware.md) |

Each guide contains the experiment-specific run order, commands, and reporting notes.

## Data and results

- The supplied datasets are included under `data/` so the documented commands can run without manually merging the original ZIP files.
- Keep `data/E7` unchanged. Verify its manifest before and after experiments with:

  ```bash
  python src/e7/checksum_data.py --verify data/E7
  ```

- Training outputs and generated summaries belong under `results/`. This folder is ignored by Git to prevent large checkpoints, logs, and derived files from being committed accidentally.

## GitHub upload

1. Extract or copy the `Code/` folder into your local project directory.
2. Open that folder in VS Code or a terminal.
3. Initialize Git and commit the files:

   ```bash
   git init
   git add README.md .gitignore requirements src data docs
   git commit -m "Organize CSE516 DECI experiment code"
   ```

4. Create an empty GitHub repository, add it as `origin`, and push your branch using the URL GitHub provides.

Before pushing, check `git status` and confirm that no generated `results/`, virtual environment, API key, or personal file is staged.

