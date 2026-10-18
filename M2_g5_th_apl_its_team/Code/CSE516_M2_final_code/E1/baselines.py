"""
baselines.py - run PC, ICALiNGAM and NOTEARS-MLP (gcastle) on the E1 datasets.

Usage (from the folder that contains data/ and this file):
    python baselines.py --methods pc icalingam notears_mlp --data data/E1 --out results/E1
    python baselines.py --methods pc --data data/E1 --out results/E1 --limit 2     # quick test

Each method writes results/E1/<method>.csv with one row per dataset
(method, n_train, seed, expected SHD, F1 scores, runtime, status).
A failed run is recorded with status="failed" and the error, never dropped silently.

Protocol choices (log them in the report):
  * Data are z-scored with training statistics before every method (same for all).
  * One fixed edge threshold per method (below), never tuned on the true graph.
  * PC returns a CPDAG: every DAG in its Markov equivalence class gets equal probability
    (Section 9.3). Exact enumeration if the class is small, else uniform sampling.
  * PC uses gcastle's default Fisher-z test, which is a LINEAR independence test applied
    to non-linear data. State this as a limitation.
"""

import os
# Colab sets MPLBACKEND to a notebook-only backend that does not exist in a separate venv; force a file backend.
os.environ["MPLBACKEND"] = "Agg"
try:
    import torch  # noqa: F401  keep first on Windows: numpy/pandas imported before torch can cause WinError 1114
except Exception:
    pass
import argparse
import glob
import json
import os
import time
import traceback

import numpy as np
import pandas as pd

from metrics import cpdag_to_dags, expected_metrics

ICALINGAM_THRESHOLD = 0.3        # gcastle default weight threshold
NOTEARS_THRESHOLD = 0.3          # gcastle default weight threshold
PC_ALPHA = 0.05
PC_MAX_CLASS_SAMPLES = 100       # S = 100 as in Section 9.3


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def load_folder(folder: str):
    cfg = json.load(open(os.path.join(folder, "config.json")))
    X = pd.read_csv(os.path.join(folder, "train.csv")).values.astype(float)
    A = np.loadtxt(os.path.join(folder, "adj_matrix.csv"), delimiter=",", ndmin=2).astype(int)
    return cfg, X, A


def zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


# ----------------------------------------------------------------------------
# Methods: each returns an array (S, d, d) of graph samples (S = 1 for point estimates)
# ----------------------------------------------------------------------------
def run_pc(X: np.ndarray, seed: int) -> np.ndarray:
    from castle.algorithms import PC
    model = PC(alpha=PC_ALPHA)
    model.learn(X)
    cpdag = (np.array(model.causal_matrix) != 0).astype(int)
    return cpdag_to_dags(cpdag, max_samples=PC_MAX_CLASS_SAMPLES, rng=np.random.default_rng(seed))


def run_icalingam(X: np.ndarray, seed: int) -> np.ndarray:
    from castle.algorithms import ICALiNGAM
    model = ICALiNGAM(thresh=ICALINGAM_THRESHOLD, random_state=seed)
    model.learn(X)
    return (np.array(model.causal_matrix) != 0).astype(int)[None]


def run_notears_mlp(X: np.ndarray, seed: int) -> np.ndarray:
    import torch
    from castle.algorithms import NotearsNonlinear
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NotearsNonlinear(w_threshold=NOTEARS_THRESHOLD, device_type="cpu")
    model.learn(X)
    return (np.array(model.causal_matrix) != 0).astype(int)[None]


METHODS = {"pc": run_pc, "icalingam": run_icalingam, "notears_mlp": run_notears_mlp}


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def run_method_on_folder(method: str, folder: str) -> dict:
    cfg, X, A = load_folder(folder)
    row = dict(method=method, experiment=cfg["experiment"], d=cfg["d"], n_edges=cfg["n_edges"],
               n_train=cfg["n_train"], seed=cfg["seed"], folder=os.path.basename(folder))
    start = time.time()
    try:
        samples = METHODS[method](zscore(X), cfg["seed"])
        row.update(expected_metrics(samples, A))
        row.update(status="ok", error="")
    except Exception as exc:                          # record, do not drop
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    row["runtime_s"] = round(time.time() - start, 2)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--data", default="data/E1")
    parser.add_argument("--out", default="results/E1")
    parser.add_argument("--limit", type=int, default=None, help="only the first N folders (testing)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    folders = sorted(glob.glob(os.path.join(args.data, "*")))
    if args.limit:
        folders = folders[: args.limit]

    for method in args.methods:
        path = os.path.join(args.out, f"{method}.csv")
        done = set()
        rows = []
        if os.path.exists(path) and not args.overwrite:
            old = pd.read_csv(path)
            rows = old.to_dict("records")
            done = set(old.loc[old["status"] == "ok", "folder"])
            rows = [r for r in rows if r["folder"] in done]          # retry failed ones
        for idx, folder in enumerate(folders, 1):
            name = os.path.basename(folder)
            if name in done:
                print(f"[{method}] ({idx}/{len(folders)}) {name}: already done, skipping", flush=True)
                continue
            print(f"[{method}] ({idx}/{len(folders)}) {name}: started ...", flush=True)   # shows the run is alive
            row = run_method_on_folder(method, folder)
            rows.append(row)
            print(f"[{method}] ({idx}/{len(folders)}) {name}: {row['status']}  "
                  f"SHD={row.get('expected_shd', float('nan')):.2f}  ({row['runtime_s']}s)", flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)               # save after every run
        print(f"{method}: {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
