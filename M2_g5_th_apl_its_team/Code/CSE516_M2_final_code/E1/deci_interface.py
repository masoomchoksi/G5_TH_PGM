"""
deci_interface.py - connects your Causica DECI runs to the same metrics as the baselines.

Why this file does not call Causica directly
    Causica's training API changes between versions, and it could not be installed or run
    where this file was written. So the contract is a pair of files per DECI run, which any
    Causica version can produce, and this script does everything after that.

FILE CONTRACT (one folder per dataset and DECI variant)
    results/E1/<variant>/<dataset_folder>/posterior_samples.npy
        array of shape (S, d, d), values 0/1, S = 100 graphs drawn from q_phi(G),
        with adj[j, i] == 1 meaning edge j -> i (same as the dataset adj_matrix.csv).
    results/E1/<variant>/<dataset_folder>/training_log.json   (recommended)
        {"final_elbo": float, "final_acyclicity_penalty": float,
         "outer_rounds": int, "converged": bool, "runtime_s": float}
    <variant> is "deci_s" (spline noise) or "deci_g" (Gaussian noise).

A ready-made training script for Causica 0.4.6 is train_deci.py (same folder). If you use a
    different Causica version, your own script must do the following:
    1. Load data/E1/<folder>/train.csv, z-score it with training statistics.
    2. Build DECI with spline noise (deci_s) or Gaussian noise (deci_g), using the Table 3
       settings of the report: 2x128 residual MLPs, 8-bin spline, lambda_s = 5,
       Adam lr 0.01, augmented Lagrangian (rho init 1, alpha init 0), Gumbel-softmax 0.25.
       Log every deviation in the experiment config.
    3. Train, then draw S = 100 adjacency matrices from the learned graph posterior.
    4. Run check_orientation_convention() on one dataset BEFORE the full sweep (below).
    5. Save posterior_samples.npy and training_log.json as described above.

Usage
    python deci_interface.py --variant deci_s --data data/E1 --results results/E1
    python deci_interface.py --variant deci_g --data data/E1 --results results/E1
Datasets without a saved run are written as status="missing" (reported, never dropped).
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from metrics import expected_metrics, shd


def check_orientation_convention(samples: np.ndarray, true: np.ndarray) -> str:
    """Sanity check for the transpose convention. Compares the expected SHD of the samples
    with that of their transpose. If the transposed samples are much closer to the truth,
    your Causica adjacency uses the opposite convention: save samples.transpose(0, 2, 1)."""
    s = np.mean([shd(G, true) for G in samples])
    s_t = np.mean([shd(G.T, true) for G in samples])
    if s_t < 0.7 * s:
        return f"WARNING: transposed SHD {s_t:.2f} << direct SHD {s:.2f}. Convention is probably flipped."
    return f"OK: direct SHD {s:.2f} vs transposed SHD {s_t:.2f}."


def evaluate_variant(variant: str, data_dir: str, results_dir: str) -> pd.DataFrame:
    rows = []
    for folder in sorted(glob.glob(os.path.join(data_dir, "*"))):
        name = os.path.basename(folder)
        cfg = json.load(open(os.path.join(folder, "config.json")))
        true = np.loadtxt(os.path.join(folder, "adj_matrix.csv"), delimiter=",", ndmin=2).astype(int)
        row = dict(method=variant, experiment=cfg["experiment"], d=cfg["d"], n_edges=cfg["n_edges"],
                   n_train=cfg["n_train"], seed=cfg["seed"], folder=name)
        run_dir = os.path.join(results_dir, variant, name)
        sample_path = os.path.join(run_dir, "posterior_samples.npy")
        if not os.path.exists(sample_path):
            row.update(status="missing", error="no posterior_samples.npy")
            rows.append(row)
            continue
        try:
            samples = np.load(sample_path)
            if samples.ndim != 3 or samples.shape[1:] != (cfg["d"], cfg["d"]):
                raise ValueError(f"bad shape {samples.shape}, expected (S, {cfg['d']}, {cfg['d']})")
            row.update(expected_metrics((samples > 0.5).astype(int), true))
            row.update(status="ok", error="")
            log_path = os.path.join(run_dir, "training_log.json")
            if os.path.exists(log_path):
                log = json.load(open(log_path))
                for k in ("final_elbo", "final_acyclicity_penalty", "outer_rounds", "converged", "runtime_s"):
                    row[k] = log.get(k)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["deci_s", "deci_g"])
    parser.add_argument("--data", default="data/E1")
    parser.add_argument("--results", default="results/E1")
    args = parser.parse_args()
    df = evaluate_variant(args.variant, args.data, args.results)
    os.makedirs(args.results, exist_ok=True)
    out = os.path.join(args.results, f"{args.variant}.csv")
    df.to_csv(out, index=False)
    print(df["status"].value_counts().to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
