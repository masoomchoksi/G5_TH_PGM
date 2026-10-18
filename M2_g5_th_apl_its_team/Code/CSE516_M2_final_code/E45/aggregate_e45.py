"""
aggregate_e45.py - build Table 5 (E4-E5, CSuite replication check) from the saved DECI runs.

Reads   results/E45/<variant>/<dataset>_seed<k>/{posterior_samples.npy, ate.json, training_log.json}
        data/E45/<dataset>/adj_matrix.csv                       (the true graph)
Writes  <out>/per_run.csv          one row per (variant, dataset, seed), every metric, status
        <out>/table5.csv / .md     Table 5 in the report's layout (median over the training seeds)
        <out>/table5_mean_sd.csv   mean, sd and n of every column, for the report's appendix

Columns of Table 5
    Adj. F1, Orient. F1   expected over the 100 posterior graph samples (metrics.expected_metrics)
    Posterior mass on G0  fraction of the 100 samples that equal the true graph exactly
    ATE RMSE (ours)       Causica's own evaluator, STANDARDISED units (see train_deci_csuite.py); raw units are in
                          per_run.csv (ate_rmse_raw) and the mean/sd file
    ATE RMSE (published)  the medians quoted in the report's Table 5 (PUBLISHED below)
Each cell is the MEDIAN over the training seeds, to match the "published median" column and to stay robust to one
bad seed. A (dataset, method) with no finished run is printed as "Not run"; failed runs are listed, never dropped.

Usage:  python aggregate_e45.py --data data/E45 --results results/E45 --out results/E45/summary
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from metrics import expected_metrics

ORDER = ["lingauss", "linexp", "nonlingauss", "nonlin_simpson"]
VARIANTS = [("deci_g", "DECI-G"), ("deci_s", "DECI-S")]
# published median ATE RMSE, as quoted in the report's Table 5
PUBLISHED = {("lingauss", "deci_g"): 0.120, ("lingauss", "deci_s"): 0.149,
             ("linexp", "deci_g"): 1.029, ("linexp", "deci_s"): 0.022,
             ("nonlingauss", "deci_g"): 0.042, ("nonlingauss", "deci_s"): 0.043,
             ("nonlin_simpson", "deci_g"): 1.995, ("nonlin_simpson", "deci_s"): 1.997}


def collect(data_dir: str, results_dir: str) -> pd.DataFrame:
    rows = []
    for variant, _ in VARIANTS:
        for ds in ORDER:
            true = np.loadtxt(os.path.join(data_dir, ds, "adj_matrix.csv"), delimiter=",", ndmin=2).astype(int)
            runs = sorted(glob.glob(os.path.join(results_dir, variant, f"{ds}_seed*")))
            for run in runs:
                row = dict(variant=variant, dataset=ds, seed=int(run.rsplit("seed", 1)[1]), run_dir=run)
                try:
                    if os.path.exists(os.path.join(run, "error.txt")) and not os.path.exists(os.path.join(run, "ate.json")):
                        raise RuntimeError(open(os.path.join(run, "error.txt")).read().strip())
                    samples = np.load(os.path.join(run, "posterior_samples.npy"))
                    if samples.ndim != 3 or samples.shape[1:] != true.shape:
                        raise ValueError(f"bad shape {samples.shape}, expected (S, {true.shape[0]}, {true.shape[1]})")
                    m = expected_metrics((samples > 0.5).astype(int), true)
                    ate = json.load(open(os.path.join(run, "ate.json")))
                    row.update(adj_f1=m["expected_adj_f1"], orient_f1=m["expected_orient_f1"],
                               posterior_mass_g0=m["posterior_mass_g0"], expected_shd=m["expected_shd"],
                               cyclic_rate=m["cyclic_rate"],
                               ate_rmse_standardised=ate["ate_rmse_standardised"], ate_rmse_raw=ate["ate_rmse_raw"],
                               est_ate_standardised=[e["est_ate_standardised"] for e in ate["environments"]],
                               true_ate_standardised=[e["true_ate_standardised"] for e in ate["environments"]])
                    log_path = os.path.join(run, "training_log.json")
                    if os.path.exists(log_path):
                        log = json.load(open(log_path))
                        for k in ("converged", "runtime_s", "outer_rounds", "mean_edges_in_samples", "prior_sparsity_lambda"):
                            row[k] = log.get(k)
                    row.update(status="ok", error="")
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
    return pd.DataFrame(rows)


def table5(per_run: pd.DataFrame) -> pd.DataFrame:
    ok = per_run[per_run["status"] == "ok"] if len(per_run) else per_run
    out = []
    for ds in ORDER:
        for variant, label in VARIANTS:
            sub = ok[(ok["dataset"] == ds) & (ok["variant"] == variant)] if len(ok) else ok
            row = {"Dataset": ds, "Method": label}
            if len(sub) == 0:
                row.update({"Adj. F1": "Not run", "Orient. F1": "Not run", "Posterior mass on G0": "Not run",
                            "ATE RMSE (ours)": "Not run"})
            else:
                row.update({"Adj. F1": f"{sub['adj_f1'].median():.3f}",
                            "Orient. F1": f"{sub['orient_f1'].median():.3f}",
                            "Posterior mass on G0": f"{sub['posterior_mass_g0'].median():.3f}",
                            "ATE RMSE (ours)": f"{sub['ate_rmse_standardised'].median():.3f}"})
            row["ATE RMSE (published median)"] = f"{PUBLISHED[(ds, variant)]:.3f}"
            out.append(row)
    return pd.DataFrame(out)


def mean_sd(per_run: pd.DataFrame) -> pd.DataFrame:
    ok = per_run[per_run["status"] == "ok"] if len(per_run) else per_run
    cols = ["adj_f1", "orient_f1", "posterior_mass_g0", "expected_shd", "ate_rmse_standardised", "ate_rmse_raw", "runtime_s"]
    rows = []
    for ds in ORDER:
        for variant, label in VARIANTS:
            sub = ok[(ok["dataset"] == ds) & (ok["variant"] == variant)] if len(ok) else ok
            row = {"Dataset": ds, "Method": label, "n seeds": int(len(sub))}
            for c in cols:
                vals = pd.to_numeric(sub[c], errors="coerce").dropna() if len(sub) and c in sub else pd.Series(dtype=float)
                row[f"{c} mean"] = round(float(vals.mean()), 4) if len(vals) else None
                row[f"{c} sd"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/E45")
    parser.add_argument("--results", default="results/E45")
    parser.add_argument("--out", default="results/E45/summary")
    args = parser.parse_args()
    per_run = collect(args.data, args.results)
    os.makedirs(args.out, exist_ok=True)
    per_run.drop(columns=["run_dir"], errors="ignore").to_csv(os.path.join(args.out, "per_run.csv"), index=False)
    t5 = table5(per_run)
    t5.to_csv(os.path.join(args.out, "table5.csv"), index=False)
    mean_sd(per_run).to_csv(os.path.join(args.out, "table5_mean_sd.csv"), index=False)
    md = ["| " + " | ".join(t5.columns) + " |", "|" + "---|" * len(t5.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t5.itertuples(index=False)]
    open(os.path.join(args.out, "table5.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    if len(per_run):
        print("\nrun status:", per_run["status"].value_counts().to_dict())
        for r in per_run[per_run["status"] == "failed"].itertuples():
            print(f"  FAILED {r.variant}/{r.dataset} seed {r.seed}: {r.error}")
        if "converged" in per_run:
            nc = per_run[(per_run["status"] == "ok") & (per_run["converged"] == False)]  # noqa: E712
            if len(nc):
                print(f"WARNING: {len(nc)} run(s) hit the step cap (converged=false); read their rows with care.")
    else:
        print("\nno runs found under", args.results)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
