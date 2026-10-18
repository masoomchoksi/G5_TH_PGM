"""
aggregate_e1.py - builds Table 4 and the sample-size plot from results/E1/*.csv.

Usage:
    python aggregate_e1.py --results results/E1 --out results/E1/summary

Outputs (in --out)
    table4.csv / table4.md       expected SHD, mean +/- SD over seeds, rows = n, columns = method
    table4_counts.csv            how many seeds contributed to each cell (ok / failed / missing)
    shd_vs_n.png                 sample size vs SHD with +/-1 SD bands (log x axis)
    loglog_slopes.csv            slope of log(SHD + eps) on log(n), with 95% CI (t distribution)

Rules from the report
    * A cell with no successful run is printed as "missing", never silently dropped.
    * Trends are reported as a regression slope with a CI, not read off a plot (Section 12).
    * log(SHD) is undefined at SHD = 0, so we regress log(SHD + eps); eps is printed in the
      output and must be stated in the report (default 0.5).
"""

import os
# Colab sets MPLBACKEND to a notebook-only backend that does not exist in a separate venv; force a file backend.
os.environ["MPLBACKEND"] = "Agg"
import argparse
import glob

import numpy as np
import pandas as pd
from scipy import stats

METHOD_ORDER = ["deci_s", "deci_g", "notears_mlp", "pc", "icalingam"]
METHOD_LABEL = {"deci_s": "DECI-S", "deci_g": "DECI-G", "notears_mlp": "NOTEARS-MLP",
                "pc": "PC", "icalingam": "ICALiNGAM"}


def load_results(results_dir: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(results_dir, "*.csv")))]
    if not frames:
        raise FileNotFoundError(f"no result CSVs found in {results_dir}")
    df = pd.concat(frames, ignore_index=True)
    return df[df["method"].isin(METHOD_ORDER)]


def build_table(df: pd.DataFrame, sizes):
    ok = df[df["status"] == "ok"]
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    cells, counts = {}, {}
    for n in sizes:
        for m in methods:
            sub = ok[(ok["method"] == m) & (ok["n_train"] == n)]["expected_shd"]
            all_runs = df[(df["method"] == m) & (df["n_train"] == n)]
            counts[(n, m)] = dict(ok=int(len(sub)), failed=int((all_runs["status"] == "failed").sum()),
                                  missing=int((all_runs["status"] == "missing").sum()))
            if len(sub) == 0:
                cells[(n, m)] = "missing"
            elif len(sub) == 1:
                cells[(n, m)] = f"{sub.iloc[0]:.2f} (1 seed)"
            else:
                cells[(n, m)] = f"{sub.mean():.2f} ± {sub.std(ddof=1):.2f}"
    table = pd.DataFrame({METHOD_LABEL[m]: [cells[(n, m)] for n in sizes] for m in methods},
                         index=pd.Index(sizes, name="n"))
    count_table = pd.DataFrame(
        {METHOD_LABEL[m]: [f"{counts[(n, m)]['ok']} ok / {counts[(n, m)]['failed']} failed / "
                           f"{counts[(n, m)]['missing']} missing" for n in sizes] for m in methods},
        index=pd.Index(sizes, name="n"))
    return table, count_table, methods


def loglog_slopes(df: pd.DataFrame, methods, eps: float) -> pd.DataFrame:
    rows = []
    ok = df[df["status"] == "ok"]
    for m in methods:
        sub = ok[ok["method"] == m]
        if sub["n_train"].nunique() < 3 or len(sub) < 4:
            rows.append(dict(method=METHOD_LABEL[m], slope=np.nan, ci_low=np.nan, ci_high=np.nan,
                             n_points=len(sub), eps=eps, note="too few points"))
            continue
        x = np.log(sub["n_train"].values.astype(float))
        y = np.log(sub["expected_shd"].values.astype(float) + eps)
        fit = stats.linregress(x, y)
        half = stats.t.ppf(0.975, len(x) - 2) * fit.stderr
        rows.append(dict(method=METHOD_LABEL[m], slope=fit.slope, ci_low=fit.slope - half,
                         ci_high=fit.slope + half, n_points=len(x), eps=eps, note=""))
    return pd.DataFrame(rows)


def plot_shd_vs_n(df: pd.DataFrame, methods, path: str, sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = df[df["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        g = ok[ok["method"] == m].groupby("n_train")["expected_shd"]
        mean, sd = g.mean(), g.std(ddof=1).fillna(0.0)
        if len(mean) == 0:
            continue
        ax.plot(mean.index, mean.values, marker="o", label=METHOD_LABEL[m])
        ax.fill_between(mean.index, (mean - sd).clip(lower=0), mean + sd, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("training samples n")
    ax.set_ylabel("expected SHD (lower is better)")
    ax.set_title("E1: sample size vs structural recovery, ER(16,16)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/E1")
    parser.add_argument("--out", default="results/E1/summary")
    parser.add_argument("--eps", type=float, default=0.5, help="constant added before taking log(SHD)")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="sample sizes to show (default: every n found in the results)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_results(args.results)
    sizes = sorted(args.sizes) if args.sizes else sorted(int(n) for n in df["n_train"].dropna().unique())
    table, counts, methods = build_table(df, sizes)
    table.to_csv(os.path.join(args.out, "table4.csv"))
    counts.to_csv(os.path.join(args.out, "table4_counts.csv"))
    with open(os.path.join(args.out, "table4.md"), "w") as f:
        f.write(table.reset_index().to_markdown(index=False))
    slopes = loglog_slopes(df, methods, args.eps)
    slopes.to_csv(os.path.join(args.out, "loglog_slopes.csv"), index=False)
    plot_shd_vs_n(df, methods, os.path.join(args.out, "shd_vs_n.png"), sizes)

    print("Table 4 - expected SHD (mean ± SD over seeds)")
    print(table.to_string())
    print("\nlog-log slopes (eps = %.2f):" % args.eps)
    print(slopes.round(3).to_string(index=False))
    print(f"\nfiles written to {args.out}")


if __name__ == "__main__":
    main()
