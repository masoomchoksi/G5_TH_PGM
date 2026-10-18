"""
train_deci.py - train DECI-S / DECI-G (Causica 0.4.6) on the E1 datasets and save, for each
dataset, the files that deci_interface.py expects:

    results/E1/<variant>/<dataset_folder>/posterior_samples.npy   shape (100, d, d), adj[j, i] = 1 means j -> i
    results/E1/<variant>/<dataset_folder>/training_log.json

Written against the source of causica 0.4.6 (DECIModule, BasicDECIDataModule, AugLagLR).
Used for the reported E1 results (seeds 0-2, lambda_s = 5, i.e. --sparsity-lambda 5). Run the smoke
test below first on a new machine.

Environment (Causica 0.4.6 requires Python >=3.10,<3.11):
    python3.10 -m venv venv && source venv/bin/activate
    pip install causica==0.4.6 pandas numpy
    pip freeze > results/E1/environment_freeze.txt       # paste into the report's reproducibility record

Smoke test (one small dataset, 1 run, then check the convention and the SHD):
    python train_deci.py --variant deci_s --data data/E1 --results results/E1 --only er16_16_n250_seed0
    python deci_interface.py --variant deci_s --data data/E1 --results results/E1

Full sweeps (25 runs each):
    python train_deci.py --variant deci_s --data data/E1 --results results/E1
    python train_deci.py --variant deci_g --data data/E1 --results results/E1

What differs from the paper's Table 3 (log these in Section 9.2 / the experiment config)
    Causica 0.4.6 is a rewrite of the code the DECI paper used. Its defaults are used here unchanged:
    node-embedding functional relationships (embedding 32, out_dim_g 32, 2 layers each; not the
    paper's 2x128 residual MLPs), prior sparsity lambda 0.05 (paper: lambda_s = 5), Adam with per-module
    learning rates (graph 0.1, functional 0.0003, noise 0.003; paper: 0.01), inner-loop cap 3000 steps
    (paper: 6000), early-stopping patience 500 (paper: 1500). The augmented-Lagrangian rule
    (progress rate 0.65, rho and alpha capped at 1e13, 100 outer rounds), Gumbel-softmax temperature
    0.25 and S = 100 posterior samples match the paper. Results therefore compare to the published
    numbers only as a sanity check.
"""

import torch  # noqa: F401  MUST stay first: on Windows, importing numpy/pandas before torch can raise
#                         WinError 1114 (DLL initialisation failed) when torch is loaded afterwards.
import os
# Colab sets MPLBACKEND to a notebook-only backend that does not exist in a separate venv; force a file backend.
os.environ["MPLBACKEND"] = "Agg"
import argparse
import glob
import json
import time
import traceback

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# helpers that do not need Causica (unit-testable)
# ----------------------------------------------------------------------------
def metric_value(metrics: dict, name: str):
    """Pick a logged metric by base name. Lightning may append _step/_epoch suffixes, so prefer
    '<name>_epoch', then '<name>', then any key starting with '<name>'. Returns float or None."""
    for key in (f"{name}_epoch", name):
        if key in metrics:
            return float(metrics[key])
    for key, value in metrics.items():
        if key.startswith(name):
            return float(value)
    return None


def load_dataset(folder: str):
    cfg = json.load(open(os.path.join(folder, "config.json")))
    df = pd.read_csv(os.path.join(folder, "train.csv"))
    return cfg, df


# ----------------------------------------------------------------------------
# one training run
# ----------------------------------------------------------------------------
def train_one(folder: str, variant: str, out_dir: str, batch_size: int, max_steps: int,
              sparsity_lambda, n_samples: int = 100, progress: bool = False):
    import pytorch_lightning as pl
    import torch
    from causica.datasets.causica_dataset_format import Variable
    from causica.datasets.variable_types import VariableTypeEnum
    from causica.distributions.noise.joint import ContinuousNoiseDist
    from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
    from causica.lightning.modules.deci_module import DECIModule

    cfg, df = load_dataset(folder)
    seed = int(cfg["seed"])
    pl.seed_everything(seed, workers=True)

    # one continuous variable per column; node i = column i, so adj indices match adj_matrix.csv
    variables = [Variable(group_name=c, name=c, type=VariableTypeEnum.CONTINUOUS) for c in df.columns]
    datamodule = BasicDECIDataModule(
        dataframe=df, variables=variables, batch_size=min(batch_size, len(df)),
        normalize=True, dataset_name=os.path.basename(folder),   # z-score with training statistics
    )

    kwargs = dict(noise_dist=ContinuousNoiseDist.SPLINE if variant == "deci_s" else ContinuousNoiseDist.GAUSSIAN)
    if sparsity_lambda is not None:
        kwargs["prior_sparsity_lambda"] = sparsity_lambda
    module = DECIModule(**kwargs)

    trainer = pl.Trainer(
        max_epochs=-1,                       # stopped by the augmented-Lagrangian callback on convergence
        max_steps=max_steps,                 # hard cap; reported as not converged if reached
        accelerator="auto", devices=1,
        logger=False, enable_checkpointing=False,
        enable_progress_bar=progress, enable_model_summary=False,
    )
    start = time.time()
    trainer.fit(module, datamodule=datamodule)
    runtime = time.time() - start

    # S = 100 graphs from the learned posterior q_phi(G) (same call DECIModule.test_step_graph uses)
    module.eval()
    with torch.no_grad():
        sems = module.sem_module().sample(torch.Size([n_samples]))
        graphs = torch.stack([sem.graph for sem in sems]).cpu().numpy()
    graphs = (graphs > 0.5).astype(np.int8)

    metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
    n = len(df)
    batch_log_prob = metric_value(metrics, "batch_log_prob")
    entropy = metric_value(metrics, "vardist_entropy")
    prior = metric_value(metrics, "prior_term")
    elbo = None
    if None not in (batch_log_prob, entropy, prior):
        elbo = batch_log_prob + (entropy + prior) / n          # ELBO / N, as in the training objective
    hit_cap = max_steps is not None and max_steps > 0 and trainer.global_step >= max_steps
    log = dict(
        variant=variant, dataset=os.path.basename(folder), seed=seed, n_train=n,
        final_elbo=elbo,                                       # per training sample (ELBO / N)
        final_acyclicity_penalty=metric_value(metrics, "constraint"),
        outer_rounds=int(module.lr_scheduler.outer_opt_counter),
        converged=bool(trainer.should_stop and not hit_cap),
        steps=int(trainer.global_step), runtime_s=round(runtime, 1),
        batch_size=min(batch_size, n), prior_sparsity_lambda=module.prior_sparsity_lambda,
        mean_edges_in_samples=float(graphs.sum(axis=(1, 2)).mean()),
        causica_version="0.4.6", torch_version=torch.__version__,
    )
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "posterior_samples.npy"), graphs)
    json.dump(log, open(os.path.join(out_dir, "training_log.json"), "w"), indent=2)
    return graphs, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["deci_s", "deci_g"])
    parser.add_argument("--data", default="data/E1")
    parser.add_argument("--results", default="results/E1")
    parser.add_argument("--only", nargs="*", default=None, help="dataset folder names to run (smoke test)")
    parser.add_argument("--batch-size", type=int, default=512, help="mini-batch size (logged; tune only via a logged pilot)")
    parser.add_argument("--max-steps", type=int, default=400_000,
                        help="hard cap on optimiser steps; runs that hit it are logged converged=false")
    parser.add_argument("--sparsity-lambda", type=float, default=None, help="override DECIModule prior_sparsity_lambda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true",
                        help="show Lightning's progress bar (epoch counter) so a long run is visibly alive")
    args = parser.parse_args()

    from deci_interface import check_orientation_convention

    folders = sorted(glob.glob(os.path.join(args.data, "*")))
    if args.only:
        folders = [f for f in folders if os.path.basename(f) in set(args.only)]
    checked = False
    for folder in folders:
        name = os.path.basename(folder)
        out_dir = os.path.join(args.results, args.variant, name)
        if os.path.exists(os.path.join(out_dir, "posterior_samples.npy")) and not args.overwrite:
            print(f"[{args.variant}] {name}: exists, skipping")
            continue
        try:
            graphs, log = train_one(folder, args.variant, out_dir, args.batch_size, args.max_steps,
                                    args.sparsity_lambda, progress=args.progress)
            print(f"[{args.variant}] {name}: converged={log['converged']} steps={log['steps']} "
                  f"edges~{log['mean_edges_in_samples']:.1f} ({log['runtime_s']}s)")
            if not checked:                              # orientation convention check, once
                true = np.loadtxt(os.path.join(folder, "adj_matrix.csv"), delimiter=",", ndmin=2).astype(int)
                print("  convention check:", check_orientation_convention(graphs, true))
                checked = True
        except Exception as exc:                         # record the failure, keep going
            traceback.print_exc()
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "error.txt"), "w") as f:
                f.write(f"{type(exc).__name__}: {exc}\n")
            print(f"[{args.variant}] {name}: FAILED ({type(exc).__name__})")


if __name__ == "__main__":
    main()
