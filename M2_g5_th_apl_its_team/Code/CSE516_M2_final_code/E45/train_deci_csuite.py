"""
train_deci_csuite.py - train DECI-S / DECI-G (Causica 0.4.6) on the four CSuite datasets of E4-E5 and
save, for every (dataset, training seed), everything Table 5 needs:

    results/E45/<variant>/<dataset>_seed<k>/posterior_samples.npy   (100, d, d), adj[j, i] = 1 means j -> i
    results/E45/<variant>/<dataset>_seed<k>/training_log.json
    results/E45/<variant>/<dataset>_seed<k>/ate.json                 per interventional environment

Why this file differs from train_deci.py (E1)
    E1 uses BasicDECIDataModule (a bare dataframe). CSuite needs the interventional data, so this file
    uses Causica's own VariableSpecDataModule, the loader Causica uses for CSuite. With standardize=True it
    fits a z-score on the TRAINING split only and applies it to the interventional data and intervention
    values as well, so the model and the ground-truth ATE live on the same scale.

How the ATE RMSE is computed (identical to DECIModule.test_step_interventions in Causica 0.4.6)
    * 10 graphs are drawn from q_phi(G) (causica NUM_ATE_ITE_SEMS), 1000 Monte-Carlo samples per graph.
    * The model ATE is the mean over those graphs of  E[effect | do(a)] - E[effect | do(b)].
    * The true ATE is  mean(effect in test_data) - mean(effect in reference_data)  from interventions.json.
    * Per environment: RMSE over the effect variables (causica.training.evaluation.eval_ate_rmse).
      The reported ATE RMSE is the mean over environments (what Lightning's test loop reports).
    * Units: STANDARDISED (training std). The raw-unit value is saved too: ATE is a difference of means, so
      raw = standardised x training std of the effect variable, exactly.

Environment (Causica 0.4.6 requires Python >=3.10,<3.11):
    python3.10 -m venv venv && source venv/bin/activate
    pip install causica==0.4.6 pandas numpy
    pip freeze > results/E45/environment_freeze.txt

Smoke test (one dataset, one seed; minutes on CPU). Use nonlingauss, NOT lingauss: the orientation-convention check is
only meaningful where the edge direction is identifiable (lingauss is linear Gaussian, so x0->x1 and x1->x0 fit equally well):
    python train_deci_csuite.py --variant deci_g --only nonlingauss --seeds 0 --results results/E45_smoke --progress
    python aggregate_e45.py --data data/E45 --results results/E45_smoke --out results/E45_smoke/summary

Full sweep (4 datasets x 5 seeds = 20 runs per variant):
    python train_deci_csuite.py --variant deci_g
    python train_deci_csuite.py --variant deci_s

Settings = the Causica 0.4.6 defaults used by E1 (see the E1 README), including prior_sparsity_lambda = 0.05, so
Table 4 and Table 5 use the same DECI configuration. To run the paper's lambda_s = 5 instead, add
    --sparsity-lambda 5 --results results/E45_lambda5
(a separate folder, so the two settings never mix).
"""

import torch  # noqa: F401  MUST stay first: on Windows, importing numpy/pandas before torch can raise WinError 1114
import os

os.environ["MPLBACKEND"] = "Agg"
import argparse
import json
import time
import traceback

import numpy as np

DATASETS = ["lingauss", "linexp", "nonlingauss", "nonlin_simpson"]


# ----------------------------------------------------------------------------
# helpers that do not need Causica
# ----------------------------------------------------------------------------
def metric_value(metrics: dict, name: str):
    """Pick a logged Lightning metric by base name ('<name>_epoch', '<name>', or any prefix match)."""
    for key in (f"{name}_epoch", name):
        if key in metrics:
            return float(metrics[key])
    for key, value in metrics.items():
        if key.startswith(name):
            return float(value)
    return None


def check_orientation_convention(samples: np.ndarray, true: np.ndarray) -> str:
    """Compare the expected SHD of the samples with that of their transpose. If the transposed samples are
    much closer to the truth, the Causica adjacency uses the opposite convention."""
    def shd(P, T):
        d = P.shape[0]
        return sum((P[i, j], P[j, i]) != (T[i, j], T[j, i]) for i in range(d) for j in range(i + 1, d))
    s = np.mean([shd(G, true) for G in samples])
    s_t = np.mean([shd(G.T, true) for G in samples])
    if s_t < 0.7 * s:
        return f"WARNING: transposed SHD {s_t:.2f} << direct SHD {s:.2f}. Convention is probably flipped."
    return f"OK: direct SHD {s:.2f} vs transposed SHD {s_t:.2f}."


# ----------------------------------------------------------------------------
# one training run
# ----------------------------------------------------------------------------
def train_one(folder: str, variant: str, seed: int, out_dir: str, batch_size: int, max_steps: int,
              sparsity_lambda, n_samples: int = 100, ate_sems: int = None, progress: bool = False):
    import pytorch_lightning as pl
    from causica.distributions.noise.joint import ContinuousNoiseDist
    from causica.lightning.data_modules.variable_spec_data import VariableSpecDataModule
    from causica.lightning.modules.deci_module import NUM_ATE_ITE_SEMS, DECIModule
    from causica.sem.structural_equation_model import ate as sem_ate
    from causica.training.evaluation import eval_ate_rmse

    name = os.path.basename(folder)
    cfg = json.load(open(os.path.join(folder, "config.json")))
    true_adj = np.loadtxt(os.path.join(folder, "adj_matrix.csv"), delimiter=",", ndmin=2).astype(int)
    pl.seed_everything(seed, workers=True)

    datamodule = VariableSpecDataModule(
        root_path=folder, batch_size=min(batch_size, cfg["n_train"]), dataset_name=name,
        standardize=True,                       # z-score fitted on the training split only
        load_interventional=True,               # interventions.json, standardised with the same statistics
    )
    kwargs = dict(noise_dist=ContinuousNoiseDist.SPLINE if variant == "deci_s" else ContinuousNoiseDist.GAUSSIAN)
    if sparsity_lambda is not None:
        kwargs["prior_sparsity_lambda"] = sparsity_lambda
    module = DECIModule(**kwargs)

    trainer = pl.Trainer(
        max_epochs=-1, max_steps=max_steps, accelerator="auto", devices=1,
        logger=False, enable_checkpointing=False,
        enable_progress_bar=progress, enable_model_summary=False,
    )
    start = time.time()
    trainer.fit(module, datamodule=datamodule)      # calls datamodule.prepare_data(): loads + standardises everything
    runtime = time.time() - start

    module.eval()
    with torch.no_grad():
        # (1) S = 100 graphs from q_phi(G), same call DECIModule.test_step_graph uses
        sems = module.sem_module().sample(torch.Size([n_samples]))
        graphs = (torch.stack([sem.graph for sem in sems]).cpu().numpy() > 0.5).astype(np.int8)

        # (2) ATE RMSE, Causica's own evaluator, per interventional environment
        n_ate = ate_sems or NUM_ATE_ITE_SEMS
        ate_models = list(module.sem_module().sample(torch.Size([n_ate])))
        group_to_col = cfg["group_to_col"]
        train_std = np.loadtxt(os.path.join(folder, "train.csv"), delimiter=",", ndmin=2).std(axis=0, ddof=1)
        envs = []
        for k, intervention_with_effects in enumerate(datamodule.interventions):
            int_a, int_b, effects = intervention_with_effects
            rmse = eval_ate_rmse(ate_models, intervention_with_effects)           # {effect: RMSE}, standardised units
            per_graph = [sem_ate(sem, int_a.intervention_values, int_b.intervention_values, effects, 1000)
                         for sem in ate_models]
            for eff in sorted(effects):
                est = torch.stack([g[eff] for g in per_graph]).mean(0).reshape(-1)
                true = (int_a.intervention_data[eff].mean(0) - int_b.intervention_data[eff].mean(0)).reshape(-1)
                assert est.numel() == 1, "multi-dimensional effect variables are not used in these four datasets"
                std = float(train_std[group_to_col[eff]])
                envs.append(dict(
                    env=k, effect=eff,
                    ate_rmse_standardised=float(rmse[eff]),
                    ate_rmse_raw=float(rmse[eff]) * std,
                    est_ate_standardised=float(est), true_ate_standardised=float(true),
                    est_ate_raw=float(est) * std, true_ate_raw=float(true) * std,
                    effect_train_std=std))
        ate_summary = dict(
            ate_rmse_standardised=float(np.mean([e["ate_rmse_standardised"] for e in envs])) if envs else None,
            ate_rmse_raw=float(np.mean([e["ate_rmse_raw"] for e in envs])) if envs else None,
            n_graph_samples_for_ate=n_ate, samples_per_graph=1000, environments=envs)

    metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
    n = cfg["n_train"]
    batch_log_prob, entropy, prior = (metric_value(metrics, m) for m in ("batch_log_prob", "vardist_entropy", "prior_term"))
    elbo = batch_log_prob + (entropy + prior) / n if None not in (batch_log_prob, entropy, prior) else None
    hit_cap = max_steps is not None and max_steps > 0 and trainer.global_step >= max_steps
    log = dict(
        variant=variant, dataset=name, seed=seed, n_train=n, final_elbo=elbo,
        final_acyclicity_penalty=metric_value(metrics, "constraint"),
        outer_rounds=int(module.lr_scheduler.outer_opt_counter),
        converged=bool(trainer.should_stop and not hit_cap),
        steps=int(trainer.global_step), runtime_s=round(runtime, 1),
        batch_size=min(batch_size, n), prior_sparsity_lambda=module.prior_sparsity_lambda,
        mean_edges_in_samples=float(graphs.sum(axis=(1, 2)).mean()), true_edges=int(true_adj.sum()),
        causica_version="0.4.6", torch_version=torch.__version__,
    )
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "posterior_samples.npy"), graphs)
    json.dump(log, open(os.path.join(out_dir, "training_log.json"), "w"), indent=2)
    json.dump(ate_summary, open(os.path.join(out_dir, "ate.json"), "w"), indent=2)
    return graphs, log, ate_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["deci_s", "deci_g"])
    parser.add_argument("--data", default="data/E45")
    parser.add_argument("--results", default="results/E45")
    parser.add_argument("--only", nargs="*", default=None, choices=DATASETS, help="dataset names to run (smoke test)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4], help="training seeds (the data are fixed)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=400_000, help="hard cap; runs that hit it are logged converged=false")
    parser.add_argument("--sparsity-lambda", type=float, default=None, help="override prior_sparsity_lambda (default: Causica 0.05)")
    parser.add_argument("--ate-sems", type=int, default=None, help="graphs used for the ATE (default: Causica's 10)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    checked = False
    for name in (args.only or DATASETS):
        folder = os.path.join(args.data, name)
        if not os.path.exists(os.path.join(folder, "config.json")):
            raise SystemExit(f"{folder}/config.json missing: run  python check_csuite_data.py --data {args.data}  first")
        for seed in args.seeds:
            tag = f"{name}_seed{seed}"
            out_dir = os.path.join(args.results, args.variant, tag)
            if os.path.exists(os.path.join(out_dir, "ate.json")) and not args.overwrite:
                print(f"[{args.variant}] {tag}: exists, skipping")
                continue
            try:
                graphs, log, ate_summary = train_one(folder, args.variant, seed, out_dir, args.batch_size, args.max_steps,
                                                     args.sparsity_lambda, ate_sems=args.ate_sems, progress=args.progress)
                print(f"[{args.variant}] {tag}: converged={log['converged']} steps={log['steps']} "
                      f"edges~{log['mean_edges_in_samples']:.2f} (true {log['true_edges']}) "
                      f"ATE RMSE {ate_summary['ate_rmse_standardised']:.3f} ({log['runtime_s']}s)")
                if not checked and name != "lingauss":      # linear Gaussian: both directions fit equally well, so the check is meaningless there
                    true = np.loadtxt(os.path.join(folder, "adj_matrix.csv"), delimiter=",", ndmin=2).astype(int)
                    print("  convention check:", check_orientation_convention(graphs, true))
                    checked = True
            except Exception as exc:
                traceback.print_exc()
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "error.txt"), "w") as f:
                    f.write(f"{type(exc).__name__}: {exc}\n")
                print(f"[{args.variant}] {tag}: FAILED ({type(exc).__name__})")


if __name__ == "__main__":
    main()
