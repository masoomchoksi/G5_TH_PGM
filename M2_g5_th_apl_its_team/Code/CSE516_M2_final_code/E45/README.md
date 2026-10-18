# E4-E5 (CSE 516 M2, Table 5): replication check on CSuite

4 CSuite datasets x 2 methods (DECI-G, DECI-S) x 5 training seeds = 40 runs.
Data: `lingauss`, `linexp`, `nonlingauss` (2 variables, 4000 training rows), `nonlin_simpson` (4 variables, 2000 rows).
The CSuite files are used unmodified (`data/E45/<dataset>/`). Each dataset has one fixed training set, so the
5 seeds are **training seeds only** (model initialisation and mini-batch order), not 5 different datasets. State this in the report.

## What Table 5 measures
| Column | Definition |
|---|---|
| Adj. F1 / Orient. F1 | F1 on the undirected skeleton / on directed edges, expected over 100 graphs drawn from q(G) |
| Posterior mass on G0 | fraction of the 100 sampled graphs that equal the true graph exactly |
| ATE RMSE (ours) | Causica's own `eval_ate_rmse`: 10 sampled graphs x 1000 Monte-Carlo samples; true ATE from `interventions.json`; mean over interventional environments; **standardised units** (training z-score) |
| ATE RMSE (published median) | the medians quoted in the report's Table 5 (stored in `aggregate_e45.py`) |

Every cell is the **median over the 5 training seeds** (matches the "published median" column). Mean and sd are in `table5_mean_sd.csv`.
Raw-unit ATE RMSE is in `per_run.csv` (`ate_rmse_raw`); it equals standardised x the effect variable's training std (exact, ATE is a difference of means).

## Environments
Python 3.10 for DECI (Causica 0.4.6 requires it): `pip install causica==0.4.6 pandas numpy`.
`check_csuite_data.py` and `aggregate_e45.py` need only numpy and pandas. Keep this folder together (relative paths: `data/E45`, `results/E45`).

## Run order (one command per line)
1. validate the data (writes config.json)     `python check_csuite_data.py --data data/E45`          (output of the shipped copy: `data_check.txt`)
2. smoke test (1 dataset, 1 seed)             `python train_deci_csuite.py --variant deci_g --only nonlingauss --seeds 0 --results results/E45_smoke --progress`
                                              `python aggregate_e45.py --data data/E45 --results results/E45_smoke --out results/E45_smoke/summary`
3. DECI-G, all runs                           `python train_deci_csuite.py --variant deci_g --progress`
4. DECI-S, all runs                           `python train_deci_csuite.py --variant deci_s --progress`
5. Table 5                                    `python aggregate_e45.py --data data/E45 --results results/E45 --out results/E45/summary`

Finished runs are skipped on restart. One dataset / one seed: `--only linexp --seeds 2`.
The smoke test must show `convention check: OK` and `converged=True`; `converged=False` means the 400k step cap was hit.
Use `nonlingauss` for the smoke test, not `lingauss`: `lingauss` is linear Gaussian, so both edge directions fit equally well and the convention check is skipped there on purpose.
Causica's own source confirms the convention (`graph[j, i] > 0` means j is a parent of i), the same as `adj_matrix.csv` and `metrics.py`.

## What is the same and what differs from the paper (log these in the report)
* Same code path and settings as E1: Causica 0.4.6 defaults (see the E1 README), including **prior_sparsity_lambda = 0.05**, so Table 4 and Table 5 use one DECI configuration.
  The paper's lambda_s = 5 can be run into a separate folder: `--sparsity-lambda 5 --results results/E45_lambda5` (CSuite runs are small).
* Causica 0.4.6 is a rewrite of the code behind the published numbers, so agreement with the published column is a sanity check, not an expected exact match.
* The published median may use a different number of seeds, graph samples or data scale than the 5 seeds / 10 graphs / standardised units used here; this package follows Causica 0.4.6's own evaluation protocol.
* `nonlin_simpson` has two environments (one conditioned on x3); the ATE RMSE is their mean, as in Causica's test loop. `eval_ate_rmse` does not use the conditioning, exactly as in Causica.

## Files
check_csuite_data.py   validation, true ATEs, config.json   metrics.py   SHD, F1, expected metrics (identical to E1)
train_deci_csuite.py   DECI-S/-G training + ATE (Causica 0.4.6)   aggregate_e45.py   Table 5, per-run and mean/sd CSVs

## Data location
Unzip the dataset zip next to this folder so that `data/E45/` sits inside it (the two zips merge into one tree).
