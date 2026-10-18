# E1 (CSE 516 M2, Table 4)

5 seeds x 5 methods (DECI-S, DECI-G, NOTEARS-MLP, PC, ICALiNGAM) = 25 runs.
Data: ER(16,16), non-linear ANM with non-Gaussian (warped) noise, seeds 0-4, n in {250, 500, 1000, 2500, 5000}.
Design (A + D): every seed has its own DAG; within a seed one 5000-row pool is sampled and the training set of size n
is its first n rows (nested). Test set (2000 rows) and true graph are shared across n. Limitation for the report:
nested subsets are correlated across n, so variance at each n is understated.
Regenerate with: python data_generation.py --out data --overwrite   (old n = 250 files are replaced; the n = 250 data
now differ from the earlier un-nested version, so rerun any finished n = 250 results.)
Runs: 5 seeds x 5 sizes x 5 methods = 125 runs. All 25 datasets exist up front; run seeds 0-2 first and the last two seeds later (train_deci.py --only <folder names>).

## Environments (two, because Causica 0.4.6 needs Python 3.10)
A (DECI):      Python 3.10 ->  pip install causica==0.4.6 pandas numpy
B (baselines): any Python  ->  pip install gcastle torch pandas numpy scipy matplotlib tabulate
Keep this folder together; all scripts use relative paths (data/E1, results/E1).

## Run order (one command per line)
1. (optional) regenerate / check data     python data_generation.py --out data --validate-only
2. baselines, env B                        python baselines.py --methods pc icalingam notears_mlp --data data/E1 --out results/E1
3. DECI-S, env A                           python train_deci.py --variant deci_s --data data/E1 --results results/E1 --progress
4. DECI-G, env A                           python train_deci.py --variant deci_g --data data/E1 --results results/E1 --progress
5. score DECI runs, env A or B             python deci_interface.py --variant deci_s --data data/E1 --results results/E1
                                           python deci_interface.py --variant deci_g --data data/E1 --results results/E1
6. table + plot, env B                     python aggregate_e1.py --results results/E1 --out results/E1/summary

Finished runs are skipped on restart. Put only result CSVs of this experiment in results/E1.
To run one dataset (one n, one seed): add  --only er16_16_n250_seed0  (names: er16_16_n{250,500,1000,2500,5000}_seed{0-4}).

## Files
data_generation.py  E1 data (all n, nested) + validation   metrics.py  SHD, F1, expected metrics, CPDAG expansion
baselines.py        PC / ICALiNGAM / NOTEARS-MLP (gcastle)   train_deci.py  DECI-S/-G training (Causica 0.4.6)
deci_interface.py   scores saved DECI posterior samples      aggregate_e1.py  Table 4, plot, log-log slope

## Log in the report
Causica 0.4.6 defaults differ from the paper's Table 3 (see train_deci.py docstring); SHD counts a reversed edge as 1;
baseline thresholds 0.3; PC uses gcastle's Fisher-z test; log(SHD + 0.5) for the slope (undefined with one n).

## FINAL CONFIGURATION USED FOR THE REPORTED RESULTS
* Seeds 0-2 only (3 seeds); datasets for seeds 3-4 are shipped but were not run. The table header must say "3 seeds".
* DECI-S / DECI-G: lambda_s = 5. Add `--sparsity-lambda 5 --results results/E1_lambda5` to the train_deci.py commands in steps 3-4
  (and `--results results/E1_lambda5` in step 5). Without the flag Causica's default 0.05 is used.
* PC / NOTEARS-MLP / ICA-LiNGAM do not depend on lambda; run step 2 as written (results/E1).
* Table 4 = mean +/- sample SD (n-1) of `expected_shd` over the 3 seeds.
* Data: unzip the dataset zip next to this folder so that `data/E1/` sits inside it (the two zips merge into one tree).
