"""
check_csuite_data.py - validate the four CSuite datasets of E4-E5 and write a config.json per dataset.

Checks (each one fails loudly, because a wrong dataset would silently invalidate Table 5):
  * train/val/test have one column per variable in variables.json and no NaN
  * adj_matrix.csv is square, binary, and a DAG (convention adj[j, i] == 1 means j -> i)
  * every interventions.json environment has the intervened column constant at the stated
    intervention value (test_data) and at the stated reference value (reference_data)
  * prints the TRUE average treatment effect of each environment in raw and standardised units
    (standardised = divided by the effect variable's training std), so a bad ATE RMSE can be read
    against the size of the effect it is trying to recover.

Usage:  python check_csuite_data.py --data data/E45          (needs only numpy; no Causica)
"""
import argparse
import glob
import json
import os

import numpy as np

from metrics import has_cycle

DATASETS = ["lingauss", "linexp", "nonlingauss", "nonlin_simpson"]


def _as_list(x):
    return [float(v) for v in x] if isinstance(x, (list, tuple)) else [float(x)]


def check_one(folder: str) -> dict:
    name = os.path.basename(folder)
    meta = json.load(open(os.path.join(folder, "variables.json")))
    variables = meta["variables"]
    d = len(variables)
    train = np.loadtxt(os.path.join(folder, "train.csv"), delimiter=",", ndmin=2)
    val = np.loadtxt(os.path.join(folder, "val.csv"), delimiter=",", ndmin=2)
    test = np.loadtxt(os.path.join(folder, "test.csv"), delimiter=",", ndmin=2)
    for split, arr in (("train", train), ("val", val), ("test", test)):
        assert arr.shape[1] == d, f"{name}/{split}: {arr.shape[1]} columns, variables.json lists {d}"
        assert not np.isnan(arr).any(), f"{name}/{split}: NaN values"
    adj = np.loadtxt(os.path.join(folder, "adj_matrix.csv"), delimiter=",", ndmin=2).astype(int)
    assert adj.shape == (d, d) and set(np.unique(adj)) <= {0, 1}, f"{name}: bad adjacency matrix"
    assert not has_cycle(adj), f"{name}: adjacency matrix has a cycle"
    assert all(v["type"] == "continuous" for v in variables), f"{name}: non-continuous variable"

    std = train.std(axis=0, ddof=1)                      # torch.std (used by Causica's standardiser) is unbiased
    group_to_col = {v["group_name"]: i for i, v in enumerate(variables)}
    inter = json.load(open(os.path.join(folder, "interventions.json")))
    envs = []
    for k, env in enumerate(inter["environments"]):
        test_data = np.array(env["test_data"], dtype=float)
        ref_data = np.array(env["reference_data"], dtype=float)
        int_cols = env["intervention_idxs"]
        vals, refs = _as_list(env["intervention_values"]), _as_list(env["intervention_reference"])
        for col, v, r in zip(int_cols, vals, refs):
            assert np.allclose(test_data[:, col], v), f"{name} env{k}: intervened column is not constant at {v}"
            assert np.allclose(ref_data[:, col], r), f"{name} env{k}: reference column is not constant at {r}"
        for eff in env["effect_idxs"]:
            ate_raw = float(test_data[:, eff].mean() - ref_data[:, eff].mean())
            envs.append(dict(env=k, intervention_cols=int_cols, reference=refs, value=vals, effect_col=eff,
                             conditioning_idxs=env["conditioning_idxs"],
                             true_ate_raw=round(ate_raw, 4), true_ate_standardised=round(ate_raw / std[eff], 4)))
    cfg = dict(experiment="E4-E5", dataset=name, d=d, n_edges=int(adj.sum()), n_train=len(train),
               n_val=len(val), n_test=len(test), n_environments=len(inter["environments"]),
               interventions=envs, group_to_col=group_to_col, source="CSuite, files unmodified")
    json.dump(cfg, open(os.path.join(folder, "config.json"), "w"), indent=2)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/E45")
    args = parser.parse_args()
    for name in DATASETS:
        folder = os.path.join(args.data, name)
        assert os.path.isdir(folder), f"missing dataset folder {folder}"
        cfg = check_one(folder)
        print(f"{name}: d={cfg['d']} edges={cfg['n_edges']} train={cfg['n_train']} test={cfg['n_test']} OK")
        for e in cfg["interventions"]:
            print(f"    env{e['env']}: do(col {e['intervention_cols']}: {e['reference']} -> {e['value']}) "
                  f"effect col {e['effect_col']} cond {e['conditioning_idxs']}  "
                  f"true ATE raw {e['true_ate_raw']}  standardised {e['true_ate_standardised']}")
    extra = [os.path.basename(p) for p in glob.glob(os.path.join(args.data, "*")) if os.path.basename(p) not in DATASETS]
    if extra:
        print("note: extra folders ignored:", extra)


if __name__ == "__main__":
    main()
