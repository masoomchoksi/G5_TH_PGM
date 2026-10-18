"""
Stage 1: Data generation, validation and loading
CSE 516 M2 - DECI audit (Theoretical AI, Mode A)

Run from the command line:
    python data_generation.py --out data                 # E1, n = 250..5000, seeds 0-4
    python data_generation.py --out data --validate-only

Layout of this file
    0. Settings            the E1 grid and defaults
    1. Graph utilities     ER / scale-free DAG samplers, acyclicity, topological order
    2. Noise models        Gaussian and non-Gaussian (monotone-warped) noise
    3. The SEM             non-linear additive-noise model, sampling, true ATE
    4. Dataset I/O         build, save and validate one dataset folder
    5. Experiment runner   generate every dataset of an experiment grid
    6. CSuite loading      read CSuite folders already on disk + ground-truth ATE
    7. Command line

Design of the E1 data (nested subsets + a different DAG per seed)
    - Each seed draws its OWN ER(16,16) DAG, mechanisms and noise, so the seeds are
      different problems and mean +/- SD across seeds reflects the graph family.
    - Within a seed, ONE pool of n_pool = 5000 training rows is sampled. The training
      set of size n is the first n rows of that pool (250 < 500 < 1000 < 2500 < 5000),
      so only the amount of data changes along the n-curve. The test set and the true
      graph are identical for every n of a seed.
    - Limitation to state in the report: the subsets overlap, so errors across n within
      a seed are correlated and the variance at each n is understated.

Convention: adj[j, i] == 1 means edge j -> i (x_j is a parent of x_i).
Dependencies: numpy, pandas only.
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. Settings
# =============================================================================
@dataclass(frozen=True)
class ExperimentGrid:
    """One experiment = every combination of (graph, n_train, seed)."""
    name: str
    graphs: List[Tuple[int, int]]      # (d, n_edges), i.e. ER(d, n_edges)
    n_train: List[int]
    seeds: List[int]
    graph_type: str = "er"
    mechanism: str = "nonlinear"
    noise: str = "warped"
    n_test: int = 2000
    n_pool: int = 5000                 # rows sampled per seed; train sets are prefixes


SEEDS = [0, 1, 2, 3, 4]                # 5 seeds = minimum in Section 10.4

EXPERIMENTS: Dict[str, ExperimentGrid] = {
    # E1 (Table 4): sample size vs structural recovery on ER(16,16).
    # Nested design: one n_pool-row dataset per seed, smaller sets are its first n rows.
    "E1": ExperimentGrid(
        name="E1",
        graphs=[(16, 16)],
        n_train=[250, 500, 1000, 2500, 5000],
        seeds=SEEDS,
    ),
}


# =============================================================================
# 1. Graph utilities
# =============================================================================
def sample_er_dag(d: int, n_edges: int, rng: np.random.Generator) -> np.ndarray:
    """Erdos-Renyi DAG with d nodes and exactly n_edges edges: ER(d, n_edges)."""
    max_edges = d * (d - 1) // 2
    if n_edges > max_edges:
        raise ValueError(f"n_edges={n_edges} exceeds max {max_edges} for d={d}")
    # Draw edges on the strictly upper triangle, then hide the order by permuting nodes.
    iu = np.triu_indices(d, k=1)
    chosen = rng.choice(len(iu[0]), size=n_edges, replace=False)
    adj = np.zeros((d, d), dtype=int)
    adj[iu[0][chosen], iu[1][chosen]] = 1
    perm = rng.permutation(d)
    return adj[np.ix_(perm, perm)]


def sample_sf_dag(d: int, n_edges: int, rng: np.random.Generator) -> np.ndarray:
    """Scale-free DAG by preferential attachment. Each new node links to about
    n_edges/d earlier nodes, chosen with probability proportional to degree + 1.
    The edge count can be slightly below n_edges."""
    m = max(1, int(round(n_edges / d)))
    adj = np.zeros((d, d), dtype=int)
    degree = np.zeros(d)
    for new in range(1, d):
        k = min(m, new)
        p = degree[:new] + 1.0
        p = p / p.sum()
        for t in rng.choice(new, size=k, replace=False, p=p):
            adj[t, new] = 1                    # earlier -> later keeps it acyclic
            degree[t] += 1
            degree[new] += 1
    perm = rng.permutation(d)
    return adj[np.ix_(perm, perm)]


def is_dag(adj: np.ndarray) -> bool:
    """Acyclicity check: repeatedly remove nodes that have no remaining parents."""
    remaining = list(range(adj.shape[0]))
    while remaining:
        sub = adj[np.ix_(remaining, remaining)]
        roots = [remaining[i] for i in range(len(remaining)) if sub[:, i].sum() == 0]
        if not roots:
            return False
        remaining = [r for r in remaining if r not in roots]
    return True


def topological_order(adj: np.ndarray) -> List[int]:
    """Parents-before-children ordering (raises if the graph has a cycle)."""
    d = adj.shape[0]
    indeg = adj.sum(axis=0).astype(int).copy()
    order, stack = [], [i for i in range(d) if indeg[i] == 0]
    while stack:
        n = stack.pop()
        order.append(n)
        for c in np.where(adj[n] == 1)[0]:
            indeg[c] -= 1
            if indeg[c] == 0:
                stack.append(c)
    if len(order) != d:
        raise ValueError("Graph is not a DAG")
    return order


# =============================================================================
# 2. Noise models
# =============================================================================
def make_noise_sampler(kind: str, rng: np.random.Generator, n_knots: int = 9):
    """Return f(n, rng) -> n samples of zero-mean, unit-variance noise.

    kind = "gaussian": N(0, 1).
    kind = "warped"  : standard normal pushed through a random strictly increasing
                       piecewise-linear map, then standardised. Non-Gaussian
                       (skewed). Approximates the report's "spline-warped noise".
    """
    if kind == "gaussian":
        return lambda n, r: r.standard_normal(n)

    if kind == "warped":
        knots = np.linspace(-3.5, 3.5, n_knots)
        slopes = rng.uniform(0.2, 3.0, size=n_knots - 1)       # all > 0 -> monotone
        values = np.concatenate([[0.0], np.cumsum(slopes * np.diff(knots))])

        def warp(g):
            out = np.interp(g, knots, values)
            lo, hi = g < knots[0], g > knots[-1]               # linear extrapolation
            out[lo] = values[0] + slopes[0] * (g[lo] - knots[0])
            out[hi] = values[-1] + slopes[-1] * (g[hi] - knots[-1])
            return out

        ref = warp(np.random.default_rng(12345).standard_normal(200_000))
        mu, sd = ref.mean(), ref.std()
        return lambda n, r: (warp(r.standard_normal(n)) - mu) / sd

    raise ValueError(f"unknown noise kind: {kind}")


# =============================================================================
# 3. The structural equation model
# =============================================================================
@dataclass
class NonlinearANM:
    """x_i = f_i(x_pa(i)) + z_i, with independent noise z_i.

    mechanism: "nonlinear" (random one-hidden-layer tanh MLP) or "linear"
    noise    : "gaussian" or "warped"
    """
    adj: np.ndarray
    mechanism: str = "nonlinear"
    noise: str = "warped"
    hidden: int = 16
    noise_scale_range: Tuple[float, float] = (0.5, 1.0)
    seed: int = 0
    params: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not is_dag(self.adj):
            raise ValueError("adj must be a DAG")
        rng = np.random.default_rng(self.seed)
        self.d = self.adj.shape[0]
        self.order = topological_order(self.adj)
        self.noise_samplers = [make_noise_sampler(self.noise, rng) for _ in range(self.d)]
        self.noise_scales = rng.uniform(*self.noise_scale_range, size=self.d)
        for i in range(self.d):
            k = int(self.adj[:, i].sum())
            if k == 0:
                self.params[i] = None
            elif self.mechanism == "linear":
                w = rng.uniform(0.5, 1.5, size=k) * rng.choice([-1, 1], size=k)
                self.params[i] = ("linear", w)
            else:
                W1 = rng.normal(0, 1.0, size=(k, self.hidden))
                b1 = rng.normal(0, 0.5, size=self.hidden)
                W2 = rng.normal(0, 1.0, size=self.hidden) / np.sqrt(self.hidden) * 2.0
                self.params[i] = ("mlp", W1, b1, W2)

    def _f(self, i: int, pa_vals: np.ndarray):
        p = self.params[i]
        if p is None:
            return 0.0
        if p[0] == "linear":
            return pa_vals @ p[1]
        _, W1, b1, W2 = p
        return np.tanh(pa_vals @ W1 + b1) @ W2

    def sample(self, n: int, seed: int = 0, do: Optional[Dict[int, float]] = None) -> np.ndarray:
        """Sample n rows. do={node: value} applies do(x_node = value): the node's
        equation is replaced by that constant."""
        rng = np.random.default_rng(seed)
        do = do or {}
        X = np.zeros((n, self.d))
        for i in self.order:
            if i in do:
                X[:, i] = do[i]
                continue
            pa = np.where(self.adj[:, i] == 1)[0]
            z = self.noise_scales[i] * self.noise_samplers[i](n, rng)
            X[:, i] = (self._f(i, X[:, pa]) if len(pa) else 0.0) + z
        return X

    def true_ate(self, treatment: int, outcome: int, a: float, b: float,
                 n: int = 200_000, seed: int = 999) -> float:
        """Ground-truth ATE(a, b) = E[x_Y | do(x_T=a)] - E[x_Y | do(x_T=b)],
        simulated from the true SEM with common random numbers for both arms."""
        xa = self.sample(n, seed=seed, do={treatment: a})[:, outcome].mean()
        xb = self.sample(n, seed=seed, do={treatment: b})[:, outcome].mean()
        return float(xa - xb)


# =============================================================================
# 4. Dataset I/O: build, save, validate
# =============================================================================
def make_dataset(d=16, n_edges=16, graph="er", mechanism="nonlinear", noise="warped",
                 n_train=1000, n_test=1000, seed=0):
    """Build the SEM and sample train/test data. Fully determined by `seed`.
    The DAG, mechanisms and noise depend on `seed`, so each seed is a different problem."""
    rng = np.random.default_rng(seed)
    sampler = sample_er_dag if graph == "er" else sample_sf_dag
    adj = sampler(d, n_edges, rng)
    sem = NonlinearANM(adj, mechanism=mechanism, noise=noise, seed=seed + 1)
    X_train = sem.sample(n_train, seed=seed + 2)
    X_test = sem.sample(n_test, seed=seed + 3)
    return sem, X_train, X_test


def make_nested_datasets(sizes: List[int], n_pool: int = 5000, **kwargs):
    """One seed -> one SEM, one pool of n_pool training rows, one test set.
    Returns (sem, {n: X_train[:n]}, X_test). Prefixes of a single pool, so the sets
    are nested: X_train(250) is the first 250 rows of X_train(500), and so on.
    Sampling the pool once matters: sem.sample(250) and sem.sample(5000)[:250] differ,
    because noise is drawn node by node."""
    if max(sizes) > n_pool:
        raise ValueError(f"max size {max(sizes)} exceeds n_pool={n_pool}")
    sem, pool, X_test = make_dataset(n_train=n_pool, **kwargs)
    return sem, {n: pool[:n].copy() for n in sorted(sizes)}, X_test


def save_dataset(out_dir: str, sem: NonlinearANM, X_train: np.ndarray,
                 X_test: np.ndarray, meta: dict) -> None:
    """Write train.csv, test.csv, adj_matrix.csv and config.json to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    cols = [f"x{i}" for i in range(sem.d)]
    pd.DataFrame(X_train, columns=cols).to_csv(os.path.join(out_dir, "train.csv"), index=False)
    pd.DataFrame(X_test, columns=cols).to_csv(os.path.join(out_dir, "test.csv"), index=False)
    np.savetxt(os.path.join(out_dir, "adj_matrix.csv"), sem.adj, fmt="%d", delimiter=",")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)


def validate_dataset_folder(folder: str) -> List[str]:
    """Return a list of problems found in one generated dataset folder (empty = OK)."""
    problems = []
    try:
        cfg = json.load(open(os.path.join(folder, "config.json")))
        tr = pd.read_csv(os.path.join(folder, "train.csv"))
        te = pd.read_csv(os.path.join(folder, "test.csv"))
        adj = np.loadtxt(os.path.join(folder, "adj_matrix.csv"), delimiter=",", ndmin=2).astype(int)
    except Exception as exc:                                    # missing/corrupt file
        return [f"cannot read files: {exc}"]
    d = cfg["d"]
    if tr.shape != (cfg["n_train"], d):
        problems.append(f"train shape {tr.shape} != ({cfg['n_train']}, {d})")
    if te.shape != (cfg["n_test"], d):
        problems.append(f"test shape {te.shape} != ({cfg['n_test']}, {d})")
    if adj.shape != (d, d):
        problems.append(f"adj shape {adj.shape} != ({d}, {d})")
    elif not is_dag(adj):
        problems.append("adjacency is not a DAG")
    elif int(adj.sum()) != cfg["n_edges"]:
        problems.append(f"edge count {int(adj.sum())} != {cfg['n_edges']}")
    # nested check: this train set must be the head of every larger sibling (same seed)
    parent = os.path.dirname(folder)
    for sib in sorted(glob.glob(os.path.join(parent, f"*_seed{cfg['seed']}"))):
        try:
            scfg = json.load(open(os.path.join(sib, "config.json")))
            if scfg["n_train"] > cfg["n_train"]:
                big = pd.read_csv(os.path.join(sib, "train.csv")).values[:cfg["n_train"]]
                if not np.allclose(big, tr.values, atol=1e-6):
                    problems.append(f"not nested in {os.path.basename(sib)}")
                if not (np.loadtxt(os.path.join(sib, "adj_matrix.csv"), delimiter=",", ndmin=2).astype(int) == adj).all():
                    problems.append(f"adjacency differs from {os.path.basename(sib)}")
        except Exception as exc:
            problems.append(f"cannot compare with {os.path.basename(sib)}: {exc}")
    if not (np.isfinite(tr.values).all() and np.isfinite(te.values).all()):
        problems.append("NaN or inf in data")
    return problems


# =============================================================================
# 5. Experiment runner
# =============================================================================
def dataset_folder_name(grid: ExperimentGrid, d: int, e: int, n: int, seed: int) -> str:
    return f"{grid.graph_type}{d}_{e}_n{n}_seed{seed}"


def generate_experiment(grid: ExperimentGrid, out_root: str, skip_existing: bool = True) -> List[str]:
    """Generate every dataset of one experiment grid. Returns the folders."""
    folders = []
    for (d, e) in grid.graphs:
        for seed in grid.seeds:
            todo = {n: os.path.join(out_root, grid.name, dataset_folder_name(grid, d, e, n, seed))
                    for n in grid.n_train}
            folders.extend(todo.values())
            if skip_existing:
                todo = {n: f for n, f in todo.items()
                        if not os.path.exists(os.path.join(f, "config.json"))}
            if not todo:
                continue
            # one pool per (graph, seed); every n is a prefix of it
            sem, subsets, Xte = make_nested_datasets(
                sorted(todo), n_pool=grid.n_pool, d=d, n_edges=e, graph=grid.graph_type,
                mechanism=grid.mechanism, noise=grid.noise, n_test=grid.n_test, seed=seed)
            for n, folder in todo.items():
                meta = dict(experiment=grid.name, d=d, n_edges=e, graph=grid.graph_type,
                            mechanism=grid.mechanism, noise=grid.noise,
                            n_train=n, n_test=grid.n_test, seed=seed,
                            design="nested_prefix_of_pool", n_pool=grid.n_pool)
                save_dataset(folder, sem, subsets[n], Xte, meta)
    return folders


def validate_experiment(folders: List[str]) -> bool:
    ok = True
    for folder in folders:
        problems = validate_dataset_folder(folder)
        if problems:
            ok = False
            print(f"  PROBLEM {folder}: {'; '.join(problems)}")
    return ok


# =============================================================================
# 6. CSuite loading (folders already downloaded; nothing is fetched)
# =============================================================================
def _read_numeric_csv(path: str) -> np.ndarray:
    """Read a CSV that may or may not have a header row."""
    df = pd.read_csv(path, header=None)
    try:
        return df.astype(float).values
    except ValueError:
        return pd.read_csv(path).values.astype(float)


def load_csuite(folder: str) -> dict:
    """Load one CSuite folder: train, val, test, adj, variables, interventions."""
    out = {"adj": _read_numeric_csv(os.path.join(folder, "adj_matrix.csv")).astype(int)}
    for split in ("train", "val", "test"):
        path = os.path.join(folder, f"{split}.csv")
        if os.path.exists(path):
            out[split] = _read_numeric_csv(path)
    for name in ("interventions", "variables"):
        path = os.path.join(folder, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                out[name] = json.load(f)
    return out


def csuite_true_ates(folder: str) -> List[dict]:
    """Ground-truth ATE for every intervention environment of a CSuite dataset:
    mean(effect | intervened) - mean(effect | reference), from interventions.json."""
    with open(os.path.join(folder, "interventions.json")) as f:
        envs = json.load(f)["environments"]
    results = []
    for k, env in enumerate(envs):
        ref = np.array(env["reference_data"])
        tst = np.array(env["test_data"])
        eff = env["effect_idxs"]
        results.append(dict(
            environment=k,
            intervention_idxs=env["intervention_idxs"],
            intervention_values=env["intervention_values"],
            intervention_reference=env["intervention_reference"],
            effect_idxs=eff,
            true_ate=(tst[:, eff].mean(axis=0) - ref[:, eff].mean(axis=0)).tolist(),
        ))
    return results


# =============================================================================
# 7. Command line
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate and validate CSE 516 M2 datasets")
    parser.add_argument("--experiments", nargs="+", default=["E1"],
                        choices=sorted(EXPERIMENTS), help="which experiment grids to build")
    parser.add_argument("--out", default="data", help="output root folder")
    parser.add_argument("--overwrite", action="store_true", help="regenerate existing folders")
    parser.add_argument("--validate-only", action="store_true", help="skip generation")
    args = parser.parse_args()

    for name in args.experiments:
        grid = EXPERIMENTS[name]
        if args.validate_only:
            folders = [os.path.join(args.out, name, dataset_folder_name(grid, d, e, n, s))
                       for (d, e) in grid.graphs for n in grid.n_train for s in grid.seeds]
        else:
            folders = generate_experiment(grid, args.out, skip_existing=not args.overwrite)
        ok = validate_experiment(folders)
        print(f"{name}: {len(folders)} datasets, validation {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
