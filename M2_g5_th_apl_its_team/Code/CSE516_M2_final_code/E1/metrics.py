"""
metrics.py - structural KPIs for E1 (CSE 516 M2).

Convention everywhere: adj[j, i] == 1 means edge j -> i.

Definitions (state these in Section 9.3 of the report)
  SHD                : number of node pairs {i, j} whose edge state differs between the
                       learned and the true graph. A missing edge, an extra edge and a
                       reversed edge each count as 1 error.
  Adjacency F1       : F1 on the undirected skeleton (direction ignored).
  Orientation F1     : F1 on directed edges (an edge counts as correct only if its
                       direction is also correct).
  Expected metric    : metric(G, G0) averaged over S graph samples G ~ q (Section 9.3).
  Posterior mass G0  : fraction of samples equal to the true graph.
  Cyclic-sample rate : fraction of samples that are not DAGs.
  Ordering accuracy  : fraction of true edges i -> j for which i is an ancestor of j in
                       the learned DAG (the learned graph orders i before j).
"""

import itertools
from typing import Dict, List, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Single-graph metrics
# ----------------------------------------------------------------------------
def _binarise(adj) -> np.ndarray:
    return (np.asarray(adj) != 0).astype(int)


def shd(pred, true) -> int:
    """Structural Hamming distance; a reversal counts as 1 error."""
    P, T = _binarise(pred), _binarise(true)
    d = P.shape[0]
    errors = 0
    for i in range(d):
        for j in range(i + 1, d):
            if (P[i, j], P[j, i]) != (T[i, j], T[j, i]):
                errors += 1
    return errors


def _prf(tp: float, n_pred: float, n_true: float):
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def adjacency_scores(pred, true):
    """(precision, recall, F1) on the undirected skeleton."""
    P, T = _binarise(pred), _binarise(true)
    Ps, Ts = np.triu(np.maximum(P, P.T), 1), np.triu(np.maximum(T, T.T), 1)
    return _prf(float((Ps & Ts).sum()), float(Ps.sum()), float(Ts.sum()))


def orientation_scores(pred, true):
    """(precision, recall, F1) on directed edges."""
    P, T = _binarise(pred), _binarise(true)
    return _prf(float((P & T).sum()), float(P.sum()), float(T.sum()))


def has_cycle(adj) -> bool:
    A = _binarise(adj)
    remaining = list(range(A.shape[0]))
    while remaining:
        sub = A[np.ix_(remaining, remaining)]
        roots = [remaining[i] for i in range(len(remaining)) if sub[:, i].sum() == 0]
        if not roots:
            return True
        remaining = [r for r in remaining if r not in roots]
    return False


def ordering_accuracy(pred, true) -> float:
    """Fraction of true edges i->j with i an ancestor of j in the learned graph.
    Returns nan if the learned graph has a cycle or the true graph has no edges."""
    P, T = _binarise(pred), _binarise(true)
    if T.sum() == 0 or has_cycle(P):
        return float("nan")
    d = P.shape[0]
    reach = P.copy()
    for _ in range(d):                       # transitive closure by repeated squaring
        new = ((reach @ reach) > 0).astype(int) | reach
        if (new == reach).all():
            break
        reach = new
    edges = np.argwhere(T == 1)
    return float(np.mean([reach[i, j] for i, j in edges]))


# ----------------------------------------------------------------------------
# Expected metrics over a set of graph samples
# ----------------------------------------------------------------------------
def expected_metrics(samples: np.ndarray, true) -> Dict[str, float]:
    """samples: array (S, d, d) of 0/1 adjacency matrices (a point estimate is S = 1).
    Returns every KPI averaged over the S samples, plus posterior-level KPIs."""
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples[None]
    T = _binarise(true)
    rows = []
    for G in samples:
        ap, ar, af = adjacency_scores(G, T)
        op, orr, of = orientation_scores(G, T)
        rows.append(dict(shd=shd(G, T), adj_precision=ap, adj_recall=ar, adj_f1=af,
                         orient_precision=op, orient_recall=orr, orient_f1=of,
                         cyclic=float(has_cycle(G)), ordering_acc=ordering_accuracy(G, T),
                         is_truth=float((_binarise(G) == T).all())))
    out = {}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[f"expected_{k}"] = float("nan") if np.isnan(vals).all() else float(np.nanmean(vals))
    out["n_samples"] = int(len(samples))
    out["posterior_mass_g0"] = out.pop("expected_is_truth")
    out["cyclic_rate"] = out.pop("expected_cyclic")
    return out


# ----------------------------------------------------------------------------
# PC returns a CPDAG: expand it into the DAGs of its Markov equivalence class
# ----------------------------------------------------------------------------
def _components(n: int, undirected: List[tuple]) -> List[List[tuple]]:
    """Group undirected edges into connected components."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in undirected:
        parent[find(a)] = find(b)
    groups: Dict[int, List[tuple]] = {}
    for a, b in undirected:
        groups.setdefault(find(a), []).append((a, b))
    return list(groups.values())


def _creates_new_vstructure(adj: np.ndarray, nodes: set, cpdag: np.ndarray) -> bool:
    """True if some node k in `nodes` has two non-adjacent parents in `adj`
    that were not already a v-structure (both edges directed) in the CPDAG."""
    skeleton = np.maximum(adj, adj.T)
    for k in nodes:
        parents = np.where(adj[:, k] == 1)[0]
        for a, b in itertools.combinations(parents, 2):
            if skeleton[a, b] == 0:
                original = cpdag[a, k] == 1 and cpdag[k, a] == 0 and cpdag[b, k] == 1 and cpdag[k, b] == 0
                if not original:
                    return True
    return False


def cpdag_to_dags(cpdag, max_component_edges: int = 18, max_samples: Optional[int] = None,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Expand a CPDAG (symmetric 1s = undirected edge, single 1 = directed edge) into
    DAGs of its Markov equivalence class, each equally likely.

    The class is a product of independent per-component choices, so we enumerate the
    valid orientations of each undirected component and
      - return the full product if it has <= max_samples members (or max_samples is None
        and the product is <= 5000), or
      - draw max_samples joint DAGs uniformly (one random valid orientation per
        component), which is an exact uniform sample of the class.
    Raises ValueError if a component is too large to enumerate."""
    C = _binarise(cpdag)
    n = C.shape[0]
    rng = rng or np.random.default_rng(0)
    directed = C * (1 - C.T)                                   # i -> j only
    undirected = [(i, j) for i in range(n) for j in range(i + 1, n) if C[i, j] and C[j, i]]
    if not undirected:
        return directed[None]

    per_component = []
    for comp in _components(n, undirected):
        if len(comp) > max_component_edges:
            raise ValueError(f"undirected component with {len(comp)} edges is too large to enumerate")
        nodes = {v for e in comp for v in e}
        valid = []
        for bits in itertools.product([0, 1], repeat=len(comp)):
            adj = directed.copy()
            for (a, b), bit in zip(comp, bits):
                if bit:
                    adj[a, b] = 1
                else:
                    adj[b, a] = 1
            if has_cycle(adj) or _creates_new_vstructure(adj, nodes, C):
                continue
            valid.append(tuple(bits))
        if not valid:                                          # invalid CPDAG from finite-sample PC
            for bits in itertools.product([0, 1], repeat=len(comp)):
                adj = directed.copy()
                for (a, b), bit in zip(comp, bits):
                    adj[a, b] = bit
                    adj[b, a] = 1 - bit
                if not has_cycle(adj):
                    valid.append(tuple(bits))
        if not valid:
            valid = [tuple([1] * len(comp))]
        per_component.append((comp, valid))

    def build(choice):
        adj = directed.copy()
        for (comp, _), bits in zip(per_component, choice):
            for (a, b), bit in zip(comp, bits):
                if bit:
                    adj[a, b] = 1
                else:
                    adj[b, a] = 1
        return adj

    total = int(np.prod([len(v) for _, v in per_component], dtype=object))
    limit = 5000 if max_samples is None else max_samples
    if total <= limit:
        graphs = [build(choice) for choice in itertools.product(*[v for _, v in per_component])]
    else:
        graphs = [build([v[rng.integers(len(v))] for _, v in per_component]) for _ in range(limit)]
    graphs = [g for g in graphs if not has_cycle(g)]
    return np.array(graphs) if graphs else directed[None]
