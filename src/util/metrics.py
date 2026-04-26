"""
Metrics for DLcatalysis 4.0 v4-innovate.

Main addition over 3.0: **within-enzyme discrimination** metrics.

Given predictions for many (enzyme, substrate) pairs, we group by enzyme
and report per-group ranking quality — this directly measures whether the
model can tell apart substrates for the SAME enzyme, which is the
real-world use case (panel screening, mutation-driven preference shift).

Global PCC/R²/MAE remain available but should be treated as secondary.
"""
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import spearmanr, kendalltau


def global_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """PCC / SCC / R² / MSE / MAE / RMSE on the full set (ignores groups)."""
    yp = np.asarray(y_pred).ravel()
    yt = np.asarray(y_true).ravel()
    valid = np.isfinite(yp) & np.isfinite(yt)
    yp, yt = yp[valid], yt[valid]
    if len(yp) < 2:
        return {"PCC": np.nan, "SCC": np.nan, "R2": np.nan,
                "MSE": np.nan, "MAE": np.nan, "RMSE": np.nan, "n": len(yp)}
    pcc = float(np.corrcoef(yt, yp)[0, 1])
    scc, _ = spearmanr(yt, yp)
    if np.isnan(pcc): pcc = 0.0
    if np.isnan(scc): scc = 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mse = float(np.mean((yt - yp) ** 2))
    mae = float(np.mean(np.abs(yt - yp)))
    return {
        "PCC": pcc, "SCC": float(scc), "R2": r2,
        "MSE": mse, "MAE": mae, "RMSE": float(np.sqrt(mse)),
        "n": int(len(yp)),
    }


# ──────────────────────────────────────────────────────────────────────
# Within-enzyme discrimination (main novel metric)
# ──────────────────────────────────────────────────────────────────────
def within_group_spearman(y_pred: np.ndarray, y_true: np.ndarray,
                          group_ids: Sequence[str],
                          min_group_size: int = 3) -> dict:
    """For each enzyme (group_id), compute Spearman between pred and true.
    Return mean / median / coverage statistics.
    """
    yp = np.asarray(y_pred).ravel()
    yt = np.asarray(y_true).ravel()
    assert len(yp) == len(yt) == len(group_ids), \
        f"length mismatch: pred {len(yp)}, true {len(yt)}, groups {len(group_ids)}"

    buckets = defaultdict(list)
    for i, g in enumerate(group_ids):
        if np.isfinite(yp[i]) and np.isfinite(yt[i]):
            buckets[g].append((yp[i], yt[i]))

    scc_list = []
    size_dist = []
    for g, pairs in buckets.items():
        if len(pairs) < min_group_size:
            continue
        p = np.array([x[0] for x in pairs])
        t = np.array([x[1] for x in pairs])
        if np.unique(t).size < 2:  # constant target → SCC undefined
            continue
        rho, _ = spearmanr(p, t)
        if not np.isnan(rho):
            scc_list.append(rho)
            size_dist.append(len(pairs))

    n_groups_total = len(buckets)
    n_groups_used = len(scc_list)
    if n_groups_used == 0:
        return {
            "within_SCC_mean": np.nan, "within_SCC_median": np.nan,
            "within_SCC_std": np.nan,
            "n_groups_used": 0, "n_groups_total": n_groups_total,
            "coverage": 0.0, "mean_group_size": np.nan,
        }
    return {
        "within_SCC_mean":   float(np.mean(scc_list)),
        "within_SCC_median": float(np.median(scc_list)),
        "within_SCC_std":    float(np.std(scc_list)),
        "n_groups_used":     int(n_groups_used),
        "n_groups_total":    int(n_groups_total),
        "coverage":          float(n_groups_used / max(n_groups_total, 1)),
        "mean_group_size":   float(np.mean(size_dist)),
    }


def within_group_top1_hit(y_pred: np.ndarray, y_true: np.ndarray,
                          group_ids: Sequence[str],
                          min_group_size: int = 3) -> dict:
    """Fraction of enzymes where the predicted top-1 substrate matches the
    true top-1 substrate (argmax agreement)."""
    yp = np.asarray(y_pred).ravel()
    yt = np.asarray(y_true).ravel()
    buckets = defaultdict(list)
    for i, g in enumerate(group_ids):
        if np.isfinite(yp[i]) and np.isfinite(yt[i]):
            buckets[g].append((yp[i], yt[i], i))

    hits = 0
    evaluated = 0
    for g, items in buckets.items():
        if len(items) < min_group_size:
            continue
        p = np.array([x[0] for x in items])
        t = np.array([x[1] for x in items])
        if np.unique(t).size < 2:
            continue
        if int(np.argmax(p)) == int(np.argmax(t)):
            hits += 1
        evaluated += 1
    hit_rate = hits / evaluated if evaluated > 0 else np.nan
    return {"within_top1_hit": float(hit_rate), "n_groups_top1_eval": evaluated}


def within_group_pair_accuracy(y_pred: np.ndarray, y_true: np.ndarray,
                               group_ids: Sequence[str],
                               min_diff: float = 0.1,
                               min_group_size: int = 2) -> dict:
    """Pairwise accuracy within each enzyme group: fraction of within-enzyme
    substrate pairs for which the predicted sign of difference matches the
    true sign. Pairs with |Δy_true| < min_diff are excluded.
    """
    yp = np.asarray(y_pred).ravel()
    yt = np.asarray(y_true).ravel()
    buckets = defaultdict(list)
    for i, g in enumerate(group_ids):
        if np.isfinite(yp[i]) and np.isfinite(yt[i]):
            buckets[g].append((yp[i], yt[i]))

    n_total = 0
    n_correct = 0
    for g, items in buckets.items():
        if len(items) < min_group_size:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                dt = items[i][1] - items[j][1]
                if abs(dt) < min_diff:
                    continue
                dp = items[i][0] - items[j][0]
                if (dp > 0) == (dt > 0):
                    n_correct += 1
                n_total += 1
    if n_total == 0:
        return {"within_pair_acc": np.nan, "n_pairs": 0}
    return {
        "within_pair_acc": float(n_correct / n_total),
        "n_pairs": int(n_total),
    }


# ──────────────────────────────────────────────────────────────────────
# Reaction-orphan retrieval (Horizyn-1 PNAS 2026 baseline)
# ──────────────────────────────────────────────────────────────────────
def top_k_recall_per_reaction(y_pred: np.ndarray,
                              reaction_ids: Sequence[str],
                              enzyme_ids: Sequence[str],
                              true_active_pairs: set,
                              k_values: Sequence[int] = (1, 5, 10, 50)) -> dict:
    """Top-K enzyme retrieval recall, grouped by reaction.

    For each reaction R appearing in `reaction_ids`, all rows with that R
    form a candidate enzyme list; we sort them by `y_pred` (descending)
    and check whether any enzyme that actually catalyzes R (according to
    `true_active_pairs`) is in the top-K.

    `true_active_pairs` is a set of `(reaction_id, enzyme_id)` tuples that
    define the ground-truth catalysis matrix; usually this is just the
    set of (R, E) pairs observed in the orphan_test split with finite
    measured kcat/Km, but the caller can construct it however makes
    sense.

    A reaction is excluded if it has no positive enzymes in the
    candidate list (otherwise recall is trivially 0).
    """
    yp = np.asarray(y_pred).ravel()
    assert len(yp) == len(reaction_ids) == len(enzyme_ids)
    by_rxn: dict = defaultdict(list)
    for i, rxn in enumerate(reaction_ids):
        if not np.isfinite(yp[i]):
            continue
        by_rxn[rxn].append((yp[i], enzyme_ids[i]))

    out = {f"top{k}_recall": [] for k in k_values}
    n_eval = 0
    for rxn, items in by_rxn.items():
        positives = {enz for r, enz in true_active_pairs if r == rxn}
        if not positives:
            continue
        items_sorted = sorted(items, key=lambda x: -x[0])
        ranked_enz = [e for _, e in items_sorted]
        n_eval += 1
        for k in k_values:
            top_k = set(ranked_enz[:k])
            out[f"top{k}_recall"].append(float(bool(top_k & positives)))

    return {
        **{m: float(np.mean(v)) if v else float("nan") for m, v in out.items()},
        "n_reactions_eval": int(n_eval),
    }


# ──────────────────────────────────────────────────────────────────────
# Unified reporter
# ──────────────────────────────────────────────────────────────────────
def report_all(y_pred: np.ndarray,
               y_true: np.ndarray,
               group_ids: Sequence[str] | None = None,
               tag: str = "") -> dict:
    """Return {global + within-enzyme} metrics as a flat dict for logging."""
    out = {}
    g = global_metrics(y_pred, y_true)
    for k, v in g.items():
        out[f"{tag}global_{k}"] = v

    if group_ids is not None:
        ws = within_group_spearman(y_pred, y_true, group_ids)
        for k, v in ws.items():
            out[f"{tag}{k}"] = v

        wt1 = within_group_top1_hit(y_pred, y_true, group_ids)
        for k, v in wt1.items():
            out[f"{tag}{k}"] = v

        wpa = within_group_pair_accuracy(y_pred, y_true, group_ids)
        for k, v in wpa.items():
            out[f"{tag}{k}"] = v
    return out
