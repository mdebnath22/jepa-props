"""
eval_protocol.py — Statistical rigor and profiling utilities shared across
all experiments.

Addresses reviewer requests:
  - "report confidence intervals, per-candidate repeat averages, and
    significance testing"
  - "provide end-to-end wall-clock profiles separating rollout time, JEPA
    forward time, and LLM inference time"
  - the "random best > CEM best" anomaly (via bootstrap CI on best-reward)
"""

import numpy as np
import logging
from typing import List, Dict
from scipy import stats

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across seeds
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_seeds(per_seed_results: List[dict]) -> dict:
    """
    Takes a list of per-seed result dicts (each with "rewards", "best_reward",
    optionally "wallclock"), returns aggregate statistics with confidence
    intervals on BOTH the mean-of-means and the best-of-bests, since the
    reviewer specifically flags that "best" statistics need uncertainty
    quantification too (not just means).
    """
    means = np.array([np.mean(r["rewards"]) for r in per_seed_results])
    bests = np.array([r["best_reward"] for r in per_seed_results])
    n = len(per_seed_results)

    out = {
        "n_seeds": n,
        "mean_of_means": float(means.mean()),
        "std_of_means": float(means.std()),
        "mean_ci95": _normal_ci(means),
        "mean_of_bests": float(bests.mean()),
        "std_of_bests": float(bests.std()),
        "best_ci95": _normal_ci(bests),
        "best_bootstrap_ci95": _bootstrap_ci(bests, statistic=np.max, n_boot=2000),
        "max_best": float(bests.max()),
    }

    if all("wallclock" in r for r in per_seed_results):
        rollout_s = [r["wallclock"].get("rollout_s", np.nan) for r in per_seed_results]
        optimizer_s = [r["wallclock"].get("optimizer_s", np.nan) for r in per_seed_results]
        total_s = [r["wallclock"].get("total_s", np.nan) for r in per_seed_results]
        out["wallclock"] = {
            "mean_rollout_s": float(np.nanmean(rollout_s)),
            "mean_optimizer_s": float(np.nanmean(optimizer_s)),
            "mean_total_s": float(np.nanmean(total_s)),
            "rollout_fraction": (
                float(np.nanmean(rollout_s) / np.nanmean(total_s))
                if np.nanmean(total_s) > 0 else float("nan")
            ),
        }

    return out


def _normal_ci(values: np.ndarray, conf: float = 0.95) -> tuple:
    n = len(values)
    if n < 2:
        return (float(values[0]), float(values[0])) if n == 1 else (float("nan"), float("nan"))
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(0.5 + conf / 2)
    return (float(mean - z * se), float(mean + z * se))


def _bootstrap_ci(values: np.ndarray, statistic=np.mean, n_boot: int = 2000,
                   conf: float = 0.95, seed: int = 0) -> tuple:
    """
    Nonparametric bootstrap CI — used specifically for the "best" statistic,
    where the normal approximation is less appropriate (max of small-n
    samples is not normally distributed).
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    boot_stats = np.array([
        statistic(rng.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_stats, (1 - conf) / 2 * 100)
    hi = np.percentile(boot_stats, (1 + conf) / 2 * 100)
    return (float(lo), float(hi))


# ─────────────────────────────────────────────────────────────────────────────
# Significance testing between two methods
# ─────────────────────────────────────────────────────────────────────────────

def compare_methods(results_a: List[dict], results_b: List[dict],
                     name_a: str = "A", name_b: str = "B",
                     statistic: str = "mean") -> dict:
    """
    Wilcoxon signed-rank test (paired, same seeds) between two methods'
    per-seed statistic (mean-of-rewards or best-reward). Falls back to
    Mann-Whitney U if sample sizes differ (unpaired).
    """
    if statistic == "mean":
        a_vals = np.array([np.mean(r["rewards"]) for r in results_a])
        b_vals = np.array([np.mean(r["rewards"]) for r in results_b])
    else:
        a_vals = np.array([r["best_reward"] for r in results_a])
        b_vals = np.array([r["best_reward"] for r in results_b])

    out = {
        "name_a": name_a, "name_b": name_b, "statistic": statistic,
        "mean_a": float(a_vals.mean()), "mean_b": float(b_vals.mean()),
    }

    if len(a_vals) == len(b_vals) and len(a_vals) >= 5:
        try:
            stat, p = stats.wilcoxon(a_vals, b_vals)
            out["test"] = "wilcoxon_signed_rank"
            out["statistic_value"] = float(stat)
            out["p_value"] = float(p)
        except ValueError as e:
            out["test"] = "wilcoxon_failed"
            out["error"] = str(e)
    else:
        try:
            stat, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            out["test"] = "mann_whitney_u"
            out["statistic_value"] = float(stat)
            out["p_value"] = float(p)
        except ValueError as e:
            out["test"] = "mannwhitney_failed"
            out["error"] = str(e)

    if "p_value" in out:
        out["significant_at_0.05"] = out["p_value"] < 0.05

    return out


def bonferroni_correct(p_values: Dict[str, float], n_comparisons: int = None) -> Dict[str, dict]:
    """
    Applies Bonferroni correction across a dict of {comparison_name: p_value}.
    n_comparisons defaults to len(p_values) if not given.
    """
    n = n_comparisons or len(p_values)
    alpha_corrected = 0.05 / n
    return {
        name: {
            "p_value": p,
            "alpha_corrected": alpha_corrected,
            "significant_after_correction": p < alpha_corrected,
        }
        for name, p in p_values.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Wall-clock profiling summary across many methods
# ─────────────────────────────────────────────────────────────────────────────

def summarize_wallclock(all_results: Dict[str, List[dict]]) -> str:
    """
    Produces a human-readable table of rollout vs optimizer time fraction
    per method, directly addressing the reviewer's cost-claim concern:
    "the difference between JEPA-forward vs LM-forward may be negligible
    in end-to-end wall time ... concrete runtime profiles are needed."
    """
    lines = [
        f"{'Method':<25} {'Rollout (s)':>12} {'Optimizer (s)':>14} "
        f"{'Total (s)':>10} {'Rollout %':>10}"
    ]
    lines.append("-" * 75)
    for method, results in all_results.items():
        if not results or "wallclock" not in results[0]:
            lines.append(f"{method:<25} {'N/A':>12} {'N/A':>14} {'N/A':>10} {'N/A':>10}")
            continue
        rollout_s = np.mean([r["wallclock"].get("rollout_s", np.nan) for r in results])
        opt_s = np.mean([r["wallclock"].get("optimizer_s", np.nan) for r in results])
        total_s = np.mean([r["wallclock"].get("total_s", np.nan) for r in results])
        pct = 100 * rollout_s / total_s if total_s > 0 else float("nan")
        lines.append(
            f"{method:<25} {rollout_s:>12.3f} {opt_s:>14.3f} "
            f"{total_s:>10.3f} {pct:>9.1f}%"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Normalized performance (ProPS Figure 13a)
# Normalizes each method's score to [0, 1] range across all methods on a
# given task, matching Figure 13a's "Normalized Performance" y-axis.
# Formula: (score - min_score) / (max_score - min_score)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_performance(all_results: Dict[str, List[dict]],
                           statistic: str = "mean") -> Dict[str, float]:
    """
    Normalizes mean (or best) reward across methods to [0, 1] range.

    all_results: dict of {method_name: list of per-seed result dicts}
    statistic: "mean" (mean-of-means) or "best" (mean of best-per-seed)

    Returns dict of {method_name: normalized_score (float in [0,1])}
    """
    scores = {}
    for method, seed_results in all_results.items():
        if not seed_results:
            continue
        agg = aggregate_seeds(seed_results)
        if statistic == "mean":
            scores[method] = agg["mean_of_means"]
        else:
            scores[method] = agg["mean_of_bests"]

    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())
    score_range = max_score - min_score

    if score_range < 1e-10:
        return {m: 1.0 for m in scores}

    return {
        method: float((score - min_score) / score_range)
        for method, score in scores.items()
    }


def normalize_performance_cross_task(
    all_task_results: Dict[str, Dict[str, List[dict]]],
    statistic: str = "mean",
) -> Dict[str, Dict[str, float]]:
    """
    Applies normalize_performance per task, returning a nested dict:
        {task_name: {method_name: normalized_score}}

    For building a multi-task normalized comparison table matching
    ProPS Figure 13a's bar chart across Swimmer, Hopper, MountainCar,
    InvertedPendulum.
    """
    return {
        task_name: normalize_performance(method_results, statistic)
        for task_name, method_results in all_task_results.items()
    }


def format_normalized_table(normalized_results: Dict[str, Dict[str, float]]) -> str:
    """Pretty-print a normalized performance table matching Figure 13a."""
    all_methods = sorted(
        set(m for task_scores in normalized_results.values() for m in task_scores)
    )
    lines = ["Normalized Performance (0=worst, 1=best on each task)"]
    lines.append(f"{'Method':<25} " +
                 " ".join(f"{t[:12]:>13}" for t in normalized_results))
    lines.append("-" * (25 + 14 * len(normalized_results)))
    for method in all_methods:
        scores = [normalized_results[t].get(method, float("nan"))
                  for t in normalized_results]
        row = f"{method:<25} " + " ".join(f"{s:>13.3f}" for s in scores)
        lines.append(row)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Distance-from-goal metric for Ball Landing (Table 10)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_distance_metric(per_seed_results: List[dict]) -> dict:
    """
    For the ball landing task, Table 10 reports "Distance from goal (m)"
    rather than reward. Since reward = -distance, we convert here.

    per_seed_results: same format as other methods (reward = -distance)
    Returns: mean_dist, std_dist, and a [0, inf] CI.
    """
    distances = [-np.mean(r["rewards"]) for r in per_seed_results]
    arr = np.array(distances)
    mean_dist = float(arr.mean())
    std_dist = float(arr.std())
    se = std_dist / np.sqrt(max(1, len(arr)))
    return {
        "mean_dist_m": mean_dist,
        "std_dist_m": std_dist,
        "ci95": (float(max(0, mean_dist - 1.96 * se)),
                 float(mean_dist + 1.96 * se)),
        "n_seeds": len(per_seed_results),
    }
