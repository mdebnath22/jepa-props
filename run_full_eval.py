"""
run_full_eval.py — Single entrypoint for the full reviewer-requested
evaluation suite. Every experiment from the review maps to a flag here
rather than a standalone script.

USAGE EXAMPLES
--------------
# All standard baselines (CMA-ES, PSO, BO, NES, (mu+lambda)-ES, Tabu, CEM,
# Random) on CartPole and Swimmer, 5 seeds each:
    python run_full_eval.py --mode baselines --tasks CartPole-v1 Swimmer-v4 \
        --seeds 0 1 2 3 4 --max_iterations 400

# Full 12-task sweep (matches available subset of ProPS's 15 tasks):
    python run_full_eval.py --mode baselines --tasks all --seeds 0 1 2 \
        --max_iterations 400

# JEPA-specific ablations (requires cluster-only swarm_jepa_runner.py etc.):
    python run_full_eval.py --mode ablations --ablation k_sweep \
        --tasks CartPole-v1 --seeds 0 --pretrained_ckpt checkpoints/jepa_pretrained/jepa_pretrained.pt

# Repeated-evaluation protocol (multi-rollout-per-candidate) wrapped around
# any baseline, to address evaluation-noise concerns:
    python run_full_eval.py --mode baselines --tasks CartPole-v1 --seeds 0 1 2 \
        --max_iterations 400 --n_repeats 5

OUTPUT
------
Writes one CSV per (task, method) under results_dir/, plus a consolidated
summary_<task>.txt with aggregate statistics, significance tests, and a
wall-clock profiling table.
"""

import argparse
import os
import csv
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs import list_available_tasks, get_param_dim
from baselines import BASELINE_REGISTRY
from eval_protocol import aggregate_seeds, compare_methods, bonferroni_correct, summarize_wallclock


ALL_BASELINES = list(BASELINE_REGISTRY.keys())

ABLATION_REGISTRY_NAMES = [
    "k_sweep", "migration_freq", "noise_sigma", "weight_schedule",
    "online_finetune", "selection_rule", "latent_cem", "pso_terms",
]


def save_results_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "reward"])
        for i, r in enumerate(results["rewards"]):
            writer.writerow([i, r])


def run_baselines_mode(args):
    tasks = list_available_tasks() if "all" in args.tasks else args.tasks

    for task in tasks:
        param_dim = get_param_dim(task)
        log.info(f"\n{'='*60}\nTask: {task} (param_dim={param_dim})\n{'='*60}")

        baseline_names = ALL_BASELINES if "all" in args.baselines else args.baselines
        all_results = {}

        for name in baseline_names:
            if name not in BASELINE_REGISTRY:
                log.warning(f"Unknown baseline '{name}', skipping")
                continue
            fn = BASELINE_REGISTRY[name]
            seed_results = []
            for seed in args.seeds:
                csv_path = os.path.join(args.results_dir, task, f"{name}_seed{seed}.csv")
                if os.path.exists(csv_path) and not args.force_rerun:
                    log.info(f"  {name} seed={seed}: SKIPPED (already exists, "
                             f"use --force_rerun to redo)")
                    with open(csv_path) as cf:
                        reader = csv.DictReader(cf)
                        rewards = [float(row["reward"]) for row in reader]
                    if rewards:
                        seed_results.append({"rewards": rewards,
                                             "best_reward": max(rewards),
                                             "wallclock": {}})
                    continue
                try:
                    if args.n_repeats > 1:
                        from ablations import run_with_repeated_evaluation
                        r = run_with_repeated_evaluation(
                            fn, task, param_dim, seed, args.max_iterations,
                            n_repeats=args.n_repeats, n_episodes=args.n_episodes,
                        )
                    else:
                        r = fn(task, param_dim, seed, args.max_iterations,
                               n_episodes=args.n_episodes)
                    seed_results.append(r)
                    save_results_csv(
                        r, os.path.join(args.results_dir, task, f"{name}_seed{seed}.csv")
                    )
                    log.info(f"  {name} seed={seed}: best={r['best_reward']:.2f} "
                             f"mean={sum(r['rewards'])/len(r['rewards']):.2f}")
                except Exception as e:
                    log.error(f"  {name} seed={seed} FAILED: {e}")

            if seed_results:
                all_results[name] = seed_results

        _write_summary(task, all_results, args.results_dir)


def _write_summary(task, all_results, results_dir):
    summary_path = os.path.join(results_dir, f"summary_{task.replace('/', '_')}.txt")
    os.makedirs(results_dir, exist_ok=True)

    lines = [f"Summary for {task}", "=" * 60, ""]
    lines.append(f"{'Method':<22} {'Mean±Std':>18} {'Best (95% CI boot)':>28}")
    lines.append("-" * 70)

    for name, seed_results in all_results.items():
        agg = aggregate_seeds(seed_results)
        ci = agg["best_bootstrap_ci95"]
        lines.append(
            f"{name:<22} {agg['mean_of_means']:>8.2f}±{agg['std_of_means']:<8.2f} "
            f"{agg['mean_of_bests']:>10.2f} [{ci[0]:.1f}, {ci[1]:.1f}]"
        )

    lines.append("")
    lines.append("Wall-clock profile (rollout vs optimizer time):")
    lines.append(summarize_wallclock(all_results))

    if len(all_results) >= 2:
        best_method = max(
            all_results, key=lambda m: aggregate_seeds(all_results[m])["mean_of_means"]
        )
        lines.append("")
        lines.append(f"Significance tests vs. best method ({best_method}):")
        p_values = {}
        for name, seed_results in all_results.items():
            if name == best_method:
                continue
            comp = compare_methods(all_results[best_method], seed_results,
                                    best_method, name)
            if "p_value" in comp:
                p_values[name] = comp["p_value"]
                lines.append(f"  {best_method} vs {name}: p={comp['p_value']:.4f} "
                             f"({comp['test']})")
        if p_values:
            corrected = bonferroni_correct(p_values)
            lines.append("")
            lines.append("After Bonferroni correction:")
            for name, info in corrected.items():
                sig = "significant" if info["significant_after_correction"] else "not significant"
                lines.append(f"  {name}: {sig} (corrected alpha={info['alpha_corrected']:.4f})")

    summary_text = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(summary_text)
    log.info(f"Summary saved: {summary_path}")


def run_ablations_mode(args):
    """Dispatches to ablations.py functions. Requires cluster-only JEPA code."""
    import ablations as abl

    task = args.tasks[0]
    seed = args.seeds[0]

    # IMPORTANT: ablations.py wraps the EXISTING swarm_jepa_runner.py /
    # pretrained-checkpoint pipeline, which uses the cluster's original
    # param_dim convention (no bias term), NOT envs.py's convention (with
    # bias). These differ for CartPole (8 vs 10) and likely other tasks.
    # When a pretrained checkpoint is supplied, infer param_dim from the
    # checkpoint itself rather than from envs.py, to guarantee a match.
    # pso_terms uses plain parameter-space PSO (baselines.run_pso_param_space)
    # and does NOT load a JEPA checkpoint -- it must use envs.py's rollout_policy
    # and param_dim convention (with bias term), not the checkpoint's.
    if args.pretrained_ckpt and args.ablation != "pso_terms":
        import torch
        ckpt = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        decoder_bias_key = [k for k in sd.keys() if k.endswith("decoder.4.bias")]
        if decoder_bias_key:
            param_dim = sd[decoder_bias_key[0]].shape[0]
            log.info(f"Inferred param_dim={param_dim} from checkpoint "
                     f"(overriding envs.py's get_param_dim={get_param_dim(task)})")
        else:
            param_dim = get_param_dim(task)
            log.warning("Could not infer param_dim from checkpoint, using envs.py default "
                       "-- this WILL fail if it doesn't match the checkpoint")
    else:
        param_dim = get_param_dim(task)

    common_kwargs = dict(
        env_name=task, param_dim=param_dim, seed=seed,
        max_iterations=args.max_iterations,
        pretrained_ckpt=args.pretrained_ckpt, device=args.device,
    )

    if args.ablation == "k_sweep":
        results = abl.sweep_population_size(**common_kwargs)
    elif args.ablation == "migration_freq":
        results = abl.sweep_migration_frequency(**common_kwargs)
    elif args.ablation == "noise_sigma":
        results = abl.sweep_noise_std(**common_kwargs)
    elif args.ablation == "weight_schedule":
        results = abl.compare_weight_schedules(**common_kwargs)
    elif args.ablation == "online_finetune":
        results = abl.compare_online_finetuning(**common_kwargs)
    elif args.ablation == "selection_rule":
        results = abl.compare_selection_rules(**common_kwargs)
    elif args.ablation == "latent_cem":
        results = {"latent_cem": abl.run_latent_space_cem(**common_kwargs)}
    elif args.ablation == "pso_terms":
        from baselines import run_pso_param_space
        results = {}
        for term in ("all", "inertia_only", "cognitive_only", "social_only"):
            results[term] = run_pso_param_space(
                task, param_dim, seed, args.max_iterations,
                n_episodes=args.n_episodes, force_terms=term,
            )
    else:
        raise ValueError(f"Unknown ablation: {args.ablation}")

    out_path = os.path.join(args.results_dir, f"ablation_{args.ablation}_{task.replace('/','_')}.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {k: {"best_reward": v["best_reward"],
                 "mean_reward": sum(v["rewards"]) / len(v["rewards"]),
                 "wallclock": v.get("wallclock", {})}
             for k, v in results.items()},
            f, indent=2,
        )
    log.info(f"Ablation results saved: {out_path}")
    for k, v in results.items():
        log.info(f"  {k}: best={v['best_reward']:.2f} "
                 f"mean={sum(v['rewards'])/len(v['rewards']):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["baselines", "ablations"], required=True)
    p.add_argument("--tasks", nargs="+", default=["CartPole-v1"],
                   help="Task names, or 'all' for all available ProPS-matching tasks")
    p.add_argument("--baselines", nargs="+", default=["all"],
                   help=f"Baseline names from {ALL_BASELINES}, or 'all'")
    p.add_argument("--ablation", choices=ABLATION_REGISTRY_NAMES, default=None,
                   help="Which ablation to run (mode=ablations only)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max_iterations", type=int, default=400)
    p.add_argument("--n_episodes", type=int, default=5,
                   help="Episodes per single rollout evaluation")
    p.add_argument("--n_repeats", type=int, default=1,
                   help="Independent repeats per candidate (statistical rigor; reviewer request)")
    p.add_argument("--pretrained_ckpt", type=str, default="",
                   help="Path to pretrained JEPA checkpoint (ablations mode)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--results_dir", type=str, default="results/full_eval")
    p.add_argument("--force_rerun", action="store_true",
                   help="Rerun even if a result CSV already exists (default: skip/resume)")

    args = p.parse_args()

    if args.mode == "baselines":
        run_baselines_mode(args)
    elif args.mode == "ablations":
        if not args.ablation:
            p.error("--ablation required when --mode=ablations")
        run_ablations_mode(args)


if __name__ == "__main__":
    main()
