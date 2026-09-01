"""
run_all_baselines_all_envs.py

Runs ARS (V1-t, V2-t), CEM-RL, CERL, PES, and PABBO(adapted) across
ALL FIVE checkpointed environments (CartPole, Swimmer, Hopper,
InvertedPendulum, MountainCarContinuous), reusing the exact
implementations from additional_baselines.py (int-seed bug already
fixed there) so results are directly comparable to everything
computed earlier in this investigation.

This does NOT re-run JEPA-PS -- those numbers already exist across
the manuscript's tables (tab:main, tab:strong_grid, tab:standardized,
etc.) and should be pulled from there, not regenerated, to avoid
introducing yet another inconsistent protocol variant.

USAGE:
  python run_all_baselines_all_envs.py --n_seeds 5 \
      --out_dir results/all_baselines_all_envs
"""
import os, sys, json, argparse, logging, time
import numpy as np

sys.path.insert(0, os.getcwd())
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Reuse the already-fixed implementations
from additional_baselines import (
    run_ars, run_cemrl, run_cerl, run_pes, run_pabbo,
)

ENVS = {
    "CartPole-v1":              8,
    "Swimmer-v4":               16,
    "Hopper-v4":                36,
    "InvertedPendulum-v4":      5,
    "MountainCarContinuous-v0": 3,
}

METHODS = {
    "ars_v1t":  lambda env, pdim, seed: run_ars(env, pdim, seed, variant="v1t"),
    "ars_v2t":  lambda env, pdim, seed: run_ars(env, pdim, seed, variant="v2t"),
    "cemrl":    run_cemrl,
    "cerl":     run_cerl,
    "pes":      run_pes,
}


def save(d, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def sanity_check_variance(results, method, env_name, min_seeds_for_check=3):
    """
    Flags (does not silently accept) any result where std==0.0 across
    n>=3 seeds on a method/env pair that is NOT CartPole -- CartPole
    hitting a hard 500 ceiling is expected and legitimate; identical-
    to-many-sig-figs results elsewhere in this investigation have
    repeatedly turned out to be harness bugs (BallLanding baselines,
    decoder const_zero==const_fixed, D3's flattened std). This does
    not touch CartPole because 500.0+-0.0 there is a known, verified
    ceiling effect, not a suspected bug.
    """
    v = results[env_name][method]
    if env_name != "CartPole-v1" and v["n"] >= min_seeds_for_check \
            and v["std"] == 0.0:
        log.warning(f"  ⚠ SUSPICIOUS: {env_name}/{method} has std=0.0 "
                    f"across {v['n']} seeds (mean={v['mean']}). "
                    f"This pattern has repeatedly indicated a harness bug "
                    f"elsewhere in this investigation (identical BallLanding "
                    f"baselines, identical decoder draws, flattened std "
                    f"computations). Inspect seeds={v['seeds']} directly "
                    f"before trusting this number.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--envs", nargs="+", default=list(ENVS.keys()))
    p.add_argument("--methods", nargs="+", default=list(METHODS.keys()) + ["pabbo"])
    p.add_argument("--out_dir", default="results/all_baselines_all_envs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out = f"{args.out_dir}/results.json"
    results = json.load(open(out)) if os.path.exists(out) else {}

    for env_name in args.envs:
        pdim = ENVS[env_name]
        env_res = results.setdefault(env_name, {})

        for method in args.methods:
            if method in env_res:
                log.info(f"  {env_name}/{method}: done "
                         f"({env_res[method]['mean']:.2f}±{env_res[method]['std']:.2f})")
                sanity_check_variance(results, method, env_name)
                continue

            log.info(f"\n=== {env_name} / {method} ===")
            bests = []
            extra = {}
            for seed in range(args.n_seeds):
                t0 = time.perf_counter()
                if method == "pabbo":
                    b, loss = run_pabbo(env_name, pdim, seed)
                    extra.setdefault("pretrain_loss", []).append(loss)
                else:
                    b = METHODS[method](env_name, pdim, seed)
                bests.append(b)
                log.info(f"  seed={seed}: {b:.2f} ({time.perf_counter()-t0:.1f}s)")

            env_res[method] = {
                "mean": float(np.mean(bests)), "std": float(np.std(bests)),
                "n": args.n_seeds, "seeds": bests, **extra,
            }
            log.info(f"  {method}: {np.mean(bests):.2f}±{np.std(bests):.2f}")
            sanity_check_variance(results, method, env_name)
            save(results, out)

    # Print full comparison table (JEPA-PS numbers pulled from the
    # manuscript, NOT regenerated here -- see note above)
    JEPA_PS_FROM_MANUSCRIPT = {
        "CartPole-v1":              (410.9, 178.2),   # tab:main, single-agent
        "Swimmer-v4":               (90.6, 54.1),      # tab:strong_grid, single-agent
        "Hopper-v4":                (136.0, 67.4),     # tab:strong_grid, single-agent
        "InvertedPendulum-v4":      (27.9, 29.9),      # tab:main, Swarm-JEPA(K=8)
        "MountainCarContinuous-v0": (-5.1, 6.3),       # tab:main, Swarm-JEPA(K=8)
    }

    log.info("\n" + "="*70)
    log.info("FULL COMPARISON TABLE (JEPA-PS pulled from manuscript, not rerun)")
    log.info("="*70)
    method_labels = {
        "ars_v1t": "ARS (V1-t)", "ars_v2t": "ARS (V2-t)",
        "cemrl": "CEM-RL", "cerl": "CERL", "pes": "PES", "pabbo": "PABBO (adapted)",
    }
    header = "Method".ljust(20) + "".join(e.ljust(28) for e in args.envs)
    log.info(header)
    row = "JEPA-PS (manuscript)".ljust(20)
    for e in args.envs:
        m, s = JEPA_PS_FROM_MANUSCRIPT.get(e, (None, None))
        row += (f"{m:.1f}±{s:.1f}".ljust(28) if m is not None else "---".ljust(28))
    log.info(row)
    for method in list(METHODS.keys()) + (["pabbo"] if "pabbo" in args.methods else []):
        row = method_labels.get(method, method).ljust(20)
        for e in args.envs:
            v = results.get(e, {}).get(method)
            row += (f"{v['mean']:.1f}±{v['std']:.1f}".ljust(28) if v else "---".ljust(28))
        log.info(row)

    # LaTeX rows
    log.info("\n" + "="*70)
    log.info("LaTeX rows")
    log.info("="*70)
    for method in list(METHODS.keys()) + (["pabbo"] if "pabbo" in args.methods else []):
        cells = []
        for e in args.envs:
            v = results.get(e, {}).get(method)
            cells.append(f"${v['mean']:.1f}{{\\pm}}{v['std']:.1f}$" if v else "---")
        log.info(f"{method_labels.get(method, method)} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
