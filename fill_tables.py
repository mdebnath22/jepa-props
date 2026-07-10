"""
fill_tables.py

Reads all results JSON files and prints LaTeX table rows
with confirmed numbers substituted. Run after standardized_comparison.py.

USAGE:
    python fill_tables.py --results_dir results/reviewer
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.getcwd())


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(v, std=None):
    if v is None:
        return r"\todo{}"
    if std is not None:
        return f"${v:.1f}{{\\pm}}{std:.1f}$"
    return f"${v:.1f}$"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/reviewer")
    p.add_argument("--robustness_dir", default="results/robustness")
    args = p.parse_args()

    rd  = args.results_dir
    rbd = args.robustness_dir

    # Load all result files
    cart_std  = load(f"{rd}/standardized_cartpole.json")
    swim_std  = load(f"{rd}/standardized_swimmer.json")
    adv_cart  = load(f"{rbd}/advanced_baselines_CartPole.json")
    adv_swim  = load(f"{rbd}/advanced_baselines_Swimmer.json")
    budget    = load(f"{rbd}/budget_scaling_CartPole-v1.json")
    pop       = load(f"{rbd}/population_extended_Swimmer-v4.json")

    # Helper: get value from standardized result
    def std_val(d, key, field="mean_best"):
        if key in d:
            return d[key].get(field)
        return None

    def std_pm(d, key):
        if key in d:
            return std_val(d, key, "mean_best"), std_val(d, key, "std_best")
        return None, None

    print("=" * 70)
    print("TABLE: tab:standardized (CartPole standardized, n=10, budget=400)")
    print("=" * 70)
    rows = [
        ("Swarm-JEPA (single agent)", 410.9, 178.2, 500.0),
    ]
    for name, key in [("CEM",    "cem"),
                       ("CMA-ES", "cmaes"),
                       ("PSO",    "pso"),
                       ("Random", "random")]:
        mu, sd = std_pm(cart_std, key)
        med = std_val(cart_std, key, "median_best")
        rows.append((name, mu, sd, med))

    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Method & Mean & Std & Median \\")
    print(r"\midrule")
    for name, mu, sd, med in rows:
        mu_s  = fmt(mu)  if mu  is not None else r"\todo{}"
        sd_s  = fmt(sd)  if sd  is not None else r"\todo{}"
        med_s = fmt(med) if med is not None else r"\todo{}"
        print(f"{name} & {mu_s} & {sd_s} & {med_s} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print()
    print("=" * 70)
    print("TABLE: tab:advanced_baselines (CartPole + Swimmer, multi-seed)")
    print("=" * 70)

    # CartPole
    print(r"\multirow{5}{*}{CartPole}")
    print(f"  Swarm-JEPA K=8 & --- & $416.1$ & $28.7$ \\\\")
    for name, key in [("TPE (Optuna)", "tpe"), ("IPOP-CMA-ES", "ipop"),
                       ("L-BFGS-B FD", "lbfgsb")]:
        mu, sd = std_pm(cart_std, key)
        # Fall back to single-seed from adv_cart
        single = None
        if key in adv_cart and "error" not in adv_cart[key]:
            single = adv_cart[key].get("best_reward")
        s_mu  = fmt(mu) if mu is not None else r"\todo{}"
        s_sd  = fmt(sd) if sd is not None else r"\todo{}"
        s_1   = fmt(single) if single is not None else "---"
        print(f"  {name} & {s_1} & {s_mu} & {s_sd} \\\\")

    # Swimmer
    print(r"\midrule")
    print(r"\multirow{3}{*}{Swimmer}")
    print(f"  Swarm-JEPA K=8 & --- & $293.1$ & $28.7$ \\\\")
    for name, key in [("IPOP-CMA-ES", "ipop"), ("TPE (Optuna)", "tpe")]:
        mu, sd = std_pm(swim_std, key)
        single = None
        if key in adv_swim and "error" not in adv_swim[key]:
            single = adv_swim[key].get("best_reward")
        s_mu = fmt(mu) if mu is not None else r"\todo{}"
        s_sd = fmt(sd) if sd is not None else r"\todo{}"
        s_1  = fmt(single) if single is not None else "---"
        print(f"  {name} & {s_1} & {s_mu} & {s_sd} \\\\")

    print()
    print("=" * 70)
    print("TABLE: tab:population_scaling")
    print("=" * 70)
    known = {1: (133.38, 0.0), 2: (147.96, 0.0), 4: (181.52, 0.0),
             8: (241.24, 19.72), 16: (251.65, 12.73)}
    # K=32: from pop json or hardcode seed=0=322.81
    if "K_32" in pop and pop["K_32"].get("seeds"):
        s32 = pop["K_32"]["seeds"]
        known[32] = (float(np.mean(s32)), float(np.std(s32)))
    print(r"$K$ & " + " & ".join(str(k) for k in [1,2,4,8,16,32]) + r" \\")
    means = []
    stds  = []
    for k in [1, 2, 4, 8, 16, 32]:
        if k in known:
            means.append(f"${known[k][0]:.1f}$")
            stds.append(f"${known[k][1]:.1f}$")
        else:
            means.append(r"\todo{}")
            stds.append(r"\todo{}")
    print("Mean best & " + " & ".join(means) + r" \\")
    print("Std       & " + " & ".join(stds)  + r" \\")

    print()
    print("=" * 70)
    print("TABLE: tab:budget_scaling (CEM row from budget_scaling json)")
    print("=" * 70)
    cem_row = {}
    if "cem" in budget:
        for b_str, v in budget["cem"].items():
            try:
                b = int(b_str)
                cem_row[b] = v.get("mean", v) if isinstance(v, dict) else v
            except Exception:
                pass
    budgets = [10, 20, 50, 100, 200, 400]
    cem_vals = [fmt(cem_row.get(b)) for b in budgets]
    print("CEM & " + " & ".join(cem_vals) + r" \\")

    print()
    print("=" * 70)
    print("SUMMARY: what still needs to run")
    print("=" * 70)
    missing = []
    for key in ["cem", "cmaes", "random", "pso", "tpe", "ipop"]:
        if key not in cart_std:
            missing.append(f"CartPole/{key}")
    for key in ["tpe", "ipop"]:
        if key not in swim_std:
            missing.append(f"Swimmer/{key}")
    if missing:
        print("  Missing:", ", ".join(missing))
    else:
        print("  All results present!")


if __name__ == "__main__":
    main()
