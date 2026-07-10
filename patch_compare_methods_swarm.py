"""
Patch compare_methods.py to add Swarm-JEPA as a new comparison condition.

Run from /scratch/mroycho1/mujoco/jepa_props/:
    python patch_compare_methods_swarm.py
"""

with open("analysis/compare_methods.py") as f:
    content = f.read()

# 1. Add run_swarm_jepa function after run_jepa
old = '''def run_jepa_hybrid(env_name: str, param_dim: int, seed: int, max_iterations: int, device: str):'''

new = '''def run_swarm_jepa(env_name: str, param_dim: int, seed: int, max_iterations: int, device: str):
    """Run K-island Swarm-JEPA using PSO social force + isolation-lineage."""
    import sys, os
    sys.path.insert(0, os.getcwd())
    from swarm_jepa_runner import run_swarm_jepa as _run
    cfg = _load_cfg(env_name)
    return _run(
        env_name=env_name,
        param_dim=cfg["param_dim"],
        seed=seed,
        max_iterations=max_iterations,
        device=device,
        K=3,
        noise_std=float(cfg.get("noise_std", 0.5)),
        n_candidates=cfg.get("n_candidates", 20),
        n_eval_episodes=cfg.get("n_eval_episodes", 10),
        desired_pct=float(cfg.get("desired_pct", 90.0)),
        pretrained_ckpt=cfg.get("pretrained_ckpt", ""),
        online_finetune=cfg.get("online_finetune", True),
        online_lr=float(cfg.get("online_lr", 1e-4)),
        online_finetune_every=cfg.get("online_finetune_every", 5),
        online_finetune_steps=cfg.get("online_finetune_steps", 10),
        migration_every=30,
        isolation_every=60,
        isolation_steps=6,
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_enc_layers=cfg.get("num_enc_layers", 3),
        num_pred_layers=cfg.get("num_pred_layers", 3),
        dim_ff=cfg.get("dim_ff", 256),
        ema_decay=float(cfg.get("ema_decay", 0.996)),
        log_dir=f"results/compare/{env_name}",
        run_name=f"swarm_jepa_seed{seed}",
    )


def run_jepa_hybrid(env_name: str, param_dim: int, seed: int, max_iterations: int, device: str):'''

assert old in content, "run_jepa_hybrid anchor not found"
content = content.replace(old, new, 1)

# 2. Add SWARM-JEPA to the color map
old_colors = '''"JEPA (ours)":    "#2196F3",'''
new_colors = '''"JEPA (ours)":    "#2196F3",
    "Swarm-JEPA (ours)": "#7C4DFF",'''
assert old_colors in content, "color map anchor not found"
content = content.replace(old_colors, new_colors, 1)

# 3. Add Swarm-JEPA run block in main(), right after JEPA block
old_main = '''    log.info(f"\\n{\'=\'*50}\\nRunning Pure JEPA on {env}\\n{\'=\'*50}")
    results["JEPA (ours)"] = [
        run_jepa(env, s, args.max_iterations, args.device)["rewards"]
        for s in args.seeds
    ]'''

new_main = '''    log.info(f"\\n{\'=\'*50}\\nRunning Pure JEPA on {env}\\n{\'=\'*50}")
    results["JEPA (ours)"] = [
        run_jepa(env, s, args.max_iterations, args.device)["rewards"]
        for s in args.seeds
    ]

    # ── Swarm-JEPA (K=3 islands + PSO social force + isolation-lineage) ───
    log.info(f"\\n{\'=\'*50}\\nRunning Swarm-JEPA on {env}\\n{\'=\'*50}")
    results["Swarm-JEPA (ours)"] = [
        run_swarm_jepa(env, param_dim, s, args.max_iterations, args.device)["rewards"]
        for s in args.seeds
    ]'''

assert old_main in content, "main JEPA block anchor not found"
content = content.replace(old_main, new_main, 1)

with open("analysis/compare_methods.py", "w") as f:
    f.write(content)

print("Patched: run_swarm_jepa added to compare_methods.py")
print("New conditions in results table:")
print("  - JEPA (ours)         [single-island, existing]")
print("  - Swarm-JEPA (ours)   [K=3 islands, PSO social force, isolation-lineage]")
