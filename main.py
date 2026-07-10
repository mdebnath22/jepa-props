#!/usr/bin/env python3
"""
JEPA-ProPS: Prompted Policy Search with JEPA instead of LLMs
=============================================================
Usage:
    python main.py --config configs/cartpole_jepa.yaml
    python main.py --config configs/cartpole_jepa_hybrid.yaml
    python main.py --config configs/swimmer_jepa.yaml

Config keys:
    task        : 'cont_space_jepa'           → pure JEPA (no LLM)
                  'cont_state_jepa_semantics'  → LLM+JEPA hybrid
                  'pretrain_jepa'              → offline pretraining
    env_name    : Gymnasium environment ID
    param_dim   : number of policy parameters
    ... (see individual runners for full key lists)
"""

import yaml
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="JEPA-based Prompted Policy Search")
    parser.add_argument(
        "--config", type=str, default="configs/cartpole_jepa.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task = config.get("task", "cont_space_jepa")
    log.info(f"Running task: {task}")

    if task in ["cont_space_jepa", "disc_state_jepa"]:
        # Pure JEPA — no LLM
        from runner.jepa_runner import run_training_loop
        results = run_training_loop(**config)

    elif task in ["cont_state_jepa_semantics", "disc_state_jepa_semantics"]:
        # LLM+JEPA hybrid
        from runner.jepa_semantics_runner import run_training_loop
        results = run_training_loop(**config)

    elif task == "pretrain_jepa":
        # Offline JEPA pretraining
        _run_pretraining(config)
        return

    else:
        raise ValueError(f"Unknown task: {task}. "
                         f"Choose from: cont_space_jepa, cont_state_jepa_semantics, pretrain_jepa")

    log.info(f"Done. Best reward: {results['best_reward']:.4f}")
    log.info(f"Log saved to: {results['log_path']}")


def _run_pretraining(config: dict):
    """Run offline JEPA pretraining."""
    import os
    from agent.jepa_agent import JEPAAgent
    from agent.jepa_pretrain import (
        build_pretraining_dataset,
        JEPAPretrainDataset,
        pretrain_jepa,
    )

    env_configs = config.get("env_configs", [
        {"name": "CartPole-v1",              "param_dim": 8},
        {"name": "MountainCarContinuous-v0", "param_dim": 6},
        {"name": "InvertedPendulum-v4",      "param_dim": 16},
        {"name": "Swimmer-v4",               "param_dim": 17},
    ])

    data_path = config.get("data_path", "data/jepa_pretrain.pkl")
    save_dir = config.get("save_dir", "checkpoints")
    device = config.get("device", "cpu")
    param_dim = config.get("param_dim", 17)   # max param dim across envs

    # Use the param_dim from the first env as the representative
    first_env_param_dim = env_configs[0]["param_dim"]

    # Generate data (or load if exists)
    import pickle, os
    if os.path.exists(data_path):
        log.info(f"Loading existing pretraining data from {data_path}")
        with open(data_path, "rb") as f:
            samples = pickle.load(f)
    else:
        samples = build_pretraining_dataset(
            env_configs=env_configs,
            n_trajectories_per_env=config.get("n_trajectories_per_env", 20),
            n_iterations=config.get("n_iterations", 200),
            history_window=config.get("history_window", 50),
            save_path=data_path,
        )

    # Filter samples to those whose param_dim matches
    # (in a real setup, use separate JEPA per env, or pad/project params)
    samples = [s for s in samples if s["param_dim"] == first_env_param_dim]
    log.info(f"Filtered to {len(samples)} samples with param_dim={first_env_param_dim}")

    dataset = JEPAPretrainDataset(
        samples=samples,
        max_ctx_len=config.get("max_ctx_len", 50),
    )

    agent = JEPAAgent(
        param_dim=first_env_param_dim,
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        num_enc_layers=config.get("num_enc_layers", 3),
        num_pred_layers=config.get("num_pred_layers", 3),
        dim_ff=config.get("dim_ff", 256),
        ema_decay=config.get("ema_decay", 0.996),
        device=device,
    )

    # Cast numeric config values that YAML may read as strings
    for _k in ('lr', 'n_epochs', 'batch_size', 'n_trajectories_per_env', 'n_iterations', 'history_window', 'max_ctx_len'):
        if _k in config and isinstance(config[_k], str):
            config[_k] = float(config[_k]) if '.' in config[_k] or 'e' in config[_k].lower() else int(config[_k])

    pretrain_jepa(
        agent=agent,
        dataset=dataset,
        n_epochs=config.get("n_epochs", 50),
        batch_size=config.get("batch_size", 64),
        lr=config.get("lr", 3e-4),
        save_dir=save_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
