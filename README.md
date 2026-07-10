# JEPA-ProPS: Prompted Policy Search with JEPA

A **JEPA-based replacement for the LLM** in [ProPS (NeurIPS 2025)](https://props-llm.github.io/), 
implementing the same policy search framework but using a 
**Joint Embedding Predictive Architecture** instead of GPT-4o / Gemini.

---

## Overview

ProPS uses an LLM as an in-context optimizer:
```
history {(θ₁,r₁),...,(θₜ,rₜ)} → LLM prompt → θₜ₊₁
```

JEPA-ProPS replaces the LLM call with a learned neural predictor:
```
history {(θ₁,r₁),...,(θₜ,rₜ)} → Context Encoder (Transformer) → latent s_x
                                → Predictor MLP (conditioned on r*) → ŝ_y
                                → Decoder MLP → θₜ₊₁
```

**Three variants:**

| Variant | Description | Analog in ProPS |
|---|---|---|
| **Pure JEPA** | JEPA proposes θ with no LLM | ProPS |
| **JEPA+Semantic** | Sentence-transformer encodes task once; conditions JEPA | ProPS+ |
| **Offline Pretrain** | Pretrain JEPA on random rollout data, then adapt online | Fine-tuned Qwen |

---

## Codebase Structure

```
jepa_props/
├── main.py                        # Entry point (mirrors ProPS main.py)
├── requirements.txt
├── agent/
│   ├── jepa_context_encoder.py    # Transformer encoder over (θ,r) history
│   ├── jepa_predictor.py          # Predictor MLP + Target Encoder (EMA) + Decoder
│   ├── jepa_agent.py              # Full agent: history buffer + propose()
│   └── jepa_pretrain.py           # Offline pretraining pipeline
├── runner/
│   ├── jepa_runner.py             # Pure JEPA policy search loop
│   └── jepa_semantics_runner.py   # LLM+JEPA hybrid loop
├── configs/
│   ├── cartpole_jepa.yaml         # Pure JEPA, CartPole
│   ├── cartpole_jepa_hybrid.yaml  # Hybrid, CartPole
│   ├── swimmer_jepa.yaml          # Pure JEPA, Swimmer
│   └── pretrain_jepa.yaml         # Offline pretraining
├── analysis/
│   └── compare_methods.py         # JEPA vs CEM vs Random comparison
└── scripts/
    ├── run_pretrain.sh             # SLURM: pretraining
    ├── run_benchmark_array.sh      # SLURM: all 15 envs × 10 seeds (array job)
    └── run_compare.sh              # SLURM: comparison for one env
```

---

## Installation

```bash
conda activate mdebnath
pip install -r requirements.txt

# For MuJoCo environments:
pip install mujoco gymnasium[mujoco]
```

---

## Running

### Step 1 (Optional): Offline Pretraining

Pretrain the JEPA on diverse rollout data across multiple environments.
This is analogous to how ProPS benefits from the LLM's pretraining.

```bash
# Locally
python main.py --config configs/pretrain_jepa.yaml

# On ASU HPC
sbatch scripts/run_pretrain.sh
```

### Step 2: Policy Search

```bash
# Pure JEPA (no LLM, no API cost)
python main.py --config configs/cartpole_jepa.yaml

# LLM+JEPA Hybrid (one sentence-transformer call per task)
python main.py --config configs/cartpole_jepa_hybrid.yaml

# With pretrained weights
# Edit config: pretrained_ckpt: checkpoints/jepa_pretrained/jepa_pretrained.pt
python main.py --config configs/swimmer_jepa.yaml
```

### Step 3: Comparison with Baselines

```bash
# Compare JEPA vs CEM vs Random Search
python analysis/compare_methods.py --env CartPole-v1 --seeds 0 1 2 3 4

# On HPC (all 15 ProPS envs × 10 seeds)
sbatch scripts/run_benchmark_array.sh
```

---

## JEPA Architecture Details

### Context Encoder (`jepa_context_encoder.py`)
- **Input**: sequence of `(θ_i, r_i)` tokens, variable length T
- **Architecture**: causal Transformer encoder, Pre-LN, mean-pool output
- **Output**: latent `s_x ∈ ℝ^{d_model}`

### Predictor (`jepa_predictor.py`)
- **Input**: `(s_x, r*)` where `r*` is the desired reward (P80 of seen rewards)
- **Architecture**: MLP with GELU activations
- **Output**: predicted target latent `ŝ_y ∈ ℝ^{d_model}`
- This is **goal-conditioned**: by conditioning on a high `r*`, we predict the latent of a good policy

### Target Encoder (`jepa_predictor.py`)
- **Input**: single policy vector `θ`
- **Architecture**: small MLP with EMA weights (never receives gradients directly)
- **Output**: target latent `s_y ∈ ℝ^{d_model}`
- EMA decay τ = 0.996 (from I-JEPA)

### JEPA Loss
```
L = smooth_L1( ŝ_y ,  stop_grad(s_y) )
```
Latent-only loss — **no reconstruction in parameter space** during pretraining.

### Decoder (`jepa_predictor.py`)
- Trained with auxiliary reconstruction loss: `θ → latent → θ̂ ≈ θ`
- Used at inference to convert predicted latent `ŝ_y → θ_proposed`

---

## Key Design Decisions vs ProPS

| Decision | ProPS | JEPA-ProPS |
|---|---|---|
| History representation | Text tokens | Continuous (θ,r) sequence |
| Optimization reasoning | LLM in-context | Latent-space prediction |
| Semantic grounding | Full prompt every iter | Sentence embedding once per task |
| Exploration | LLM proposes one candidate | Sample n_candidates with noise |
| Cost per iteration | ~$0.01 (GPT-4o) | ~0.001s (CPU inference) |
| Context length limit | ~128K tokens | Any length (Transformer) |
| Parallelizable | No (sequential LLM) | Yes |

---

## Expected Results

Based on the JEPA architecture and ProPS results, we expect:

- **Swimmer**: JEPA should match or exceed ProPS (purely numerical task, no semantic benefit)
- **CartPole**: JEPA+Semantic should match ProPS+ (benefits from constraint knowledge)
- **Walker/Hopper**: JEPA may underperform ProPS+ (complex dynamics, semantic hints help)
- **FrozenLake/CliffWalking**: unclear — discrete state space may favor tabular approaches

---

## BibTeX

If you build on this, cite ProPS:
```bibtex
@inproceedings{zhou2025props,
    title     = {Prompted Policy Search: Reinforcement Learning through Linguistic and Numerical Reasoning in LLMs},
    author    = {Zhou, Yifan and Grover, Sachin and El Mistiri, Mohamed and ...},
    booktitle = {NeurIPS 2025},
    year      = {2025},
}
```
