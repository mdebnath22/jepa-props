#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM: Pretrain JEPA repair model + run SP2 ablation
# Run BEFORE any combined experiments.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=jepa_repair_pretrain
#SBATCH --partition=htc
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/jepa_repair_%j.out
#SBATCH --error=logs/jepa_repair_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mroycho1@asu.edu

module purge
module load mamba/latest
conda activate mdebnath

cd /scratch/mroycho1/jepa_props
mkdir -p logs checkpoints/jepa_repair data results

echo "=== JEPA Repair Model Pretraining ==="
echo "Start: $(date)"

# Step 1: Pretrain the JEPA repair model on synthetic XMLs
python -c "
import sys, logging
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, '.')

from validator.jepa_repair_model import JEPARepairMatcher
from validator.repair_dataset import JEPARepairDataset
from validator.jepa_repair_pretrain import pretrain_jepa_repair, evaluate_repair_accuracy

# Build synthetic dataset
print('Building synthetic dataset...')
dataset = JEPARepairDataset.from_synthetic(
    n_xmls=2000,
    joints_range=(3, 16),
    n_samples_per_xml=15,
    seed=42,
)
print(f'Dataset size: {len(dataset)} samples')

# Build model
model = JEPARepairMatcher(
    d_model=64,
    char_embed_dim=16,
    n_filters=64,
    nhead=4,
    num_context_layers=2,
    num_pred_layers=3,
    ema_decay=0.996,
)

# Pretrain
model = pretrain_jepa_repair(
    model=model,
    dataset=dataset,
    n_epochs=50,
    batch_size=256,
    lr=3e-4,
    save_dir='checkpoints/jepa_repair',
    device='cuda',
)

# Evaluate on held-out set
eval_ds = JEPARepairDataset.from_synthetic(
    n_xmls=200, joints_range=(3, 16), n_samples_per_xml=10, seed=999,
)
for mode in ['match', 'predict']:
    results = evaluate_repair_accuracy(model, eval_ds, mode=mode, device='cuda')
    print(f'Mode={mode}: accuracy={results[\"accuracy\"]:.3f} '
          f'recovery={results[\"repair_rate\"]:.3f} '
          f'false_repair={results[\"false_repair\"]:.3f}')
    for ctype, r in results['by_type'].items():
        n = r['total']
        if n > 0:
            print(f'  {ctype}: correct={r[\"correct\"]/n:.2%} ({n} samples)')
"

echo "Pretraining done: $(date)"

# Step 2: Run SP2 ablation experiment
echo "=== SP2 Ablation ==="
python validator/ablation_runner.py \
    --n_docs 500 \
    --ckpt checkpoints/jepa_repair/best.pt \
    --device cuda \
    --seed 42

echo "Ablation done: $(date)"
