"""
JEPA Repair Model Pretraining
==============================
Trains the JEPARepairMatcher on the self-supervised corruption dataset.

After pretraining, the model is used as a drop-in replacement for
edit-distance in SP2 of the layered validator.
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Dict, List

from validator.jepa_repair_model import JEPARepairMatcher
from validator.repair_dataset import JEPARepairDataset

log = logging.getLogger(__name__)


def pretrain_jepa_repair(
    model: JEPARepairMatcher,
    dataset: JEPARepairDataset,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.1,
    save_dir: str = "checkpoints/jepa_repair",
    log_every: int = 50,
    device: str = "cpu",
) -> JEPARepairMatcher:
    """
    Pretrain the JEPA repair model on symbol-level corruption tasks.

    Loss: smooth_L1( predicted_latent, stop_grad(target_latent) )
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)
    model = model.to(device)

    # Train/val split
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    # Optimizer: exclude EMA target encoder
    params = [p for n, p in model.named_parameters()
              if "target_symbol_encoder" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(train_loader)
    )

    best_val_loss = float("inf")
    step = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            broken_ids  = batch["broken_ids"].to(device)
            context_ids = batch["context_ids"].to(device)
            correct_ids = batch["correct_ids"].to(device)

            loss = model.jepa_repair_loss(broken_ids, context_ids, correct_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            model.update_ema()

            epoch_loss += loss.item()
            step += 1

            if step % log_every == 0:
                log.info(f"[RepairPretrain] epoch={epoch+1} step={step} loss={loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                broken_ids  = batch["broken_ids"].to(device)
                context_ids = batch["context_ids"].to(device)
                correct_ids = batch["correct_ids"].to(device)
                val_loss += model.jepa_repair_loss(broken_ids, context_ids, correct_ids).item()
        val_loss /= max(1, len(val_loader))

        avg_train = epoch_loss / len(train_loader)
        log.info(f"[RepairPretrain] Epoch {epoch+1}/{n_epochs} "
                 f"train={avg_train:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"epoch_{epoch+1}.pt"))

    final_path = os.path.join(save_dir, "jepa_repair_pretrained.pt")
    torch.save(model.state_dict(), final_path)
    log.info(f"Repair pretraining complete → {final_path}")
    return model


def evaluate_repair_accuracy(
    model: JEPARepairMatcher,
    dataset: JEPARepairDataset,
    mode: str = "predict",
    confidence_threshold: float = 0.5,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate JEPA repair accuracy against ground-truth correct names.
    Reports: top-1 accuracy, repair rate, false-repair rate, by corruption type.
    """
    model.eval()
    results = {"correct": 0, "total": 0, "below_threshold": 0}
    by_type = {}

    for sample in dataset.samples:
        broken   = sample["broken_ref"]
        correct  = sample["correct_ref"]
        context  = sample["xml_context"]
        declared = sample["all_declared"]
        ctype    = sample.get("corruption_type", "unknown")

        pred, conf = model.match(
            broken_ref=broken,
            declared_names=declared,
            xml_context=context,
            mode=mode,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        results["total"] += 1
        if ctype not in by_type:
            by_type[ctype] = {"correct": 0, "total": 0, "skipped": 0}
        by_type[ctype]["total"] += 1

        if pred is None:
            results["below_threshold"] += 1
            by_type[ctype]["skipped"] += 1
        elif pred == correct:
            results["correct"] += 1
            by_type[ctype]["correct"] += 1

    n = results["total"]
    results["accuracy"]     = results["correct"] / n if n > 0 else 0.0
    results["repair_rate"]  = (n - results["below_threshold"]) / n if n > 0 else 0.0
    results["false_repair"] = (
        (n - results["below_threshold"] - results["correct"])
        / max(1, n - results["below_threshold"])
    )
    results["by_type"] = by_type
    return results
