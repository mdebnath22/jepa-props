"""
Ablation Experiment Runner
===========================
Reproduces the ablation table comparing:
  - Baseline (edit-distance SP2)
  - JEPA-SP2-match (latent nearest-neighbor)
  - JEPA-SP2-predict (predictive repair)

Metrics mirror Table 2 and Table 3 from the paper:
  - R1 recovery rate
  - False-repair rate  (graph cosine similarity < 0.70)
  - Partial-repair rate (0.70 ≤ sim < 0.90)
  - Correct-repair rate (sim ≥ 0.90)
  - Wall-clock time per document

Usage:
    python validator/ablation_runner.py --n_docs 500 --device cpu
    python validator/ablation_runner.py --n_docs 500 --ckpt checkpoints/jepa_repair/best.pt
"""

import os
import time
import argparse
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from validator.repair_dataset import (
    generate_synthetic_mjcf,
    generate_repair_samples_from_xml,
    corrupt_name,
)
from validator.layered_validator import (
    LayeredValidator,
    detect_errors,
    get_declared_joints,
    parse_xml_safe,
)
from validator.jepa_repair_model import JEPARepairMatcher
import random

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Actuator-graph similarity (mirrors paper's metric)
# ─────────────────────────────────────────────────────────────────────────────

def actuator_graph_similarity(xml_original: str, xml_repaired: str) -> float:
    """
    Compute cosine similarity between actuator-to-joint assignment graphs
    of the original and repaired documents.

    v(G)[i] = index of the joint assigned to motor i (in declared_joints order)
    sim = cosine(v(G_orig), v(G_repaired))
    """
    def get_assignment_vector(xml_str: str) -> np.ndarray:
        root = parse_xml_safe(xml_str)
        if root is None:
            return np.zeros(1)
        declared = get_declared_joints(root)
        joint_idx = {name: i for i, name in enumerate(declared)}
        motors = root.findall(".//actuator/motor")
        vec = []
        for m in motors:
            ref = m.get("joint", "")
            vec.append(float(joint_idx.get(ref, -1)))
        return np.array(vec) if vec else np.zeros(1)

    v1 = get_assignment_vector(xml_original)
    v2 = get_assignment_vector(xml_repaired)

    # Align lengths
    n = max(len(v1), len(v2))
    v1 = np.pad(v1, (0, n - len(v1)))
    v2 = np.pad(v2, (0, n - len(v2)))

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ─────────────────────────────────────────────────────────────────────────────
# Generate test documents with injected R1 errors
# ─────────────────────────────────────────────────────────────────────────────

def generate_r1_test_cases(
    n_docs: int = 200,
    joints_range: Tuple[int, int] = (3, 16),
    seed: int = 42,
) -> List[Dict]:
    """
    Generate test cases: each is a valid XML with one R1 error injected.
    Returns list of {'original_xml', 'corrupted_xml', 'broken_ref', 'correct_ref', 'task'}
    """
    rng = random.Random(seed)
    tasks = ["locomotion", "jumping", "swimming", "crawling", "pushing"]
    test_cases = []

    for i in range(n_docs):
        n_joints = rng.randint(*joints_range)
        task = rng.choice(tasks)
        orig_xml = generate_synthetic_mjcf(n_joints=n_joints, task=task, rng=rng)

        # Get motor references
        import re
        motor_matches = list(re.finditer(
            r'(<motor[^>]+joint=["\'])([^"\']+)(["\'])', orig_xml
        ))
        if not motor_matches:
            continue

        # Corrupt ONE random motor reference
        m = rng.choice(motor_matches)
        correct_ref = m.group(2)
        broken_ref = corrupt_name(correct_ref, rng)
        if broken_ref == correct_ref:
            continue

        corrupted_xml = orig_xml[:m.start(2)] + broken_ref + orig_xml[m.end(2):]

        # Verify the original is valid and corrupted has R1 error
        orig_errors = detect_errors(orig_xml)
        corr_errors = detect_errors(corrupted_xml)
        if orig_errors or "R1_JOINT_REF_MISMATCH" not in corr_errors:
            continue

        test_cases.append({
            "original_xml":  orig_xml,
            "corrupted_xml": corrupted_xml,
            "broken_ref":    broken_ref,
            "correct_ref":   correct_ref,
            "task":          task,
            "n_joints":      n_joints,
        })

    log.info(f"Generated {len(test_cases)} R1 test cases")
    return test_cases


# ─────────────────────────────────────────────────────────────────────────────
# Run one ablation condition
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(
    test_cases: List[Dict],
    sp2_mode: str,
    jepa_matcher=None,
    confidence_threshold: float = 0.5,
    delta: float = 0.6,
    device: str = "cpu",
) -> Dict:
    """Run all test cases under one SP2 condition and collect metrics."""
    correct = 0
    partial = 0
    wrong = 0
    below_threshold = 0
    total_time = 0.0

    for tc in test_cases:
        validator = LayeredValidator(
            sp2_mode=sp2_mode,
            jepa_matcher=jepa_matcher,
            edit_dist_delta=delta,
            jepa_confidence_threshold=confidence_threshold,
            task=tc["task"],
            device=device,
        )

        t0 = time.perf_counter()
        repaired_xml, success, repair_log = validator(tc["corrupted_xml"])
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        if not success:
            below_threshold += 1
            continue

        # Measure fidelity vs oracle
        sim = actuator_graph_similarity(tc["original_xml"], repaired_xml)

        if sim >= 0.90:
            correct += 1
        elif sim >= 0.70:
            partial += 1
        else:
            wrong += 1

    n = len(test_cases)
    n_attempted = n - below_threshold

    return {
        "total":             n,
        "below_threshold":   below_threshold,
        "correct":           correct,
        "partial":           partial,
        "wrong":             wrong,
        "recovery_rate":     n_attempted / n if n > 0 else 0.0,
        "correct_rate":      correct / n if n > 0 else 0.0,
        "false_repair_rate": wrong / max(1, n_attempted),
        "avg_time_ms":       1000.0 * total_time / n if n > 0 else 0.0,
        # F1 matching Table 1 in paper
        "f1": (
            2 * (n_attempted / n) * (correct / max(1, n_attempted))
            / max(1e-9, (n_attempted / n) + (correct / max(1, n_attempted)))
        ) if n > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print results table
# ─────────────────────────────────────────────────────────────────────────────

def print_ablation_table(results: Dict[str, Dict]):
    """Print Table matching paper format (Table 1 / Table 2 style)."""
    print()
    print("=" * 90)
    print("ABLATION: SP2 Repair Method Comparison")
    print("Metric definitions match paper: recovery = repair attempted / total;")
    print("false-repair = wrong / attempted; F1 = harmonic mean(recovery, correct_rate)")
    print("=" * 90)
    header = f"{'Method':<22} {'Recovery':>10} {'Correct':>10} {'Partial':>9} {'FalseRep':>10} {'F1':>8} {'ms/doc':>8}"
    print(header)
    print("-" * 90)
    for method, r in results.items():
        print(
            f"{method:<22} "
            f"{r['recovery_rate']:>9.1%} "
            f"{r['correct_rate']:>9.1%} "
            f"{r['partial']/r['total']:>8.1%} "
            f"{r['false_repair_rate']:>9.1%} "
            f"{r['f1']:>7.3f} "
            f"{r['avg_time_ms']:>7.2f}"
        )
    print("=" * 90)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type=int, default=200)
    parser.add_argument("--n_joints_min", type=int, default=3)
    parser.add_argument("--n_joints_max", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to pretrained JEPA repair checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_jepa", action="store_true",
                        help="Run only edit-distance baseline (no JEPA needed)")
    args = parser.parse_args()

    # Generate test cases
    test_cases = generate_r1_test_cases(
        n_docs=args.n_docs,
        joints_range=(args.n_joints_min, args.n_joints_max),
        seed=args.seed,
    )

    if not test_cases:
        log.error("No test cases generated. Check synthetic XML generation.")
        return

    # Load JEPA repair model if checkpoint provided
    jepa_matcher = None
    if not args.skip_jepa:
        jepa_matcher = JEPARepairMatcher(d_model=64, nhead=4)
        if args.ckpt and os.path.exists(args.ckpt):
            log.info(f"Loading JEPA repair checkpoint: {args.ckpt}")
            jepa_matcher.load_state_dict(
                torch.load(args.ckpt, map_location=args.device), strict=False
            )
        else:
            log.warning(
                "No checkpoint provided — JEPA matcher uses random weights. "
                "Run validator/jepa_repair_pretrain.py first for meaningful results."
            )
        jepa_matcher.eval()

    # Run ablation conditions
    results = {}

    log.info("Running: edit-distance baseline (δ=0.4)")
    results["EditDist (δ=0.4)"] = run_condition(
        test_cases, "editdist", delta=0.4, device=args.device
    )

    log.info("Running: edit-distance baseline (δ=0.6, paper default)")
    results["EditDist (δ=0.6)"] = run_condition(
        test_cases, "editdist", delta=0.6, device=args.device
    )

    if not args.skip_jepa and jepa_matcher is not None:
        log.info("Running: JEPA-SP2-match (latent nearest-neighbor)")
        results["JEPA-SP2-match"] = run_condition(
            test_cases, "jepa-match", jepa_matcher=jepa_matcher,
            confidence_threshold=0.5, device=args.device
        )

        log.info("Running: JEPA-SP2-predict (predictive repair)")
        results["JEPA-SP2-predict"] = run_condition(
            test_cases, "jepa-predict", jepa_matcher=jepa_matcher,
            confidence_threshold=0.5, device=args.device
        )

        # Threshold sensitivity for JEPA (mirrors Table 1 in paper)
        log.info("Running: JEPA-SP2-predict threshold sweep")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            results[f"JEPA-pred (τ={thresh})"] = run_condition(
                test_cases, "jepa-predict", jepa_matcher=jepa_matcher,
                confidence_threshold=thresh, device=args.device
            )

    print_ablation_table(results)

    # Save results
    import json
    out_path = "results/ablation_sp2.json"
    os.makedirs("results", exist_ok=True)
    # Convert to serializable
    serial = {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                  for kk, vv in v.items()} for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(serial, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
