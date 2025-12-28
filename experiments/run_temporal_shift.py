#!/usr/bin/env python3
"""
Temporal Shift Experiment for ConformalDrift Paper
Tests conformal coverage under temporal distribution shift (FastAPI v1 -> v2)

This experiment validates Table 2 of the ConformalDrift paper:
- Setting 1: FastAPI v1 -> v1 (no shift, baseline)
- Setting 2: FastAPI v1 -> v2 (temporal shift)
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.core_algorithm import (
    ConformalRAGGuardrails,
    CRGConfig,
    RAGExample
)


@dataclass
class TemporalShiftResult:
    """Results from a temporal shift experiment."""
    setting: str
    n_calibration: int
    n_test: int
    target_coverage: float
    effective_coverage: float
    coverage_drop: float
    false_positive_rate: float
    threshold: float
    status: str  # PASS or FAIL


def load_fastapi_drift_tasks(data_path: str) -> Dict:
    """Load FastAPI drift tasks from DriftBench."""
    with open(data_path, 'r') as f:
        return json.load(f)


def create_v1_examples(tasks: List[Dict], include_drift: bool = True) -> List[RAGExample]:
    """
    Create RAGExample instances for V1 calibration.

    For calibration, we use hallucinated examples (label=1):
    - Query + V1 evidence but WRONG answer -> hallucination

    We create synthetic hallucinations by using V2 answers with V1 evidence.
    """
    examples = []

    for task in tasks:
        # Faithful example: V1 evidence + V1 answer
        examples.append(RAGExample(
            query=task['question'],
            documents=[task['evidence_v1']],
            response=task['answer_v1'],
            label=0  # faithful
        ))

        if include_drift:
            # Hallucinated example: V1 evidence + V2 answer (wrong for V1)
            examples.append(RAGExample(
                query=task['question'],
                documents=[task['evidence_v1']],
                response=task['answer_v2'],
                label=1  # hallucination (answer doesn't match V1 evidence)
            ))

    return examples


def create_v1_to_v1_test(tasks: List[Dict]) -> List[RAGExample]:
    """
    Create test set for V1 -> V1 (no shift).
    Ground truth is V1 knowledge.
    """
    examples = []

    for task in tasks:
        # Faithful: V1 evidence + V1 answer
        examples.append(RAGExample(
            query=task['question'],
            documents=[task['evidence_v1']],
            response=task['answer_v1'],
            label=0
        ))

        # Hallucinated: V1 evidence + wrong answer
        wrong_answer = "This information is not available in the documentation."
        examples.append(RAGExample(
            query=task['question'],
            documents=[task['evidence_v1']],
            response=wrong_answer,
            label=1
        ))

    return examples


def create_v1_to_v2_test(tasks: List[Dict]) -> List[RAGExample]:
    """
    Create test set for V1 -> V2 (temporal shift).

    Key insight: When knowledge drifts, the V2 answer becomes correct,
    but our V1-calibrated model still uses V1 evidence.

    - V2 answer with V1 evidence: Looks like hallucination to V1 model
      (the answer contradicts V1 documentation)
    - V1 answer with V1 evidence: Looks faithful but is actually outdated
    """
    examples = []

    for task in tasks:
        # Under temporal shift, V2 answers are now the ground truth
        # But V1 evidence doesn't support them -> detected as hallucination
        examples.append(RAGExample(
            query=task['question'],
            documents=[task['evidence_v1']],
            response=task['answer_v2'],
            label=1  # V2 answer not supported by V1 evidence
        ))

        # V1 answers are still supported by V1 evidence
        # but they're outdated (false negatives under drift)
        examples.append(RAGExample(
            query=task['question'],
            documents=[task['evidence_v1']],
            response=task['answer_v1'],
            label=0  # Still looks faithful to V1 model
        ))

    return examples


def run_temporal_shift_experiment(
    data_path: str,
    target_coverage: float = 0.95,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
) -> Dict:
    """Run the complete temporal shift experiment."""

    print("=" * 70)
    print("ConformalDrift: Temporal Shift Experiment")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Target Coverage: {target_coverage:.1%}")
    print(f"Device: {device}")
    print()

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("Loading FastAPI drift tasks...")
    drift_data = load_fastapi_drift_tasks(data_path)
    tasks = drift_data['tasks']
    print(f"  Loaded {len(tasks)} drift tasks")

    # Split tasks for calibration and testing
    n_cal = int(len(tasks) * 0.5)
    cal_tasks = tasks[:n_cal]
    test_tasks = tasks[n_cal:]

    print(f"  Calibration tasks: {n_cal}")
    print(f"  Test tasks: {len(test_tasks)}")

    # Initialize model
    print("\nInitializing CRG model...")
    config = CRGConfig(
        device=device,
        use_fp16=True,
        use_weighted_rad=True,
        use_sentence_level_sec=True,
        grounding_threshold=0.3,
        temperature=0.1,
    )
    model = ConformalRAGGuardrails(config)

    # Create calibration data (hallucinated examples from V1)
    print("\nCreating calibration examples...")
    cal_examples = create_v1_examples(cal_tasks, include_drift=True)
    cal_hallucinated = [ex for ex in cal_examples if ex.label == 1]
    print(f"  Calibration hallucinated examples: {len(cal_hallucinated)}")

    # Calibrate on V1 hallucinated examples
    print("\nCalibrating on V1 data...")
    alpha = 1 - target_coverage
    model.calibrate(cal_hallucinated, alpha=alpha)
    threshold = model._threshold
    print(f"  Threshold: {threshold:.4f}")

    results = {
        'experiment': 'temporal_shift',
        'data_source': 'fastapi_drift_tasks.json',
        'n_tasks': len(tasks),
        'target_coverage': target_coverage,
        'settings': []
    }

    # Setting 1: V1 -> V1 (no shift)
    print("\n" + "-" * 70)
    print("Setting 1: FastAPI v1 -> v1 (no shift)")
    print("-" * 70)

    v1_test = create_v1_to_v1_test(test_tasks)
    v1_scores = model.compute_individual_scores(v1_test)
    v1_ensemble = model.compute_ensemble_score(v1_scores)
    v1_labels = torch.tensor([ex.label for ex in v1_test])

    v1_predictions = (v1_ensemble >= threshold).float().cpu()
    v1_labels_cpu = v1_labels.cpu()

    v1_hall_mask = v1_labels_cpu == 1
    v1_faith_mask = v1_labels_cpu == 0

    v1_coverage = (v1_predictions[v1_hall_mask] == 1).float().mean().item() if v1_hall_mask.sum() > 0 else 0
    v1_fpr = (v1_predictions[v1_faith_mask] == 1).float().mean().item() if v1_faith_mask.sum() > 0 else 0
    v1_drop = (target_coverage - v1_coverage) * 100  # As percentage points

    v1_status = "PASS" if v1_coverage >= target_coverage - 0.05 else "FAIL"

    print(f"  Coverage: {v1_coverage:.1%} (target: {target_coverage:.1%})")
    print(f"  Coverage Drop: {v1_drop:+.1f}%")
    print(f"  FPR: {v1_fpr:.1%}")
    print(f"  Status: {v1_status}")

    results['settings'].append({
        'name': 'FastAPI v1 -> v1',
        'shift_type': 'none',
        'n_test': len(v1_test),
        'effective_coverage': round(v1_coverage * 100, 1),
        'coverage_drop': round(v1_drop, 1),
        'fpr': round(v1_fpr * 100, 0),
        'threshold': round(threshold, 4),
        'status': v1_status
    })

    # Setting 2: V1 -> V2 (temporal shift)
    print("\n" + "-" * 70)
    print("Setting 2: FastAPI v1 -> v2 (temporal shift)")
    print("-" * 70)

    v2_test = create_v1_to_v2_test(test_tasks)
    v2_scores = model.compute_individual_scores(v2_test)
    v2_ensemble = model.compute_ensemble_score(v2_scores)
    v2_labels = torch.tensor([ex.label for ex in v2_test])

    v2_predictions = (v2_ensemble >= threshold).float().cpu()
    v2_labels_cpu = v2_labels.cpu()

    v2_hall_mask = v2_labels_cpu == 1
    v2_faith_mask = v2_labels_cpu == 0

    v2_coverage = (v2_predictions[v2_hall_mask] == 1).float().mean().item() if v2_hall_mask.sum() > 0 else 0
    v2_fpr = (v2_predictions[v2_faith_mask] == 1).float().mean().item() if v2_faith_mask.sum() > 0 else 0
    v2_drop = (target_coverage - v2_coverage) * 100  # As percentage points

    v2_status = "PASS" if v2_coverage >= target_coverage - 0.05 else "FAIL"

    print(f"  Coverage: {v2_coverage:.1%} (target: {target_coverage:.1%})")
    print(f"  Coverage Drop: {v2_drop:+.1f}%")
    print(f"  FPR: {v2_fpr:.1%}")
    print(f"  Status: {v2_status}")

    results['settings'].append({
        'name': 'FastAPI v1 -> v2',
        'shift_type': 'temporal',
        'n_test': len(v2_test),
        'effective_coverage': round(v2_coverage * 100, 1),
        'coverage_drop': round(v2_drop, 1),
        'fpr': round(v2_fpr * 100, 0),
        'threshold': round(threshold, 4),
        'status': v2_status
    })

    # Calculate recalibration cost
    print("\n" + "-" * 70)
    print("Recalibration Analysis")
    print("-" * 70)

    # What threshold would be needed for V2 to maintain coverage?
    v2_hall_scores = v2_ensemble[v2_hall_mask].cpu().numpy()
    if len(v2_hall_scores) > 0:
        required_quantile = target_coverage
        required_threshold = np.percentile(v2_hall_scores, (1 - required_quantile) * 100)
        threshold_drift = abs(required_threshold - threshold)

        print(f"  Original threshold: {threshold:.4f}")
        print(f"  Required threshold for V2: {required_threshold:.4f}")
        print(f"  Threshold drift: {threshold_drift:.4f} ({threshold_drift/threshold*100:.1f}% relative)")

        results['recalibration'] = {
            'original_threshold': round(float(threshold), 4),
            'required_threshold_v2': round(float(required_threshold), 4),
            'threshold_drift_absolute': round(float(threshold_drift), 4),
            'threshold_drift_relative_pct': round(float(threshold_drift/threshold*100), 1)
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Table 2 Validation")
    print("=" * 70)
    print(f"{'Setting':<25} {'Cov_eff':>10} {'dCov':>10} {'FPR':>8} {'Status':>8}")
    print("-" * 70)

    for setting in results['settings']:
        print(f"{setting['name']:<25} {setting['effective_coverage']:>9.1f}% {setting['coverage_drop']:>+9.1f}% {setting['fpr']:>7.0f}% {setting['status']:>8}")

    results['timestamp'] = datetime.now().isoformat()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run temporal shift experiment")
    parser.add_argument(
        "--data",
        type=str,
        default=r"C:\Users\dsinh\research-papers\driftbench\data\fastapi_drift_tasks.json",
        help="Path to FastAPI drift tasks JSON"
    )
    parser.add_argument("--coverage", type=float, default=0.95, help="Target coverage")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    results = run_temporal_shift_experiment(
        data_path=args.data,
        target_coverage=args.coverage,
        device=args.device,
        seed=args.seed
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = args.output or (output_dir / "temporal_shift_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
