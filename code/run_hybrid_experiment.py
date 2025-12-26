#!/usr/bin/env python3
"""
Experiment: Test Hybrid Detector on HaluEval

Runs locally on RTX 3090 first. If OOM, consider cloud.
Budget: $200 max for cloud if needed.
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.core_algorithm import (
    ConformalRAGGuardrails,
    CRGConfig,
    RAGExample
)

# Import HybridDetector from new module
try:
    from src.models.hybrid_detector import HybridDetector, HybridConfig
    HYBRID_AVAILABLE = True
    print("HybridDetector loaded successfully")
except ImportError as e:
    HYBRID_AVAILABLE = False
    print(f"Warning: HybridDetector not available: {e}")

from datasets import load_dataset


def load_halueval(n_samples=800, seed=42):
    """Load HaluEval test data."""
    np.random.seed(seed)
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    examples = []
    for item in dataset:
        if len(examples) >= n_samples:
            break
        question = item.get('question', '')
        knowledge = item.get('knowledge', '')
        answer = item.get('answer', '')
        is_hall = item.get('hallucination', 'no').lower() == 'yes'

        if question and knowledge and answer:
            examples.append(RAGExample(
                query=question,
                documents=[knowledge],
                response=answer,
                label=1 if is_hall else 0
            ))

    np.random.shuffle(examples)
    return examples


def run_experiment():
    """Run the hybrid detector experiment."""
    print("="*60)
    print("HYBRID DETECTOR EXPERIMENT")
    print("="*60)

    # Load data
    print("\nLoading HaluEval...")
    examples = load_halueval(n_samples=800)

    # Split
    n_cal = int(len(examples) * 0.5)
    cal_data = examples[:n_cal]
    test_data = examples[n_cal:]

    cal_hall = [e for e in cal_data if e.label == 1]
    test_labels = torch.tensor([e.label for e in test_data])

    print(f"Calibration: {len(cal_hall)} hallucinated")
    print(f"Test: {len(test_data)} total")

    # Initialize models
    config = CRGConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    crg = ConformalRAGGuardrails(config)

    results = {"timestamp": datetime.now().isoformat(), "experiments": {}}

    # Baseline CRG
    print("\n--- Baseline CRG ---")
    crg.calibrate(cal_hall, alpha=0.05)

    test_scores = crg.compute_individual_scores(test_data)
    test_ensemble = crg.compute_ensemble_score(test_scores)
    predictions = (test_ensemble >= crg._threshold).float().cpu()

    hall_mask = test_labels == 1
    faith_mask = test_labels == 0

    baseline_coverage = (predictions[hall_mask] == 1).float().mean().item()
    baseline_fpr = (predictions[faith_mask] == 1).float().mean().item()
    baseline_acc = (predictions == test_labels.float()).float().mean().item()

    print(f"Coverage: {baseline_coverage:.1%}")
    print(f"FPR: {baseline_fpr:.1%}")
    print(f"Accuracy: {baseline_acc:.1%}")

    results["experiments"]["baseline_crg"] = {
        "coverage": baseline_coverage,
        "fpr": baseline_fpr,
        "accuracy": baseline_acc
    }

    # Hybrid detector (if available)
    if HYBRID_AVAILABLE:
        print("\n--- Hybrid Detector ---")
        try:
            hybrid_config = HybridConfig(
                crg_weight=0.3,
                contradiction_weight=0.4,
                entity_weight=0.3
            )
            hybrid = HybridDetector(config, hybrid_config)

            # Calibrate on hallucinated examples
            print("Calibrating hybrid detector...")
            start_time = time.time()
            cal_hybrid_scores = []
            for ex in tqdm(cal_hall, desc="Calibrating"):
                score = hybrid.score(ex)
                cal_hybrid_scores.append(score)

            cal_hybrid_scores = torch.tensor(cal_hybrid_scores)
            sorted_scores = torch.sort(cal_hybrid_scores)[0]
            n = len(sorted_scores)
            alpha = 0.05
            quantile_idx = min(int(np.ceil((n + 1) * alpha)), n) - 1
            threshold = sorted_scores[quantile_idx].item()
            hybrid._threshold = threshold
            hybrid._calibrated = True

            cal_time = time.time() - start_time
            print(f"Calibration time: {cal_time:.1f}s")

            # Get hybrid scores on test set
            print("Scoring test set...")
            start_time = time.time()
            hybrid_scores = []
            score_details = []
            for ex in tqdm(test_data, desc="Hybrid scoring"):
                details = hybrid.compute_hybrid_score(ex)
                hybrid_scores.append(details['combined'])
                score_details.append(details)

            hybrid_scores = torch.tensor(hybrid_scores)
            eval_time = time.time() - start_time
            print(f"Evaluation time: {eval_time:.1f}s")

            hybrid_preds = (hybrid_scores >= threshold).float()

            hybrid_coverage = (hybrid_preds[hall_mask] == 1).float().mean().item()
            hybrid_fpr = (hybrid_preds[faith_mask] == 1).float().mean().item()
            hybrid_acc = (hybrid_preds == test_labels.float()).float().mean().item()

            print(f"Coverage: {hybrid_coverage:.1%}")
            print(f"FPR: {hybrid_fpr:.1%}")
            print(f"Accuracy: {hybrid_acc:.1%}")

            # Analyze score components
            hall_details = [d for d, l in zip(score_details, test_labels.tolist()) if l == 1]
            faith_details = [d for d, l in zip(score_details, test_labels.tolist()) if l == 0]

            if hall_details and faith_details:
                print("\nScore component analysis:")
                for component in ['crg', 'contradiction', 'entity_mismatch']:
                    hall_mean = np.mean([d[component] for d in hall_details])
                    faith_mean = np.mean([d[component] for d in faith_details])
                    gap = hall_mean - faith_mean
                    print(f"  {component}: hall={hall_mean:.3f}, faith={faith_mean:.3f}, gap={gap:+.3f}")

            results["experiments"]["hybrid"] = {
                "coverage": hybrid_coverage,
                "fpr": hybrid_fpr,
                "accuracy": hybrid_acc,
                "threshold": threshold,
                "cal_time_sec": cal_time,
                "eval_time_sec": eval_time
            }

            # Improvement
            fpr_improvement = baseline_fpr - hybrid_fpr
            print(f"\nFPR Improvement: {fpr_improvement:.1%}")
            results["fpr_improvement"] = fpr_improvement
            results["target_met"] = hybrid_fpr < 0.30 and hybrid_coverage > 0.90

        except Exception as e:
            import traceback
            print(f"Hybrid experiment failed: {e}")
            traceback.print_exc()
            results["experiments"]["hybrid"] = {"error": str(e)}

    # Save results
    output_path = Path("outputs/hybrid_experiment_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_experiment()
